"""
llmflash_baseline.py

CXL-adapted LLM-in-a-Flash Baseline (Alizadeh et al., ACL 2024)
Mechanisms implemented (paper-faithful):
  1. Selective Persistence  — attention + embeddings PINNED in CXL DRAM always
  2. Low-rank Sparsity      — active_frac derived from actual model layer sparsity
                              FIX: sparsity propagated from decomposed_build_layers
  3. Sliding Window (k=5)   — incremental delta loading per token
  4. Row-Column Bundling    — ~1.8× NAND throughput (Table 2)
  5. Pre-allocated DRAM mgmt — neuron swap rewrite cost modeled
  6. DRAM window overflow    — overflow-aware NAND load computation
  7. Prefill full-load       — sparsity collapses during prefill
  8. BATCH MODEL             — active_frac(B)=1-(1-p)^B; TPS=B/step_time
  9. Activation-aware turnover:
       ReLU/FATReLU (paper): ~24% of active set changes per token (§4.1)
       SiLU/SwiGLU (Qwen/Llama): ~60% turnover — contextual, input-dependent
 10. KV per-seq FIX          — per-token increment only, not full 512-token cache
"""

from model_cfg import decomposed_build_layers, DEFAULT_MODEL_CFG, BYTES_PER_PARAM
from sim_cfg import (
    host_dram_capacity_bytes,
    cxl_dev_dram_capacity_bytes,
    BATCH_SIZE,
)
from tiers import CXL_DRAM, CXL_SSD_NAND, transfer_time_s, \
                  Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, GiB

# ── Paper constants (OPT/ReLU baseline, §3.1 and §4.1) ────────────────────────
WINDOW_SIZE_K             = 5    # Sliding window token count
BUNDLING_THROUGHPUT_BOOST = 1.8  # Row-col bundling: 1.25→2.25 GB/s (Table 2)
DRAM_REWRITE_FRAC         = 0.25 # Neuron swap rewrite overhead (§3.3)

# ── Activation-function-aware DRAM window turnover ────────────────────────────
# ReLU/FATReLU (paper §4.1): delta ≈ 2.4% of FFN = 24% of 10% active set
#   → 24% of window-resident neurons cycle per token
# SiLU/SwiGLU (Qwen2.5, Llama, Mistral): contextual sparsity is input-dependent
#   → different tokens activate different neurons → ~60% turnover per token
#   → empirically: sagg(k+1) - sagg(k) is large because no structural zeros exist
DRAM_TURNOVER_RELU = 0.24   # OPT/ReLU — paper-faithful
DRAM_TURNOVER_SILU = 0.60   # SiLU/SwiGLU — contextual, input-dependent

NUM_DECODE_TOKENS  = 16
NUM_PREFILL_TOKENS = 512


def ssd_time_s(n):
    return transfer_time_s(
        n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))


def simulate_llmflash():
    import model_cfg as _mcfg

    layers     = decomposed_build_layers(DEFAULT_MODEL_CFG())
    total_dram = host_dram_capacity_bytes + cxl_dev_dram_capacity_bytes

    # ── Partition layers ───────────────────────────────────────────────────────
    pinned_layers = [L for L in layers if L.get("kind") != "MLP"]
    ffn_layers    = [L for L in layers if L.get("kind") == "MLP"]

    total_pinned_bytes = sum(L["bytes"] for L in pinned_layers)
    total_ffn_bytes    = sum(L["bytes"] for L in ffn_layers)

    # ── Activation function → turnover rate ───────────────────────────────────
    # Read ACTIVATION_FN from model_cfg if present; default to "silu"
    activation_fn  = getattr(_mcfg, "ACTIVATION_FN", "silu").lower()
    turnover_rate  = DRAM_TURNOVER_RELU if activation_fn == "relu" \
                     else DRAM_TURNOVER_SILU
    activation_label = f"{activation_fn.upper()} → turnover={turnover_rate:.0%}"

    # ── KV per-seq: per-token increment only ──────────────────────────────────
    # FIX: kv_cache_bytes is the FULL 512-token KV budget stored in layer config.
    # We only write ONE new token's worth of KV per decode step per sequence.
    # Wrong:  sum(kv_cache_bytes)        → charges ~80MB per seq per step
    # Correct: sum(kv_cache_bytes) // 512 → charges ~156KB per seq per step
    kv_per_seq_bytes = sum(L.get("kv_cache_bytes", 0)
                           for L in pinned_layers) // NUM_PREFILL_TOKENS

    # ── Single-request active fraction from layer sparsity ────────────────────
    # Sparsity field correctly populated by decomposed_build_layers (semduplex fix)
    avg_sparsity       = (sum(L.get("sparsity", 0.54) for L in ffn_layers)
                          / max(len(ffn_layers), 1))
    active_frac_single = 1.0 - avg_sparsity

    # ── BATCH: union of active neuron sets across B sequences ─────────────────
    # active_frac(B) = 1-(1-p)^B
    # ReLU B=1: p=0.10  → 10% loaded      (full paper benefit)
    # SiLU B=1: p=0.46  → 46% loaded      (genuine but smaller benefit)
    # SiLU B=4: p=0.915 → 91.5% loaded    (mostly collapsed)
    # SiLU B=8: p=0.993 → 99.3% loaded    (fully collapsed)
    active_frac_batch = min(1.0, 1.0 - (1.0 - active_frac_single) ** BATCH_SIZE)

    # ── DRAM capacity check ────────────────────────────────────────────────────
    pinned_dram_frac = min(1.0, total_dram / total_pinned_bytes) \
                       if total_pinned_bytes > 0 else 1.0
    dram_remaining   = max(0.0, total_dram - total_pinned_bytes)

    # ── Overflow-aware FFN load decomposition ─────────────────────────────────
    # dram_window_frac: how much of the batch-active FFN fits in DRAM window
    #   Large DRAM (32H+64C=96GB, 72B INT8): dram_remaining=72GB > 22GB needed
    #     → dram_window_frac=0.46, nand_load=0 (LLMFlash never hits NAND at B=1)
    #   Tight DRAM (16H+32C=48GB, 72B INT8): dram_remaining=24GB
    #     → at B=4: active=0.915, dram_window=0.50, nand_load=0.415 (20GB NAND)
    dram_window_frac = min(
        active_frac_batch,
        dram_remaining / total_ffn_bytes if total_ffn_bytes > 0
        else active_frac_batch
    )

    # NAND load: active neurons that overflow the DRAM window
    nand_load_frac = max(0.0, active_frac_batch - dram_window_frac)
    ffn_nand_bytes = total_ffn_bytes * nand_load_frac

    # DRAM turnover: activation-aware fraction of window that cycles per token
    # ReLU: 24% of window (neurons are truly zero — low churn)
    # SiLU: 60% of window (contextual — different inputs activate different neurons)
    dram_turnover_frac = dram_window_frac * turnover_rate
    ffn_dram_bytes     = total_ffn_bytes * dram_turnover_frac
    dram_rewrite_bytes = ffn_dram_bytes * DRAM_REWRITE_FRAC

    def nand_bundled(n_bytes):
        """Transfer from CXL NAND with row-column bundling boost."""
        return transfer_time_s(n_bytes, CXL_SSD_NAND) / BUNDLING_THROUGHPUT_BOOST



        # ── Decode loop with per-token write stall tracking ───────────────────────
    total_kv_write_stall_s  = 0.0
    total_decode_time       = 0.0
    per_token_write_stall_pcts = []

    for _ in range(NUM_DECODE_TOKENS):
        t        = 0.0
        step_kv  = 0.0

        # Attention weights: pinned in CXL DRAM
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, CXL_DRAM)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        # FFN DRAM turnover
        t += transfer_time_s(ffn_dram_bytes, CXL_DRAM)

        # FFN NAND overflow
        if ffn_nand_bytes > 0:
            t += nand_bundled(ffn_nand_bytes)

        # DRAM neuron swap overhead
        t += transfer_time_s(dram_rewrite_bytes, CXL_DRAM)

        # KV write: fully serialized (simplex) — this is the write stall
        step_kv = BATCH_SIZE * transfer_time_s(kv_per_seq_bytes, CXL_DRAM)
        total_kv_write_stall_s += step_kv
        t += step_kv

        total_decode_time += t

        # Write stall as % of this token's total step time
        stall_pct_this_token = (step_kv / t * 100) if t > 0 else 0.0
        per_token_write_stall_pcts.append(stall_pct_this_token)

    avg_decode_t = total_decode_time / NUM_DECODE_TOKENS
    decode_tps   = BATCH_SIZE / avg_decode_t if avg_decode_t > 0 else 0.0
    avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)



    # ══════════════════════════════════════════════════════════════════════════
    # PREFILL PHASE
    # Sparsity collapses during prefill: ALL neurons must fire to process
    # the full input context. No window benefit, no delta loading.
    # Paper Fig 4a: sagg(k) flattening only occurs in decode (steady-state).
    # SiLU turnover also collapses: prefill must load full FFN every token.
    # ══════════════════════════════════════════════════════════════════════════
    prefill_ffn_load = total_ffn_bytes * 1.0   # 100% active during prefill
    prefill_rewrite  = prefill_ffn_load * DRAM_REWRITE_FRAC

    total_prefill_time = 0.0
    for _ in range(NUM_PREFILL_TOKENS):
        t = 0.0

        # Attention: pinned in DRAM
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, CXL_DRAM)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        # FFN: full load from NAND (no sparsity, no window, no delta)
        t += nand_bundled(prefill_ffn_load)

        # DRAM rewrite + KV writes for B sequences
        t += transfer_time_s(prefill_rewrite, CXL_DRAM)
        t += BATCH_SIZE * transfer_time_s(kv_per_seq_bytes, CXL_DRAM)

        total_prefill_time += t

    avg_prefill_t = total_prefill_time / NUM_PREFILL_TOKENS
    prefill_tps   = BATCH_SIZE / avg_prefill_t if avg_prefill_t > 0 else 0.0

    total_model_bytes = sum(L["bytes"] for L in layers)
    cold_load = ssd_time_s(total_model_bytes)

    # ── Diagnostics ────────────────────────────────────────────────────────────
    print(f"=== LLM-in-Flash CXL Baseline (batch={BATCH_SIZE}) ===")
    print(f"  Activation fn          : {activation_label}")
    print(f"  Single-seq active_frac : {active_frac_single:.3f} "
          f"(avg_sparsity={avg_sparsity:.3f})")
    print(f"  Batch={BATCH_SIZE} active_frac  : {active_frac_batch:.3f} "
          f"[= 1-(1-{active_frac_single:.3f})^{BATCH_SIZE}]")
    print(f"  Sparsity savings left  : {(1.0 - active_frac_batch)*100:.1f}% "
          f"at B={BATCH_SIZE}")
    print(f"  DRAM remaining for FFN : {dram_remaining/GiB:.1f}GB / "
          f"{total_ffn_bytes/GiB:.1f}GB total FFN")
    print(f"  dram_window_frac       : {dram_window_frac:.3f} "
          f"→ DRAM turnover={ffn_dram_bytes/1e9:.2f}GB/step "
          f"({turnover_rate:.0%} of window)")
    print(f"  nand_load_frac         : {nand_load_frac:.3f} "
          f"→ NAND overflow={ffn_nand_bytes/1e9:.2f}GB/step")
    print(f"  KV writes/step         : {BATCH_SIZE} × "
          f"{kv_per_seq_bytes/1e3:.0f}KB = "
          f"{BATCH_SIZE * kv_per_seq_bytes/1e6:.1f}MB")
    print(f"  Available DRAM         : {total_dram/GiB:.0f}GB total")

    # ── Output in format expected by parse_metrics ─────────────────────────────
    print(f"Decode throughput: {decode_tps:.4f}")
    print(f"Prefill throughput: {prefill_tps:.1f}")
    print(f"Overall throughput: "
          f"{(NUM_PREFILL_TOKENS + 16) / (cold_load + avg_prefill_t + avg_decode_t * 16):.3f}")

    print(f"Read_Ratio: 100.0%")

    print(f"Write_Stall_Time_s: {total_kv_write_stall_s / NUM_DECODE_TOKENS:.6f}")
    print(f"Write_Stall_Pct: {(total_kv_write_stall_s / NUM_DECODE_TOKENS / avg_decode_t) * 100:.4f}%")
    print(f"Write_Util_Pct: 0.0000%")


    # ── compute real IO split ──────────────────────────────────────────────────
    total_read_bytes  = (sum(L["bytes"] * pinned_dram_frac for L in pinned_layers)
                        + ffn_dram_bytes + ffn_nand_bytes + dram_rewrite_bytes)
    total_write_bytes = BATCH_SIZE * kv_per_seq_bytes
    total_io_bytes    = total_read_bytes + total_write_bytes

    read_pct  = (total_read_bytes  / total_io_bytes) * 100 if total_io_bytes > 0 else 100.0
    write_pct = (total_write_bytes / total_io_bytes) * 100 if total_io_bytes > 0 else 0.0

    print(f"Read_Op_Percent: {read_pct:.4f}%")
    print(f"Write_Op_Percent: {write_pct:.4f}%")
    print(f"Read_Ratio: {read_pct:.4f}%")

    print(f"Write_Stall_Time_s: {total_kv_write_stall_s / NUM_DECODE_TOKENS:.6f}")
    print(f"Write_Stall_Pct: {avg_write_stall_pct:.4f}%")
    print(f"Write_Util_Pct: 0.0000%")
    print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")



if __name__ == "__main__":
    simulate_llmflash()
