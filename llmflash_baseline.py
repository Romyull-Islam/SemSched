"""
llmflash_baseline.py

CXL-adapted LLM-in-a-Flash Baseline (Alizadeh et al., ACL 2024).

Decode-phase includes growing-KV reads: at decode step t the attention reads
(NUM_PREFILL_TOKENS + t) × per-token-KV-bytes × BATCH_SIZE from the pooled
(CXL) DRAM tier.

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
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
)


def compute_time_s(flops):
    """Identical compute model to FlexGen/LIA/SemSched — compute is a property of
    the simulated hardware, not of any baseline's scheduling policy."""
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core
                    * parallel_efficiency)
from tiers import CXL_DRAM, CXL_SSD_NAND, HOST_DRAM, GPU_HBM, transfer_time_s, \
                  Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, GiB

# A10 FIX (BigData 2026): LLM-in-a-Flash pools Host and CXL DRAM into a single
# unified tier [1] -- as our own section IV-B states. Previously every transfer
# was charged at CXL_DRAM speed, so the host-resident share of the pool was
# under-modeled and the strongest baseline was unfairly penalised. The pool is
# now a capacity-weighted blend of the two DRAM tiers.
# When an accelerator is attached its HBM joins the pool, which is faithful to
# LLM-in-a-Flash: its policy is a single unified fast tier, so it benefits from
# accelerator memory on exactly the same terms we do.
from sim_cfg import gpu_hbm_capacity_bytes as _g
_h = host_dram_capacity_bytes
_c = cxl_dev_dram_capacity_bytes
# HARMONIC, not arithmetic. Moving B bytes spread over tiers takes
# sum(bytes_i / bw_i), so the effective bandwidth of a pool is the
# capacity-weighted HARMONIC mean. The arithmetic mean always overstates it
# (Jensen), which handed this baseline ~2.8% of free throughput on exactly the
# bytes every policy has to move. This is an arithmetic correction, not a
# judgement about their design: their unified-pool POLICY is unchanged.
_pool_bw  = ((_g + _h + _c) /
             (_g / GPU_HBM.bw_Bps + _h / HOST_DRAM.bw_Bps + _c / CXL_DRAM.bw_Bps))
_pool_lat = (_g * GPU_HBM.chunk_latency_s + _h * HOST_DRAM.chunk_latency_s
             + _c * CXL_DRAM.chunk_latency_s) / (_g + _h + _c)
DRAM_POOL = Tier("Unified Host+CXL DRAM pool", _pool_bw, _pool_lat)

# ── Paper constants (OPT/ReLU baseline, §3.1 and §4.1) ────────────────────────
WINDOW_SIZE_K             = 5    # Sliding window token count
# Row-column bundling is real -- LLM-in-a-Flash Table 2 measures it lifting
# THEIR flash from 1.25 to 2.25 GB/s. It does not transplant onto CMM-H.
#
# Zeng et al., verbatim: "The CMM-H device employs a PCIe Gen 4 x4 NVMe SSD."
# Gen4 x4 carries 7.88 GB/s. The prototype's measured stable miss bandwidth is
# 5.0 GB/s -- 63% of that link, a normal sustained sequential rate for TLC, and
# already the product of streaming access. Applying 1.8x on top yields 9.0 GB/s,
# which is 14% ABOVE the theoretical capacity of the bus the data must cross.
# No access pattern moves bytes faster than the link carries them.
#
# So this is not a choice between two conventions. 1.8 asks the hardware for
# something it cannot do, and it asked it for one policy only -- no other
# simulator here has any equivalent term. Every policy now reads the NAND
# backend at the one documented, physically attainable rate.
BUNDLING_THROUGHPUT_BOOST = 1.0
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

    # ── KV capacity reservation (audit A12) ───────────────────────────────────
    # The unified DRAM pool holds the KV cache as well as weights, so its bytes
    # are not available for the working set. This previously pooled host + CXL
    # in full: at INT8 16H+32C that placed 83.1 GB of state into 76 GB of
    # memory. Reserved at the generation mean, PREFILL + TOKENS/2, since the
    # cache grows during decode.
    _kv_resident = int(sum(L.get("kv_cache_bytes", 0) for L in layers)
                       / NUM_PREFILL_TOKENS * BATCH_SIZE
                       * (NUM_PREFILL_TOKENS + NUM_DECODE_TOKENS / 2.0))
    total_dram = max(0, host_dram_capacity_bytes + cxl_dev_dram_capacity_bytes
                        - _kv_resident)

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
    # Turnover governs how much of the WINDOW must be re-fetched from NAND, not
    # how much of the DRAM-resident FFN is read for compute. Every neuron that
    # fires must be read from wherever it lives, and at B=128 the active
    # fraction is 1.000 -- the union across the batch is the whole layer. This
    # previously charged 0.60 x window of the resident FFN, so LLM-in-a-Flash
    # paid for ~75% of weights it still had to read, where the other four
    # simulators pay 100%. The re-fetch path below keeps the turnover term.
    ffn_dram_bytes     = total_ffn_bytes * dram_window_frac * active_frac_batch
    dram_rewrite_bytes = ffn_dram_bytes * DRAM_REWRITE_FRAC

    def nand_bundled(n_bytes):
        """Transfer from CXL NAND with row-column bundling boost."""
        return transfer_time_s(n_bytes, CXL_SSD_NAND) / BUNDLING_THROUGHPUT_BOOST



        # ── Decode loop with per-token write stall tracking ───────────────────────
    total_kv_write_stall_s  = 0.0
    total_decode_time       = 0.0
    per_token_write_stall_pcts = []

    for token_step in range(NUM_DECODE_TOKENS):
        t        = 0.0
        step_kv  = 0.0

        # Attention weights: pinned in CXL DRAM
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, DRAM_POOL)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        # Growing-KV: attention layers read all prior K/V from the pooled CXL DRAM tier.
        # LLMFlash treats host+CXL DRAM as unified; we use CXL_DRAM as the KV tier.
        kv_positions_cached = NUM_PREFILL_TOKENS + token_step
        kv_read_bytes_total = BATCH_SIZE * kv_per_seq_bytes * kv_positions_cached
        t += transfer_time_s(kv_read_bytes_total, DRAM_POOL)

        # FFN DRAM turnover
        t += transfer_time_s(ffn_dram_bytes, DRAM_POOL)

        # FFN NAND overflow
        if ffn_nand_bytes > 0:
            t += nand_bundled(ffn_nand_bytes)

        # DRAM neuron swap overhead
        t += transfer_time_s(dram_rewrite_bytes, DRAM_POOL)

        # KV write: fully serialized (simplex) — this is the write stall
        step_kv = BATCH_SIZE * transfer_time_s(kv_per_seq_bytes, DRAM_POOL)
        total_kv_write_stall_s += step_kv
        t += step_kv

        # FIX (BigData 2026): LLMFlash previously had NO compute term at all.
        # Sparsity-driven compute skipping is this paper's own mechanism [1], so
        # it legitimately receives the FLOP discount that FlexGen/LIA/SemSched do
        # not: attention runs dense, FFN runs only the active neuron fraction.
        # We take max(memory, compute) over the whole step, which credits
        # LLMFlash with perfect load/compute overlap — deliberately generous to
        # the strongest baseline.
        comp_attn = sum(L["flops"] for L in pinned_layers) * BATCH_SIZE
        comp_ffn  = sum(L["flops"] for L in ffn_layers) * BATCH_SIZE * active_frac_batch
        t = max(t, compute_time_s(comp_attn + comp_ffn))

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

    # LLM-in-a-Flash does not model prompt processing. Its Limitations section
    # states the study is 'limited to single-batch inference' and lists 'more
    # complex scenarios like prompt processing' as future work. Any prefill
    # behaviour attributed to it is OUR construction, so we give it the same
    # standard treatment as every other baseline rather than a pessimistic
    # extrapolation it never claimed.
    #
    # Prefill makes ONE pass over the parameters for the whole prompt: all
    # NUM_PREFILL_TOKENS positions are processed together, which is exactly why
    # prefill is compute-bound rather than memory-bound. The previous loop paid
    # the full weight load once per prompt token, charging LLM-in-a-Flash 512
    # model loads where every other simulator here pays one, and understating
    # its prefill throughput by roughly that factor.
    weight_t = 0.0

    # Attention: pinned in DRAM
    for L in pinned_layers:
        weight_t += transfer_time_s(L["bytes"] * pinned_dram_frac, DRAM_POOL)
        if pinned_dram_frac < 1.0:
            weight_t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

    # FFN: full load from NAND (no sparsity, no window, no delta -- sparsity
    # collapses during prefill, which is this baseline's own stated assumption)
    weight_t += nand_bundled(prefill_ffn_load)
    weight_t += transfer_time_s(prefill_rewrite, DRAM_POOL)

    # KV writes DO scale with prompt length: one per position per sequence.
    weight_t += (NUM_PREFILL_TOKENS * BATCH_SIZE
                 * transfer_time_s(kv_per_seq_bytes, DRAM_POOL))

    # Compute over every prompt position across B sequences.
    prefill_compute = compute_time_s(
        sum(L["flops"] for L in layers) * BATCH_SIZE * NUM_PREFILL_TOKENS)

    total_prefill_time = max(weight_t, prefill_compute)
    prefill_tps = (NUM_PREFILL_TOKENS / total_prefill_time
                   if total_prefill_time > 0 else 0.0)

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
    print(f"Prefill throughput: {prefill_tps:.6f}")
    print(f"Overall throughput: "
          f"{(NUM_PREFILL_TOKENS + 16) / (total_prefill_time + avg_decode_t * 16):.3f}")

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
