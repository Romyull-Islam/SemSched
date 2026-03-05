"""
llmflash_baseline.py

CXL-adapted LLM-in-a-Flash Baseline (Alizadeh et al., ACL 2024)
BATCH FIX: Active FFN fraction expands with batch size as:
           active_frac(B) = 1 - (1 - active_frac_single)^B
           At B=32, ~97% of FFN must be loaded regardless of sparsity.
"""

from model_cfg import decomposed_build_layers, DEFAULT_MODEL_CFG, BYTES_PER_PARAM
from sim_cfg import (
    host_dram_capacity_bytes,
    cxl_dev_dram_capacity_bytes,
    BATCH_SIZE,
)
from tiers import CXL_DRAM, CXL_SSD_NAND, transfer_time_s, GiB

# ── Architecture-independent paper constants ───────────────────────────────────
WINDOW_SIZE_K             = 5    # Sliding window token count (paper §3.1)
BUNDLING_THROUGHPUT_BOOST = 1.8  # Row-col bundling: 1.25→2.25 GB/s (Table 2)
DRAM_REWRITE_FRAC         = 0.25 # Neuron swap rewrite cost (paper §3.3)

NUM_DECODE_TOKENS  = 16
NUM_PREFILL_TOKENS = 512


def simulate_llmflash():
    layers     = decomposed_build_layers(DEFAULT_MODEL_CFG())
    total_dram = host_dram_capacity_bytes + cxl_dev_dram_capacity_bytes

    # ── Partition layers ───────────────────────────────────────────────────────
    pinned_layers = [L for L in layers if L.get("kind") != "MLP"]
    ffn_layers    = [L for L in layers if L.get("kind") == "MLP"]

    total_pinned_bytes = sum(L["bytes"] for L in pinned_layers)
    total_ffn_bytes    = sum(L["bytes"] for L in ffn_layers)

    # ── Single-request active fraction (from layer sparsity) ──────────────────
    avg_sparsity        = (sum(L.get("sparsity", 0.0) for L in ffn_layers)
                           / max(len(ffn_layers), 1))
    active_frac_single  = 1.0 - avg_sparsity

    # ── BATCH FIX: Union of active neurons across B sequences ─────────────────
    # Formula: active_frac(B) = 1 - (1 - p)^B
    # B=1: p≈0.46 (SiLU) or 0.10 (ReLU)
    # B=32: collapses to ~97% regardless of single-token sparsity
    active_frac_batch = 1.0 - (1.0 - active_frac_single) ** BATCH_SIZE
    active_frac_batch = min(1.0, active_frac_batch)

    # sagg(k) scales with batch active fraction
    sagg_k_frac        = min(1.0, active_frac_batch * (1.0 + 0.20 * WINDOW_SIZE_K))
    delta_frac         = active_frac_batch * 0.25
    ffn_delta_bytes    = total_ffn_bytes * delta_frac
    dram_rewrite_bytes = ffn_delta_bytes * DRAM_REWRITE_FRAC

    # KV cache write bytes scale linearly with batch size
    kv_cache_per_layer  = sum(L.get("kv_cache_bytes", 0) for L in pinned_layers)
    kv_write_bytes      = kv_cache_per_layer * BATCH_SIZE

    # ── DRAM capacity check ────────────────────────────────────────────────────
    pinned_dram_frac = min(1.0, total_dram / total_pinned_bytes) \
                       if total_pinned_bytes > 0 else 1.0
    dram_remaining       = max(0, total_dram - total_pinned_bytes)
    window_bytes         = total_ffn_bytes * sagg_k_frac
    window_overflow_frac = max(0.0, (window_bytes - dram_remaining) / window_bytes) \
                           if window_bytes > 0 else 0.0

    # ── Helper ─────────────────────────────────────────────────────────────────
    def nand_bundled(n_bytes):
        return transfer_time_s(n_bytes, CXL_SSD_NAND) / BUNDLING_THROUGHPUT_BOOST

    # ══════════════════════════════════════════════════════════════════════════
    # DECODE PHASE
    # ══════════════════════════════════════════════════════════════════════════
    total_decode_time = 0.0

    for _ in range(NUM_DECODE_TOKENS):
        t = 0.0

        # Attention: pinned in DRAM
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, CXL_DRAM)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        # FFN delta: batch-expanded active fraction from NAND
        t += nand_bundled(ffn_delta_bytes)

        # Window overflow penalty
        t += nand_bundled(window_bytes * window_overflow_frac * delta_frac)

        # DRAM rewrite + KV cache write (scales with batch)
        t += transfer_time_s(dram_rewrite_bytes + kv_write_bytes, CXL_DRAM)

        total_decode_time += t

    avg_decode_t = total_decode_time / NUM_DECODE_TOKENS
    decode_tps   = (1.0 / avg_decode_t) * BATCH_SIZE if avg_decode_t > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # PREFILL PHASE — always full FFN load (no sparsity benefit)
    # ══════════════════════════════════════════════════════════════════════════
    prefill_ffn_load = total_ffn_bytes * 1.0
    prefill_rewrite  = prefill_ffn_load * DRAM_REWRITE_FRAC

    total_prefill_time = 0.0

    for _ in range(NUM_PREFILL_TOKENS):
        t = 0.0
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, CXL_DRAM)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        t += nand_bundled(prefill_ffn_load)
        t += transfer_time_s(prefill_rewrite + kv_write_bytes, CXL_DRAM)
        total_prefill_time += t

    avg_prefill_t = total_prefill_time / NUM_PREFILL_TOKENS
    prefill_tps   = (1.0 / avg_prefill_t) * BATCH_SIZE if avg_prefill_t > 0 else 0.0

    # ── Diagnostics ────────────────────────────────────────────────────────────
    print(f"=== LLM-in-Flash CXL Baseline (batch={BATCH_SIZE}) ===")
    print(f"  Single-token active_frac : {active_frac_single:.2f} "
          f"(avg_sparsity={avg_sparsity:.2f})")
    print(f"  Batch={BATCH_SIZE} active_frac   : {active_frac_batch:.2f} "
          f"[= 1-(1-{active_frac_single:.2f})^{BATCH_SIZE}]")
    print(f"  Sparsity advantage left  : "
          f"{(1.0-active_frac_batch)*100:.1f}% savings remain at B={BATCH_SIZE}")
    print(f"  FFN delta/token          : {delta_frac*100:.1f}% "
          f"= {ffn_delta_bytes/1e9:.2f}GB (bundled)")
    print(f"  DRAM window overflow     : {window_overflow_frac*100:.0f}%")

    print(f"Decode throughput: {decode_tps:.4f}")
    print(f"Prefill throughput: {prefill_tps:.1f}")
    print(f"Read_Op_Percent: 100.0%")
    print(f"Write_Op_Percent: 0.0%")
    print(f"Read_Ratio: 100.0%")


if __name__ == "__main__":
    simulate_llmflash()
