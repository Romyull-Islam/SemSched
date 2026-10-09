"""
CXL-adapted LLM-in-a-Flash baseline (Alizadeh et al., ACL 2024).

At decode step t attention reads (NUM_PREFILL_TOKENS + t) x per-token KV bytes
x BATCH_SIZE from the unified DRAM pool.

Mechanisms modeled:
  1. Selective persistence: non-FFN layers (attention, embeddings) stay in the
     DRAM pool; only what does not fit is read from NAND.
  2. Sparsity: active fraction from the per-layer sparsity that
     decomposed_build_layers sets.
  3. Sliding window (k=5), used as the prefetch depth of the decode pipeline.
  4. Row-column bundling: factor 1.0 on CMM-H, see BUNDLING_THROUGHPUT_BOOST.
  5. Preallocated DRAM: no neuron rewrite cost, see DRAM_REWRITE_FRAC.
  6. DRAM window overflow: active neurons beyond the window come from NAND.
  7. Prefill: sparsity collapses, so the full FFN is loaded once.
  8. Batching: active_frac(B) = 1-(1-p)^B, TPS = B / step_time.
  9. Window turnover by activation (reported, not charged): ~24% per token for
     ReLU/FATReLU (Sec 4.1), ~60% for SiLU/SwiGLU.
 10. KV writes: one token's increment per sequence per step, not the full cache.
"""

from model_cfg import decomposed_build_layers, DEFAULT_MODEL_CFG, BYTES_PER_PARAM
from sim_cfg import (
    host_dram_capacity_bytes,
    cxl_dev_dram_capacity_bytes,
    BATCH_SIZE,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
)


def compute_time_s(flops):
    """Same compute model as FlexGen, LIA and SemSched: compute is a property of
    the simulated hardware, not of the scheduling policy."""
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core
                    * parallel_efficiency)
from tiers import kv_growth_spill_time_s
from pipeline import pipelined_time_s
from tiers import CXL_DRAM, CXL_SSD_NAND, HOST_DRAM, GPU_HBM, transfer_time_s, \
                  Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, GiB

# LLM-in-a-Flash treats DRAM as one unified tier [1], so host DRAM and CXL
# device DRAM form a single pool. Accelerator memory joins the pool when one is
# attached, which gives this policy the same accelerator SemSched gets.
from sim_cfg import gpu_hbm_capacity_bytes as _g
_h = host_dram_capacity_bytes
_c = cxl_dev_dram_capacity_bytes
# The pool fills fastest first, as any unified allocator would. Reading bytes
# spread over tiers takes sum(bytes_i / bw_i), so the pool's bandwidth is derived
# from the bytes resident in each member (a harmonic mean, not an arithmetic
# one). This keeps it monotone in every member's capacity.
_POOL_TIERS = [(_g, GPU_HBM.bw_Bps), (_h, HOST_DRAM.bw_Bps), (_c, CXL_DRAM.bw_Bps)]
_pool_lat = (_g * GPU_HBM.chunk_latency_s + _h * HOST_DRAM.chunk_latency_s
             + _c * CXL_DRAM.chunk_latency_s) / (_g + _h + _c)


def _pool_eff_bw(nbytes):
    """Serial time to read nbytes resident fastest-first, as a bandwidth."""
    if nbytes <= 0:
        return GPU_HBM.bw_Bps
    left, t = float(nbytes), 0.0
    for cap, bw in _POOL_TIERS:
        take = min(left, float(cap))
        t += take / bw
        left -= take
        if left <= 0:
            break
    if left > 0:                      # beyond pool capacity: slowest member
        t += left / _POOL_TIERS[-1][1]
    return nbytes / t


class _FastFirstPool:
    """Duck-typed Tier whose bandwidth depends on the working set actually
    read. bw_Bps reflects the full resident set; transfer_time_s() consults
    bw_Bps and chunk_latency_s only, so this slots in unchanged."""
    name = "Unified DRAM pool (fastest-first)"
    chunk_latency_s = _pool_lat

    def __init__(self):
        self.bw_Bps = _pool_eff_bw(_g + _h + _c)

    def set_working_set(self, nbytes):
        self.bw_Bps = _pool_eff_bw(nbytes)


DRAM_POOL = _FastFirstPool()

_WS_SPLIT = {}          # tier-name -> fraction of the resident set it holds


def _set_ws_split(nbytes):
    """Residency of the working set, fastest-first, as per-tier fractions.
    A read of the resident set is spread in these proportions, and the shared
    engine runs each tier as an independent bus."""
    global _WS_SPLIT
    names = ["GPU HBM", "Host DRAM", "CXL Device DRAM"]
    left = float(max(nbytes, 1.0))
    out = {}
    for (cap, _bw), nm in zip(_POOL_TIERS, names):
        take = min(left, float(cap))
        if take > 0:
            out[nm] = take / nbytes
        left -= take
        if left <= 0:
            break
    _WS_SPLIT = out


def pool_bytes(n):
    """n bytes of pooled reads, as a per-tier byte dict for the engine."""
    return {t: n * f for t, f in _WS_SPLIT.items()}

# Paper constants (OPT/ReLU, Sec 3.1 and 4.1).
WINDOW_SIZE_K             = 5    # Sliding window token count
# Row-column bundling lifts the paper's flash from 1.25 to 2.25 GB/s (Table 2),
# but does not carry over to CMM-H. Zeng et al.: "The CMM-H device employs a PCIe
# Gen 4 x4 NVMe SSD", a 7.88 GB/s link, and the measured 5.0 GB/s miss bandwidth
# already reflects streaming access. A 1.8x boost would give 9.0 GB/s, above the
# link, so every policy reads NAND at the measured rate.
BUNDLING_THROUGHPUT_BOOST = 1.0
# Sec 3.3 presents rewriting ~25% of the FFNs in DRAM as the cost that
# LLM-in-a-Flash avoids by "preallocation of all necessary memory", so no rewrite
# is charged.
DRAM_REWRITE_FRAC         = 0.0  # preallocated, Sec 3.3

# DRAM window turnover per token, by activation function.
# ReLU/FATReLU (Sec 4.1): the delta is ~2.4% of the FFN, i.e. 24% of a 10%
# active set. SiLU/SwiGLU (Qwen2.5, Llama, Mistral) has no structural zeros, so
# different tokens activate different neurons and sagg(k+1) - sagg(k) stays
# large: ~60% of the window changes per token.
DRAM_TURNOVER_RELU = 0.24   # OPT/ReLU, Sec 4.1
DRAM_TURNOVER_SILU = 0.60   # SiLU/SwiGLU, input dependent

NUM_DECODE_TOKENS  = 16
NUM_PREFILL_TOKENS = 512


def ssd_time_s(n):
    return transfer_time_s(
        n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))


def simulate_llmflash():
    import model_cfg as _mcfg

    layers     = decomposed_build_layers(DEFAULT_MODEL_CFG())

    # KV capacity reservation: the pool holds the KV cache as well as weights.
    # Reserved at the generation mean, PREFILL + TOKENS/2, since the cache grows
    # during decode.
    _kv_resident = int(sum(L.get("kv_cache_bytes", 0) for L in layers)
                       / NUM_PREFILL_TOKENS * BATCH_SIZE
                       * (NUM_PREFILL_TOKENS + NUM_DECODE_TOKENS / 2.0))
    # Pool capacity includes accelerator memory when attached, matching the pool
    # bandwidth above.
    from sim_cfg import gpu_hbm_capacity_bytes as _gcap
    total_dram = max(0, _gcap + host_dram_capacity_bytes
                        + cxl_dev_dram_capacity_bytes - _kv_resident)

    # Selective persistence: everything except the FFNs is pinned in the pool.
    pinned_layers = [L for L in layers if L.get("kind") != "MLP"]
    ffn_layers    = [L for L in layers if L.get("kind") == "MLP"]

    total_pinned_bytes = sum(L["bytes"] for L in pinned_layers)
    _ws = min(_gcap + host_dram_capacity_bytes + cxl_dev_dram_capacity_bytes,
              sum(L["bytes"] for L in layers) + _kv_resident)
    DRAM_POOL.set_working_set(_ws)
    _set_ws_split(_ws)
    total_ffn_bytes    = sum(L["bytes"] for L in ffn_layers)

    # Turnover rate from model_cfg.ACTIVATION_FN, default "silu".
    activation_fn  = getattr(_mcfg, "ACTIVATION_FN", "silu").lower()
    turnover_rate  = DRAM_TURNOVER_RELU if activation_fn == "relu" \
                     else DRAM_TURNOVER_SILU
    activation_label = f"{activation_fn.upper()} → turnover={turnover_rate:.0%}"

    # kv_cache_bytes is the full 512-token KV budget from the layer config; a
    # decode step writes one token's worth per sequence.
    kv_per_seq_bytes = sum(L.get("kv_cache_bytes", 0)
                           for L in pinned_layers) // NUM_PREFILL_TOKENS

    # Single-sequence active fraction from the per-layer sparsity.
    avg_sparsity       = (sum(L.get("sparsity", 0.54) for L in ffn_layers)
                          / max(len(ffn_layers), 1))
    active_frac_single = 1.0 - avg_sparsity

    # Union of active neuron sets across B sequences, active_frac(B) = 1-(1-p)^B:
    #   ReLU, B=1: 0.10  (full paper benefit)
    #   SiLU, B=1: 0.46
    #   SiLU, B=4: 0.915 (mostly collapsed)
    #   SiLU, B=8: 0.993 (fully collapsed)
    active_frac_batch = min(1.0, 1.0 - (1.0 - active_frac_single) ** BATCH_SIZE)

    # Share of the pinned layers that fits in the pool; the rest comes from NAND.
    pinned_dram_frac = min(1.0, total_dram / total_pinned_bytes) \
                       if total_pinned_bytes > 0 else 1.0
    dram_remaining   = max(0.0, total_dram - total_pinned_bytes)

    # dram_window_frac: share of the FFN that is active in the batch and fits in
    # the pool beside the pinned layers. With ample DRAM it equals the active
    # fraction and nothing is read from NAND; in a tight pool the excess active
    # neurons overflow to NAND.
    dram_window_frac = min(
        active_frac_batch,
        dram_remaining / total_ffn_bytes if total_ffn_bytes > 0
        else active_frac_batch
    )

    # NAND load: active neurons that overflow the DRAM window
    nand_load_frac = max(0.0, active_frac_batch - dram_window_frac)
    ffn_nand_bytes = total_ffn_bytes * nand_load_frac

    # Share of the window that turns over per token (24% ReLU, 60% SiLU). It is
    # reported but not charged: turnover sets what is refetched, not what is read.
    dram_turnover_frac = dram_window_frac * turnover_rate
    # Every neuron that fires is read from where it lives, so the DRAM read is the
    # whole window and window + NAND overflow = active fraction. dram_window_frac
    # already includes sparsity; multiplying by active_frac_batch would count it
    # twice.
    ffn_dram_bytes     = total_ffn_bytes * dram_window_frac
    dram_rewrite_bytes = ffn_dram_bytes * DRAM_REWRITE_FRAC

    def nand_bundled(n_bytes):
        """Transfer from CXL NAND, divided by the bundling factor (1.0 here)."""
        return transfer_time_s(n_bytes, CXL_SSD_NAND) / BUNDLING_THROUGHPUT_BOOST



    # Decode loop with per-token write stall tracking.
    total_kv_write_stall_s  = 0.0
    total_decode_time       = 0.0
    per_token_write_stall_pcts = []

    for token_step in range(NUM_DECODE_TOKENS):
        t        = 0.0
        step_kv  = 0.0

        # Serial cost of the step. The pipelined model below replaces t; only
        # step_kv and kv_read_bytes_total are carried over from this part.

        # Pinned layers from the pool, any overflow from NAND
        for L in pinned_layers:
            t += transfer_time_s(L["bytes"] * pinned_dram_frac, DRAM_POOL)
            if pinned_dram_frac < 1.0:
                t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

        # Growing KV: attention reads every cached position from the pool.
        kv_positions_cached = NUM_PREFILL_TOKENS + token_step
        kv_read_bytes_total = BATCH_SIZE * kv_per_seq_bytes * kv_positions_cached
        t += transfer_time_s(kv_read_bytes_total, DRAM_POOL)

        # Active FFN neurons resident in DRAM
        t += transfer_time_s(ffn_dram_bytes, DRAM_POOL)

        # FFN NAND overflow
        if ffn_nand_bytes > 0:
            t += nand_bundled(ffn_nand_bytes)

        # Neuron rewrite, zero with preallocation
        t += transfer_time_s(dram_rewrite_bytes, DRAM_POOL)

        # KV write, serialized after the reads (simplex); this is the write stall
        step_kv = BATCH_SIZE * transfer_time_s(kv_per_seq_bytes, DRAM_POOL)
        total_kv_write_stall_s += step_kv
        t += step_kv

        # Skipping inactive neurons is this paper's own mechanism [1], so it gets
        # a FLOP discount that FlexGen, LIA and SemSched do not: attention runs
        # dense, the FFN runs only its active fraction. max(memory, compute) over
        # the step assumes perfect load/compute overlap, which favors this baseline.
        comp_attn = sum(L["flops"] for L in pinned_layers) * BATCH_SIZE
        comp_ffn  = sum(L["flops"] for L in ffn_layers) * BATCH_SIZE * active_frac_batch

        # The pool and the NAND backend are separate paths, so a window refill
        # can proceed while resident neurons are read. The k=5 window serves as
        # the prefetch depth, and pool space left after the resident set is the
        # staging room.
        _units, _nl = [], max(1, len(ffn_layers))
        for _L in pinned_layers:
            _by = pool_bytes(_L["bytes"] * pinned_dram_frac)
            _by["CXL Device NAND"] = _by.get("CXL Device NAND", 0.0) \
                + _L["bytes"] * (1.0 - pinned_dram_frac)
            _units.append((_by, compute_time_s(_L["flops"] * BATCH_SIZE)))
        for _L in ffn_layers:
            _by = pool_bytes((ffn_dram_bytes + dram_rewrite_bytes) / _nl)
            _units.append(({**_by,
                            "CXL Device NAND": _by.get("CXL Device NAND", 0.0)
                                               + ffn_nand_bytes / _nl},
                           compute_time_s(_L["flops"] * BATCH_SIZE * active_frac_batch)))
        _units.append((pool_bytes(kv_read_bytes_total), 0.0))
        _window = max(0.0, total_dram - total_pinned_bytes * pinned_dram_frac
                      - ffn_dram_bytes - dram_rewrite_bytes)
        t = pipelined_time_s(_units, WINDOW_SIZE_K,
                             {"GPU HBM": GPU_HBM.bw_Bps,
                              "Host DRAM": HOST_DRAM.bw_Bps,
                              "CXL Device DRAM": CXL_DRAM.bw_Bps,
                              "CXL Device NAND": CXL_SSD_NAND.bw_Bps * BUNDLING_THROUGHPUT_BOOST},
                             {"GPU HBM": GPU_HBM.chunk_latency_s,
                              "Host DRAM": HOST_DRAM.chunk_latency_s,
                              "CXL Device DRAM": CXL_DRAM.chunk_latency_s,
                              "CXL Device NAND": CXL_SSD_NAND.chunk_latency_s},
                             inflight_budget=_window) + step_kv
        t = max(t, compute_time_s(comp_attn + comp_ffn))

        t += kv_growth_spill_time_s(
            kv_per_seq_bytes * BATCH_SIZE * (NUM_PREFILL_TOKENS + token_step + 1),
            _kv_resident, DRAM_POOL)
        total_decode_time += t

        # Write stall as % of this token's total step time
        stall_pct_this_token = (step_kv / t * 100) if t > 0 else 0.0
        per_token_write_stall_pcts.append(stall_pct_this_token)

    avg_decode_t = total_decode_time / NUM_DECODE_TOKENS
    decode_tps   = BATCH_SIZE / avg_decode_t if avg_decode_t > 0 else 0.0
    avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)



    # Prefill. Sparsity collapses: every neuron fires over the full prompt, so
    # there is no window or delta loading (sagg(k) flattens only in decode,
    # paper Fig 4a).
    prefill_ffn_load = total_ffn_bytes * 1.0   # 100% active during prefill
    prefill_rewrite  = prefill_ffn_load * DRAM_REWRITE_FRAC

    # LLM-in-a-Flash does not model prompt processing: its Limitations section
    # says the study is 'limited to single-batch inference' and leaves 'more
    # complex scenarios like prompt processing' to future work. Prefill gets the
    # same treatment as the other baselines, one pass over the weights for all
    # NUM_PREFILL_TOKENS positions, which is why it is compute bound.
    weight_t = 0.0

    # Pinned layers from the pool, any overflow from NAND
    for L in pinned_layers:
        weight_t += transfer_time_s(L["bytes"] * pinned_dram_frac, DRAM_POOL)
        if pinned_dram_frac < 1.0:
            weight_t += nand_bundled(L["bytes"] * (1.0 - pinned_dram_frac))

    # FFN: full load from NAND, since sparsity collapses during prefill (this
    # baseline's own assumption)
    weight_t += nand_bundled(prefill_ffn_load)
    weight_t += transfer_time_s(prefill_rewrite, DRAM_POOL)

    # KV writes scale with prompt length: one per position per sequence.
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

    # Diagnostics
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

    # Metrics in the format parse_metrics expects
    print(f"Decode throughput: {decode_tps:.4f}")
    print(f"Prefill throughput: {prefill_tps:.6f}")
    print(f"Overall throughput: "
          f"{(NUM_PREFILL_TOKENS + 16) / (total_prefill_time + avg_decode_t * 16):.3f}")

    print(f"Read_Ratio: 100.0%")

    print(f"Write_Stall_Time_s: {total_kv_write_stall_s / NUM_DECODE_TOKENS:.6f}")
    print(f"Write_Stall_Pct: {(total_kv_write_stall_s / NUM_DECODE_TOKENS / avg_decode_t) * 100:.4f}%")
    print(f"Write_Util_Pct: 0.0000%")


    # Read/write byte split of one decode step
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
