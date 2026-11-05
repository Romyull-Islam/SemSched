# CXL-native baseline modeled on CXLAimPod (Yang et al., 2025, arXiv 2508.15980)
# [yang2025cxlaimpod].
#
# Policy, from the paper:
#   - Flat capacity-ordered tiering: GPU HBM (if any) -> host DRAM -> CXL device
#     DRAM -> CXL NAND, with decoder blocks as monolithic units.
#   - Device-side sequential prefetcher with a fixed lookahead window (k = 4).
#   - LRU eviction on the device cache, with no notion of layer type.
#
# Hardware model, shared with FlexGen, LIA, LLMFlash and TierSched:
#   - Same compute engine and FLOP accounting, batch-scaled:
#       decode  FLOPs = 2 * P * B      prefill FLOPs = 2 * P * seq_len * B
#   - Same tier bandwidths and latencies, growing-KV model and metrics
#     (decode TPS = mean over decode steps; prefill TPS per sequence).
#
# Baselines differ only in their own paper's policy, not in the machine model
# (Sec. IV-B, "Baseline fidelity principle").

import math

from tiers import kv_reserve_level, kv_step_split
from tiers import kv_growth_spill_time_s, kv_cascade, kv_tail
from tiers import GPU_HBM as _GPU
from pipeline import pipelined_time_s
from tiers import (HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
                   Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S)
from model_cfg import (build_layers, BYTES_PER_PARAM, DEFAULT_MODEL_CFG,
                       bytes_read, embedding_last)
from sim_cfg import (TOKENS, BATCH_SIZE, cpu_freq_hz, cpu_cores,
                     flops_per_cycle_per_core, parallel_efficiency,
                     host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes,
                     gpu_hbm_capacity_bytes)

# KV reserved at the generation mean or its peak, whichever is faster.
KV_LEVEL = kv_reserve_level(__file__)

PL_GPU_HBM = "GPU HBM"
PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

PREFETCH_WINDOW = 4      # CXLAimPod sequential lookahead depth
PREFILL_TOKENS = 512


def compute_time_s(flops):
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core
                    * parallel_efficiency)


def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def ssd_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW,
                                   NVME_STREAM_LAT_S))


# Model and flat capacity-ordered placement
cfg = DEFAULT_MODEL_CFG()
layers = build_layers(cfg, sequence_length=PREFILL_TOKENS)
head_dim = cfg.emb_dim // cfg.q_heads
kv_inc_per_tok = 2 * cfg.kv_heads * head_dim * BYTES_PER_PARAM

placement = [None] * len(layers)

# The KV cache stays resident in CXL device DRAM for the whole run, so that
# capacity cannot hold weights. It is reserved at the generation mean,
# PREFILL + TOKENS/2: the final size would starve the early steps and the initial
# size would over-commit the later ones.
_n_attn = sum(1 for L in layers if L.get("kv_cache_bytes", 0) > 0)
_kv_resident = int(kv_inc_per_tok * _n_attn * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS * KV_LEVEL))
# What the device DRAM cannot hold is read from NAND.
_kv_split, _kv_took = kv_cascade(_kv_resident,
                                 [("cxl", cxl_dev_dram_capacity_bytes)])
_kv_cxl_frac = _kv_split.get("cxl", 0.0)
_kv_nand_frac = _kv_split.get("nand", 0.0)

# With an accelerator attached, run_paper_tables.py gives every policy its HBM and
# its compute engine, so HBM heads the cascade here too. On the CPU-only platform
# gpu_hbm_capacity_bytes is 0 and nothing is placed there.
gpu_free  = gpu_hbm_capacity_bytes
host_free = host_dram_capacity_bytes
cxl_free  = max(0, cxl_dev_dram_capacity_bytes - _kv_took.get("cxl", 0.0))
for i in embedding_last(layers):
    L = layers[i]
    if L["bytes"] <= gpu_free:
        placement[i] = PL_GPU_HBM
        gpu_free -= L["bytes"]
    elif L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    elif L["bytes"] <= cxl_free:
        placement[i] = PL_CXL_DEV_DRAM
        cxl_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND


class CXLPrefetcher:
    """Sequential device-side prefetcher with LRU eviction that ignores layer type."""

    def __init__(self):
        self.link_busy_until = 0.0
        self.resident = []            # LRU order, oldest first
        self.mem_used = 0
        self.arrival = {}

    def schedule(self, idx, size, now):
        if idx in self.resident:
            return self.arrival.get(idx, 0.0)
        start = max(now, self.link_busy_until)
        finish = start + nand_time_s(size)
        self.link_busy_until = finish
        while (self.mem_used + size) > cxl_dev_dram_capacity_bytes and self.resident:
            ev = self.resident.pop(0)      # LRU, ignoring layer type
            self.mem_used -= layers[ev]["bytes"]
            self.arrival.pop(ev, None)
        self.resident.append(idx)
        self.mem_used += size
        self.arrival[idx] = finish
        return finish


def run_phase(is_prefill, token_step=0):
    """One prefill token or one decode step, across all layers."""
    pf = CXLPrefetcher()
    rows = BATCH_SIZE * (PREFILL_TOKENS if is_prefill else 1)   # embedding lookups
    units = []
    elapsed = 0.0
    read_stall = 0.0
    kv_write = 0.0

    for i, L in enumerate(layers):
        for k in range(1, PREFETCH_WINDOW + 1):
            j = i + k
            if j < len(layers) and placement[j] == PL_CXL_DEV_NAND:
                pf.schedule(j, bytes_read(layers[j], rows), elapsed)

        # Batch-scaled FLOPs; prefill additionally scales with sequence length.
        flops = L["flops"] * BATCH_SIZE * (PREFILL_TOKENS if is_prefill else 1)
        comp = compute_time_s(flops)

        b = bytes_read(L, rows)
        if placement[i] == PL_HOST_DRAM:
            mem = dram_time_s(b)
        elif placement[i] == PL_CXL_DEV_DRAM:
            mem = cxl_time_s(b)
        else:
            arrival = pf.schedule(i, b, elapsed)
            wait = max(0.0, arrival - elapsed)
            read_stall += wait
            mem = wait + cxl_time_s(b)

        _by = {placement[i]: b}
        # Growing KV: attention reads all cached positions so far. CXLAimPod
        # pools device memory, so KV is served from the CXL DRAM tier, and from
        # NAND for what it cannot hold.
        kv_bytes = L.get("kv_cache_bytes", 0)
        _kvb, _kvf = 0.0, {}
        if kv_bytes > 0 and not is_prefill:
            positions = PREFILL_TOKENS + token_step
            _kvb = positions * kv_inc_per_tok * BATCH_SIZE
            _kvf = kv_step_split(_kv_split, _kv_resident,
                                 kv_inc_per_tok * _n_attn * BATCH_SIZE * positions)
            mem += cxl_time_s(_kvb * _kv_cxl_frac) + nand_time_s(_kvb * _kv_nand_frac)
            # No duplex scheduling in this baseline: the KV write is serialized.
            w = cxl_time_s(kv_inc_per_tok * BATCH_SIZE)
            kv_write += w
            elapsed += w

        _by["CXL Device DRAM"] = _by.get("CXL Device DRAM", 0.0) + _kvb * _kvf.get("cxl", 0.0)
        if _kvf.get("nand", 0.0):
            _by["CXL Device NAND"] = _by.get("CXL Device NAND", 0.0) + _kvb * _kvf["nand"]
        units.append((_by, comp))
        elapsed += max(comp, mem)

    # CXLAimPod prefetches PREFETCH_WINDOW layers ahead over the CXL link into
    # pooled device memory, so the window is bounded by what placement left free,
    # which the greedy fill makes small. Both maps need every placement tier:
    # pipelined_time_s does not charge a tier missing from tier_bw.
    elapsed = pipelined_time_s(
        units, PREFETCH_WINDOW,
        {"GPU HBM": _GPU.bw_Bps, "Host DRAM": HOST_DRAM.bw_Bps,
         "CXL Device DRAM": CXL_DRAM.bw_Bps,
         "CXL Device NAND": CXL_SSD_NAND.bw_Bps},
        {"GPU HBM": _GPU.chunk_latency_s,
         "Host DRAM": HOST_DRAM.chunk_latency_s,
         "CXL Device DRAM": CXL_DRAM.chunk_latency_s,
         "CXL Device NAND": CXL_SSD_NAND.chunk_latency_s},
        inflight_budget=max(0.0, gpu_free + cxl_free + host_free))
    return elapsed, read_stall, kv_write


# Prefill
pf_step, _, _ = run_phase(is_prefill=True)
pf_time = pf_step                       # cost of processing the prompt

# Decode
total_dec, total_read_stall, total_kv_write = 0.0, 0.0, 0.0
per_token_write_stall_pcts = []
for t in range(TOKENS):
    step, rs, kw = run_phase(is_prefill=False, token_step=t)
    step += kv_growth_spill_time_s(
        kv_inc_per_tok * _n_attn * BATCH_SIZE * (PREFILL_TOKENS + t + 1),
        _kv_resident, kv_tail(_kv_split),
        headroom=({"gpu": gpu_free, "host": host_free, "cxl": cxl_free}
                  .get(list(_kv_split)[-1], 0.0) if _kv_split else 0.0))
    total_dec += step
    total_read_stall += rs
    total_kv_write += kw
    per_token_write_stall_pcts.append((kw / step * 100) if step > 0 else 0.0)

avg_dec = total_dec / TOKENS
decode_tps = BATCH_SIZE / avg_dec if avg_dec > 0 else 0.0

total_model_bytes = sum(L["bytes"] for L in layers)
cold_load = ssd_time_s(total_model_bytes)

weight_rd = total_model_bytes
kv_wr = kv_inc_per_tok * BATCH_SIZE * TOKENS
total_io = weight_rd + kv_wr
avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)

print(f"Decode throughput: {decode_tps:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / pf_time:.6f}")
print(f"Overall throughput: "
      f"{(PREFILL_TOKENS + TOKENS) / (cold_load + pf_time + total_dec):.6f}")
print(f"Read_Op_Percent: {weight_rd / total_io * 100:.4f}%")
print(f"Write_Op_Percent: {kv_wr / total_io * 100:.4f}%")
print(f"Read_Ratio: {weight_rd / total_io * 100:.4f}%")
print(f"Read_Stall_Time_s:  {total_read_stall / TOKENS:.6f}")
print(f"Write_Stall_Time_s: {total_kv_write / TOKENS:.6f}")
print(f"Write_Stall_Pct:    {avg_write_stall_pct:.4f}%")
print(f"Write_Util_Pct:     0.0000%")
print(f"Per_Token_Write_Stall_Pcts: "
      f"{','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")
