# cxlaimpod_baseline.py
# CXL-native baseline modeled on CXLAimPod (Yang et al., 2025) [yang2025cxlaimpod].
#
# POLICY (this baseline's own design, preserved):
#   - Flat capacity-ordered tiering: Host DRAM -> CXL device DRAM -> CXL NAND,
#     with decoder blocks treated as monolithic units (no sub-layer awareness).
#   - Device-side sequential prefetcher with a fixed lookahead window (k = 4).
#   - Blind LRU eviction on the device cache, with no notion of layer type.
#
# HARDWARE MODEL (shared verbatim with FlexGen / LIA / LLMFlash / SemSched):
#   - Same compute engine and FLOP accounting, batch-scaled:
#       decode  FLOPs = 2 * P * B      prefill FLOPs = 2 * P * seq_len * B
#   - Same tier bandwidths/latencies, same growing-KV model, same metric
#     conventions (decode TPS = mean over decode steps; prefill TPS per sequence).
#
# A baseline may differ only by its own paper's policy, never by our modeling of
# the machine. See Sec. IV-B, "Baseline fidelity principle".

import math

from tiers import (HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
                   Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S)
from model_cfg import build_layers, BYTES_PER_PARAM, DEFAULT_MODEL_CFG
from sim_cfg import (TOKENS, BATCH_SIZE, cpu_freq_hz, cpu_cores,
                     flops_per_cycle_per_core, parallel_efficiency,
                     host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes)

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


# ── Model and flat capacity-ordered placement ────────────────────────────────
cfg = DEFAULT_MODEL_CFG()
layers = build_layers(cfg, sequence_length=PREFILL_TOKENS)
head_dim = cfg.emb_dim // cfg.q_heads
kv_inc_per_tok = 2 * cfg.kv_heads * head_dim * BYTES_PER_PARAM

placement = [None] * len(layers)
host_free, cxl_free = host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes
for i, L in enumerate(layers):
    if L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    elif L["bytes"] <= cxl_free:
        placement[i] = PL_CXL_DEV_DRAM
        cxl_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND


class CXLPrefetcher:
    """Sequential device-side prefetcher with blind LRU eviction."""

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
            ev = self.resident.pop(0)      # blind LRU: no layer-type awareness
            self.mem_used -= layers[ev]["bytes"]
            self.arrival.pop(ev, None)
        self.resident.append(idx)
        self.mem_used += size
        self.arrival[idx] = finish
        return finish


def run_phase(is_prefill, token_step=0):
    """One prefill token or one decode step, across all layers."""
    pf = CXLPrefetcher()
    elapsed = 0.0
    read_stall = 0.0
    kv_write = 0.0

    for i, L in enumerate(layers):
        for k in range(1, PREFETCH_WINDOW + 1):
            j = i + k
            if j < len(layers) and placement[j] == PL_CXL_DEV_NAND:
                pf.schedule(j, layers[j]["bytes"], elapsed)

        # Batch-scaled FLOPs; prefill additionally scales with sequence length.
        flops = L["flops"] * BATCH_SIZE * (PREFILL_TOKENS if is_prefill else 1)
        comp = compute_time_s(flops)

        if placement[i] == PL_HOST_DRAM:
            mem = dram_time_s(L["bytes"])
        elif placement[i] == PL_CXL_DEV_DRAM:
            mem = cxl_time_s(L["bytes"])
        else:
            arrival = pf.schedule(i, L["bytes"], elapsed)
            wait = max(0.0, arrival - elapsed)
            read_stall += wait
            mem = wait + cxl_time_s(L["bytes"])

        # Growing KV: attention reads all cached positions so far. CXLAimPod
        # pools device memory, so KV is served from the CXL DRAM tier.
        kv_bytes = L.get("kv_cache_bytes", 0)
        if kv_bytes > 0 and not is_prefill:
            positions = PREFILL_TOKENS + token_step
            mem += cxl_time_s(positions * kv_inc_per_tok * BATCH_SIZE)
            # No duplex scheduling in this baseline: the KV write is serialized.
            w = cxl_time_s(kv_inc_per_tok * BATCH_SIZE)
            kv_write += w
            elapsed += w

        elapsed += max(comp, mem)

    return elapsed, read_stall, kv_write


# ── Prefill ──────────────────────────────────────────────────────────────────
pf_step, _, _ = run_phase(is_prefill=True)
pf_time = pf_step                       # cost of processing the prompt

# ── Decode ───────────────────────────────────────────────────────────────────
total_dec, total_read_stall, total_kv_write = 0.0, 0.0, 0.0
per_token_write_stall_pcts = []
for t in range(TOKENS):
    step, rs, kw = run_phase(is_prefill=False, token_step=t)
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
