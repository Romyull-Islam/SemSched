# LIA baseline (Kim et al., ISCA '25): weights tiered below the GPU into CXL
# memory, KV cache kept in host DDR.
#
# Each decode step is timed by pipeline.pipelined_time_s over the layers' weight
# reads and growing-KV reads; at step t attention reads
# (PREFILL_TOKENS + t) x per-token KV bytes x BATCH_SIZE from host DDR.
#
# Reported stalls (diagnostics, not added to the step time):
#   read stall  : max(0, weight_mem_time - comp_s), the CXL DRAM (or NAND)
#                 weight fetch in excess of compute
#   write stall : time to write the step's new KV to host DRAM

import math
import pandas as pd
from tiers import kv_growth_spill_time_s
from tiers import (
    GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND,
    transfer_time_s, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from tiers import GPU_HBM
from pipeline import pipelined_time_s

# One layer of weight prefetch. LIA's ISCA'25 artifact (ece-fast-lab/ISCA-2025-LIA,
# lia/modeling_opt.py) loads layer idx+1 into an alternating double buffer
# (gpu_buff_1 / gpu_buff_2) on a separate load_weight_stream, with overlap on by
# default, in the `decoding_policy in [2, 4]` branch that every CXL run in
# scripts/cxl_offloading.sh uses. FlexGen gets the same depth for the same mechanism.
PREFETCH_DEPTH = 1
TIER_BW  = {"GPU HBM": GPU_HBM.bw_Bps, "Host DRAM": HOST_DRAM.bw_Bps,
            "CXL Device DRAM": CXL_DRAM.bw_Bps,
            "CXL Device NAND": CXL_SSD_NAND.bw_Bps}
TIER_LAT = {"GPU HBM": GPU_HBM.chunk_latency_s,
            "Host DRAM": HOST_DRAM.chunk_latency_s,
            "CXL Device DRAM": CXL_DRAM.chunk_latency_s,
            "CXL Device NAND": CXL_SSD_NAND.chunk_latency_s}
from sim_cfg import (
    gpu_hbm_capacity_bytes,
    TOKENS, cpu_freq_hz, cpu_cores,
    flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, BATCH_SIZE
)

PL_GPU_HBM      = "GPU HBM"
PL_HOST_DRAM    = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

PREFILL_TOKENS          = 512
PREFILL_FLOP_MULTIPLIER = 15.0


def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def ssd_time_s(n):  return transfer_time_s(
    n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))


layers = build_layers(sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}


# Placement: GPU HBM, then CXL device DRAM, then NAND. LIA is single-GPU
# inference (Sapphire Rapids with an A100 or H100), so CXL is the tier below the
# GPU. Host DRAM holds only the KV cache.
placement = []
gpu_free  = gpu_hbm_capacity_bytes

# KV is reserved at the generation mean, PREFILL_TOKENS + TOKENS/2. What does not
# fit in host DRAM (e.g. 20.31 GiB of KV against a 16 GiB host at FP16 B=128)
# goes to CXL device DRAM, where it displaces weights and is read at 27 GB/s
# instead of 38.4 GB/s.
_kv_resident = int(sum(kv_inc.values()) * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS / 2.0))
_kv_in_host  = min(_kv_resident, host_dram_capacity_bytes)
_kv_in_cxl   = max(0, _kv_resident - host_dram_capacity_bytes)
_kv_cxl_frac = (_kv_in_cxl / _kv_resident) if _kv_resident else 0.0

cxl_free  = max(0, cxl_dev_dram_capacity_bytes - _kv_in_cxl)
for L in layers:
    if L["bytes"] <= gpu_free:
        placement.append(PL_GPU_HBM)
        gpu_free -= L["bytes"]
    elif L["bytes"] <= cxl_free:
        placement.append(PL_CXL_DEV_DRAM)
        cxl_free -= L["bytes"]
    else:
        placement.append(PL_CXL_DEV_NAND)

# Staging for the next-layer prefetch. pipeline.py moves the issue point earlier
# than unit i-1 only if the in-flight bytes fit in this budget, and the cascade
# leaves gpu_free and cxl_free near 0, so without it depth 1 behaves as depth 0.
# The size follows the artifact: two weight buffers alternated by layer parity,
# i.e. 2x the largest layer. They can use host DRAM the KV leaves free (none at
# FP16 B=128, some at INT8).
_LIA_HOST_FREE = max(0.0, host_dram_capacity_bytes - _kv_in_host)
_LIA_DOUBLE_BUF = 2.0 * max(L["bytes"] for L in layers)
_LIA_STAGING = min(_LIA_DOUBLE_BUF, gpu_free + cxl_free + _LIA_HOST_FREE)


# Prefill
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s = compute_time_s(L["flops"] * PREFILL_TOKENS * BATCH_SIZE)
    if placement[i] == PL_CXL_DEV_DRAM:
        mem_time = cxl_time_s(L["bytes"])
    elif placement[i] == PL_CXL_DEV_NAND:
        mem_time = nand_time_s(L["bytes"])
    else:
        mem_time = dram_time_s(L["bytes"])
    prefill_latency += max(comp_s, mem_time)


# Decode
total_read_stall_s         = 0.0   # CXL/NAND weight fetch in excess of compute
total_kv_write_stall_s     = 0.0   # KV write time to host DRAM
per_token_latency          = 0.0
kv_layer_count             = 0
per_token_read_stall_pcts  = []
per_token_write_stall_pcts = []

for token_step in range(TOKENS):
    units              = []
    step_time_s        = 0.0
    step_read_stall_s  = 0.0
    step_write_stall_s = 0.0

    for i, L in enumerate(layers):
        weight_sz = L["bytes"]
        kv_scaled = kv_inc[L["name"]] * BATCH_SIZE
        comp_s    = compute_time_s(L["flops"] * BATCH_SIZE)

        if placement[i] == PL_CXL_DEV_DRAM:
            weight_mem_time = cxl_time_s(weight_sz)
        elif placement[i] == PL_CXL_DEV_NAND:
            weight_mem_time = nand_time_s(weight_sz)
        else:
            weight_mem_time = dram_time_s(weight_sz)

        # Attention reads all cached K/V from host DRAM, sharing the DDR bus with
        # the KV write.
        kv_read_time = 0.0
        if kv_scaled > 0:
            kv_positions_cached = PREFILL_TOKENS + token_step
            kv_read_bytes = kv_positions_cached * kv_inc[L["name"]] * BATCH_SIZE
            # The share that did not fit in host DRAM is read from device DRAM.
            kv_read_time  = (dram_time_s(kv_read_bytes * (1.0 - _kv_cxl_frac))
                             + (cxl_time_s(kv_read_bytes * _kv_cxl_frac)
                                if _kv_cxl_frac > 0 else 0.0))

        read_stall = max(0.0, weight_mem_time - comp_s)
        total_read_stall_s += read_stall
        step_read_stall_s  += read_stall

        # Weight reads use the CXL link and KV traffic uses host DDR, so ltime is
        # the max of compute, weight read, and KV read + write (which share DDR).
        kv_write_time = 0.0
        if kv_scaled > 0:
            kv_write_time           = (dram_time_s(kv_scaled * (1.0 - _kv_cxl_frac))
                                       + (cxl_time_s(kv_scaled * _kv_cxl_frac)
                                          if _kv_cxl_frac > 0 else 0.0))
            total_kv_write_stall_s += kv_write_time
            step_write_stall_s     += kv_write_time
            if token_step == 0:
                kv_layer_count += 1

        kv_total_ddr_time = kv_read_time + kv_write_time
        ltime        = max(comp_s, weight_mem_time, kv_total_ddr_time)
        _by = {placement[i]: weight_sz}
        if kv_scaled > 0:
            for _t, _f in (("Host DRAM", 1.0 - _kv_cxl_frac),
                           ("CXL Device DRAM", _kv_cxl_frac)):
                _by[_t] = _by.get(_t, 0.0) + kv_read_bytes * _f
        units.append((_by, comp_s))
        step_time_s += 0.0

    step_time_s += pipelined_time_s(units, PREFETCH_DEPTH, TIER_BW, TIER_LAT,
                                    inflight_budget=max(0.0, _LIA_STAGING))
    step_time_s += kv_growth_spill_time_s(
        sum(kv_inc.values()) * BATCH_SIZE * (PREFILL_TOKENS + token_step + 1),
        _kv_resident, HOST_DRAM)
    per_token_latency += step_time_s

    read_pct  = (step_read_stall_s  / step_time_s * 100) if step_time_s > 0 else 0.0
    write_pct = (step_write_stall_s / step_time_s * 100) if step_time_s > 0 else 0.0
    per_token_read_stall_pcts.append(read_pct)
    per_token_write_stall_pcts.append(write_pct)

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)


# Overall throughput includes a cold load of the weights from NVMe.
total_model_bytes = sum(L["bytes"] for L in layers)
cold_load         = ssd_time_s(total_model_bytes)
pf_time           = prefill_latency
dec_time          = per_token_latency


# Reporting
total_weight_rd = sum(L["bytes"] for L in layers)
total_kv_wr     = sum(L.get("kv_cache_bytes", 0) for L in layers) / PREFILL_TOKENS
total_io_vol    = total_weight_rd + total_kv_wr

print(f"Read_Op_Percent: {(total_weight_rd / total_io_vol) * 100:.4f}%")
print(f"Write_Op_Percent: {(total_kv_wr / total_io_vol) * 100:.4f}%")
print(f"Read_Ratio: 100.0000%")

print(f"Decode throughput: {decode_tps:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.6f}")
print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / (cold_load + pf_time + dec_time):.6f}")

avg_read_stall_pct  = sum(per_token_read_stall_pcts)  / len(per_token_read_stall_pcts)
avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)

print(f"Read_Stall_Time_s:  {total_read_stall_s      / TOKENS:.6f}")
print(f"Read_Stall_Pct:     {avg_read_stall_pct:.4f}%")
print(f"Write_Stall_Time_s: {total_kv_write_stall_s  / TOKENS:.6f}")
print(f"Write_Stall_Pct:    {avg_write_stall_pct:.4f}%")
print(f"Total_Stall_Time_s: {(total_read_stall_s + total_kv_write_stall_s) / TOKENS:.6f}")
print(f"Total_Stall_Pct:    {avg_read_stall_pct + avg_write_stall_pct:.4f}%")
print(f"Write_Util_Pct:     0.0000%")
print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")
