# FlexGen baseline (Sheng et al., ICML 2023), adapted to a CMM-H machine.
#
# Weights and KV cache are placed as per-tier fractions over GPU, host DRAM,
# CXL device DRAM and device NAND, in that order. At decode step t attention
# reads (PREFILL_TOKENS + t) x per-token KV bytes x BATCH_SIZE from the tiers
# holding the cache.
#
# Stalls are reported separately:
#   read stall:  max(0, mem_time - comp_s), weight and KV fetch beyond compute
#   write stall: kv_time_s(kv_bytes), the KV write, serialized after the weight
#                read on any bus because FlexGen has no duplex path

import math
import pandas as pd
from tiers import (GiB, HOST_DRAM, CXL_DRAM, GPU_HBM, CXL_SSD_NAND, Tier,
                   NVME_STREAM_BW, NVME_STREAM_LAT_S, transfer_time_s,
                   kv_growth_spill_time_s)
from model_cfg import build_layers, BYTES_PER_PARAM, HOT_LAYERS_BY_NAME
from pipeline import pipelined_time_s

# FlexGen's block schedule overlaps "the weights load of the next layer" with
# the current layer's compute (Algorithm 1, three CUDA streams): one layer of
# lookahead.
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
    TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core,
    parallel_efficiency, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes,
    BATCH_SIZE
)

PL_GPU_HBM              = "GPU HBM"
PL_HOST_DRAM            = "Host DRAM"
# FlexGen's middle tier is "CPU memory", byte-addressable memory the CPU can
# load and store. CXL.mem exposes the CMM-H device DRAM as a NUMA node, so it
# qualifies, as it does for every other policy here. It is slower than host DDR
# (27 vs 38.4 GB/s), so it sits below host DRAM in the cascade.
PL_CXL_DEV_DRAM         = "CXL Device DRAM"
PL_HOST_SSD             = "CXL Device NAND"
PREFILL_TOKENS          = 512
PREFILL_FLOP_MULTIPLIER = 15.0


def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def gpu_time_s(n):
    return transfer_time_s(n, GPU_HBM)


def ssd_time_s(n):
    """Backing store: on CMM-H this is the device's own NAND, not a host drive.

    CMM-H is byte-addressable memory, not a filesystem. Once the model is mapped
    into the device's address space, a page that misses the device DRAM cache is
    served from its NAND. The host NVMe is read once to load the model file and
    is not on the steady-state path.
    """
    return transfer_time_s(n, CXL_SSD_NAND)


layers      = build_layers(sequence_length=PREFILL_TOKENS)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# KV bytes (K and V) that one token adds to each decoder block.
kv_cache_increment = {}
for L in layers:
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 8)
        kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * BYTES_PER_PARAM
    else:
        kv_cache_increment[L["name"]] = 0


# Placement. FlexGen is a GPU system ("with a Single GPU") that places data
# across GPU, CPU and disk; here the cascade is GPU, host DRAM, CXL device DRAM,
# then device NAND.
placement = [PL_HOST_SSD] * len(layers)

# KV capacity reservation. The KV cache is resident for the whole session, so
# the bytes it occupies cannot hold weights. It is reserved at the generation
# mean, PREFILL + TOKENS/2: the cache grows during decode, so reserving the final
# size starves the early steps and the initial size over-commits the later ones.
_kv_resident = int(sum(kv_cache_increment.values()) * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS / 2.0))

# The LP's cache variables (cg, cc, cd) are GPU, CPU and disk percentages, so
# the cache cascades through the same hierarchy as the weights, GPU first
# (Sec 4.3). What does not fit in one tier goes to the next and is charged that
# tier's bandwidth; none of it is dropped. Weights are placed after this block
# and get what the KV reservation leaves in each tier.
_kv_gpu  = min(_kv_resident, max(0, gpu_hbm_capacity_bytes
                                 - min(sum(L["bytes"] for L in layers),
                                       gpu_hbm_capacity_bytes)))
# If weights alone exceed HBM, no KV fits there; otherwise KV takes what the
# weights leave. This mirrors the LP's joint capacity constraint per device.
_wtotal  = sum(L["bytes"] for L in layers)
_kv_gpu  = min(_kv_resident, max(0, gpu_hbm_capacity_bytes - _wtotal))
_kv_host = min(_kv_resident - _kv_gpu, host_dram_capacity_bytes)
_kv_cxl  = min(_kv_resident - _kv_gpu - _kv_host, cxl_dev_dram_capacity_bytes)
_kv_nand = max(0, _kv_resident - _kv_gpu - _kv_host - _kv_cxl)
_kv_frac = {"gpu":  _kv_gpu  / _kv_resident if _kv_resident else 0.0,
            "host": _kv_host / _kv_resident if _kv_resident else 1.0,
            "cxl":  _kv_cxl  / _kv_resident if _kv_resident else 0.0,
            "nand": _kv_nand / _kv_resident if _kv_resident else 0.0}

def kv_time_s(n):
    return (gpu_time_s(n * _kv_frac["gpu"]) + dram_time_s(n * _kv_frac["host"])
            + cxl_time_s(n * _kv_frac["cxl"]) + ssd_time_s(n * _kv_frac["nand"]))

gpu_free   = max(0, gpu_hbm_capacity_bytes - _kv_gpu)
host_free  = max(0, host_dram_capacity_bytes - _kv_host)
cxl_free   = max(0, cxl_dev_dram_capacity_bytes - _kv_cxl)

# Working-memory reservation. FlexGen App. A makes working memory an LP
# constraint, gpu_w^p = 2(1-wg)(8h1^2 + 4h1h2) + ..., where the leading 2 is
# the double buffer that funds load_weight(i, j+1, k) in Algorithm 1 (LIA's
# gpu_buff_1/gpu_buff_2). We reserve 2x the largest unit and the decode pipeline
# stages into this same reserve: residency given up and staging gained are one
# trade, so they must use the same bytes. The App. A KV read term is omitted
# since units carry their KV reads separately, which makes the reserve smaller
# than FlexGen's. Taken from the fast tiers in the LP's order: GPU, host, CXL.
_FLEXGEN_WS = 2.0 * max(L["bytes"] for L in layers)
_ws_rem = _FLEXGEN_WS
_take = min(_ws_rem, gpu_free);  gpu_free  -= _take; _ws_rem -= _take
_take = min(_ws_rem, host_free); host_free -= _take; _ws_rem -= _take
_take = min(_ws_rem, cxl_free);  cxl_free  -= _take; _ws_rem -= _take
_FLEXGEN_STAGING = _FLEXGEN_WS - _ws_rem      # what the tiers could hold

# Hot layers are pinned whole, fastest tier first.
for n in HOT_LAYERS_BY_NAME:
    idx = name_to_idx.get(n)
    if idx is not None:
        sz = layers[idx]["bytes"]
        if sz <= gpu_free:
            placement[idx] = PL_GPU_HBM
            gpu_free -= sz
        elif sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz
        elif sz <= cxl_free:
            placement[idx] = PL_CXL_DEV_DRAM
            cxl_free -= sz

# The rest of the weights are placed as per-tier fractions. FlexGen's LP has
# independent percentages for weights (wg, wc, wd), cache (cg, cc, cd) and
# activations (hg, hc, hd) (Sec 4.3, Eq. 1), so a layer may be split across
# tiers. Each tier is filled to capacity in order, which is the LP optimum when
# capacity is the only binding constraint.
frac_gpu  = [0.0] * len(layers)
frac_host = [0.0] * len(layers)
frac_cxl  = [0.0] * len(layers)
for i, L in enumerate(layers):
    if placement[i] in (PL_GPU_HBM, PL_HOST_DRAM, PL_CXL_DEV_DRAM):
        frac_gpu[i]  = 1.0 if placement[i] == PL_GPU_HBM  else 0.0
        frac_cxl[i]  = 1.0 if placement[i] == PL_CXL_DEV_DRAM else 0.0
        frac_host[i] = 1.0 if placement[i] == PL_HOST_DRAM else 0.0
        continue
    rem = float(L["bytes"])
    take = min(rem, gpu_free)
    if take > 0:
        frac_gpu[i] = take / L["bytes"]; gpu_free -= take; rem -= take
    take = min(rem, host_free)
    if take > 0:
        frac_host[i] = take / L["bytes"]; host_free -= take; rem -= take
    take = min(rem, cxl_free)
    if take > 0:
        frac_cxl[i] = take / L["bytes"]; cxl_free -= take; rem -= take
    placement[i] = (PL_GPU_HBM if frac_gpu[i] >= 0.999 else
                    PL_HOST_DRAM if frac_host[i] >= 0.999 else
                    PL_CXL_DEV_DRAM if frac_cxl[i] >= 0.999 else PL_HOST_SSD)


# Prefill: one pass over the weights for the whole prompt.
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s   = compute_time_s(L["flops"] * PREFILL_TOKENS * BATCH_SIZE)
    b = L["bytes"]
    fg, fh, fc = frac_gpu[i], frac_host[i], frac_cxl[i]
    fd = max(0.0, 1.0 - fg - fh - fc)
    mem_time = (gpu_time_s(b * fg) + dram_time_s(b * fh)
                + cxl_time_s(b * fc) + ssd_time_s(b * fd))
    prefill_latency += max(comp_s, mem_time)

# Decode accumulators.
total_read_stall_s         = 0.0
total_kv_write_stall_s     = 0.0
per_token_latency          = 0.0
per_token_read_stall_pcts  = []
per_token_write_stall_pcts = []

# Decode.
for token_step in range(TOKENS):
    units              = []
    step_time_s        = 0.0
    step_read_stall_s  = 0.0
    step_write_stall_s = 0.0

    for i, L in enumerate(layers):
        comp_s   = compute_time_s(L["flops"] * BATCH_SIZE)
        b = L["bytes"]
        fg, fh, fc = frac_gpu[i], frac_host[i], frac_cxl[i]
        fd = max(0.0, 1.0 - fg - fh - fc)
        mem_time = (gpu_time_s(b * fg) if fg > 0 else 0.0) \
                 + (dram_time_s(b * fh) if fh > 0 else 0.0) \
                 + (cxl_time_s(b * fc) if fc > 0 else 0.0) \
                 + (ssd_time_s(b * fd) if fd > 0 else 0.0)

        # Growing KV: attention reads every cached position from the tiers
        # holding the cache (see _kv_frac).
        kv_inc_l = kv_cache_increment[L["name"]]
        if kv_inc_l > 0:
            kv_positions_cached = PREFILL_TOKENS + token_step
            kv_read_bytes = kv_positions_cached * kv_inc_l * BATCH_SIZE
            mem_time += kv_time_s(kv_read_bytes)

        # Read stall: weight + KV fetch in excess of compute
        read_stall = max(0.0, mem_time - comp_s)
        total_read_stall_s += read_stall
        step_read_stall_s  += read_stall

        ltime = max(comp_s, mem_time)

        # KV write stall. FlexGen issues its transfers in sequence and has no
        # duplex path, so the write is serialized after the weight read even
        # when the two use different buses.
        kv_stall = 0.0
        if kv_inc_l > 0:
            kv_write_bytes          = kv_inc_l * BATCH_SIZE
            kv_stall                = kv_time_s(kv_write_bytes)   # always
            total_kv_write_stall_s += kv_stall
            step_write_stall_s     += kv_stall

        _by = {"GPU HBM": b*fg, "Host DRAM": b*fh,
               "CXL Device DRAM": b*fc, "CXL Device NAND": b*fd}
        if kv_inc_l > 0:
            for _t, _f in (("Host DRAM", _kv_frac["host"]),
                           ("CXL Device DRAM", _kv_frac["cxl"]),
                           ("CXL Device NAND", _kv_frac["nand"])):
                _by[_t] = _by.get(_t, 0.0) + kv_read_bytes * _f
        units.append((_by, comp_s))
        step_time_s += kv_stall           # KV write, serialized after the read


    # Tiers run concurrently and a read may be issued PREFETCH_DEPTH layers
    # early, staged into the App. A working buffer reserved above. KV past its
    # reservation then displaces weights out of the tier holding it.
    step_time_s += pipelined_time_s(units, PREFETCH_DEPTH, TIER_BW, TIER_LAT,
                                    inflight_budget=max(0.0, _FLEXGEN_STAGING))
    step_time_s += kv_growth_spill_time_s(
        sum(kv_cache_increment.values()) * BATCH_SIZE * (PREFILL_TOKENS + token_step + 1),
        _kv_resident,
        GPU_HBM if _kv_frac["gpu"] >= 0.5 else HOST_DRAM)
    per_token_latency += step_time_s

    read_pct  = (step_read_stall_s  / step_time_s * 100) if step_time_s > 0 else 0.0
    write_pct = (step_write_stall_s / step_time_s * 100) if step_time_s > 0 else 0.0
    per_token_read_stall_pcts.append(read_pct)
    per_token_write_stall_pcts.append(write_pct)

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)


# Overall throughput includes a cold load of the model from device NAND.
total_model_bytes = sum(L["bytes"] for L in layers)
cold_load         = ssd_time_s(total_model_bytes)
pf_time           = prefill_latency
dec_time          = per_token_latency


# Op split of one decode step: every weight is read, one token of KV written.
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
kv_layers           = sum(1 for L in layers if kv_cache_increment[L["name"]] > 0)

# Separated stall reporting
print(f"Read_Stall_Time_s:  {total_read_stall_s      / TOKENS:.6f}")
print(f"Read_Stall_Pct:     {avg_read_stall_pct:.4f}%")
print(f"Write_Stall_Time_s: {total_kv_write_stall_s  / TOKENS:.6f}")
print(f"Write_Stall_Pct:    {avg_write_stall_pct:.4f}%")
print(f"Total_Stall_Time_s: {(total_read_stall_s + total_kv_write_stall_s) / TOKENS:.6f}")
print(f"Total_Stall_Pct:    {avg_read_stall_pct + avg_write_stall_pct:.4f}%")
print(f"Write_Stall_Count:  {kv_layers}")
print(f"Write_Util_Pct:     0.0000%")
print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")
