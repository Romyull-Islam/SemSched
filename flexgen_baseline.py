# flexgen_baseline.py
# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR: FlexGen — High-Throughput Generative Inference
#
# Decode-phase includes growing-KV reads: at decode step t the attention layer
# reads (PREFILL_TOKENS + t) × per-token-KV-bytes × BATCH_SIZE from host DRAM
# (FlexGen's KV residence tier).
#
# Reference: Sheng et al., ICML 2023.
#
# Stall accounting (separated):
#   Read  stall : max(0, mem_time - comp_s)  — weight fetch blocks compute
#   Write stall : dram_time_s(kv_bytes)      — KV write serialized AFTER weight
#
# KV stall rule (§4 bus model):
#   Weight @ DRAM + KV @ DRAM → SAME bus  → KV write fully serialized
#   Weight @ SSD  + KV @ DRAM → DIFF bus  → KV write concurrent (no stall)
# ─────────────────────────────────────────────────────────────────────────────

import math
import pandas as pd
from tiers import (GiB, HOST_DRAM, CXL_DRAM, GPU_HBM, CXL_SSD_NAND, Tier,
                   NVME_STREAM_BW, NVME_STREAM_LAT_S, transfer_time_s)
from model_cfg import build_layers, BYTES_PER_PARAM, HOT_LAYERS_BY_NAME
from sim_cfg import (
    gpu_hbm_capacity_bytes,
    TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core,
    parallel_efficiency, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes,
    BATCH_SIZE
)

PL_GPU_HBM              = "GPU HBM"
PL_HOST_DRAM            = "Host DRAM"
# FlexGen's middle tier is "CPU memory": byte-addressable memory the CPU can
# load and store from. On a CMM-H machine the device DRAM is exactly that --
# CXL.mem exposes it as a NUMA node -- and every other policy here treats it so:
# LIA places weights in it, CXLAimPod pools it, LLM-in-a-Flash unifies it with
# host DRAM by name. FlexGen alone was denied it, which left it with 44 GB of
# byte-addressable memory where the others had 92, and made its throughput
# invariant to the device cache size -- the tell. It is a slower tier than host
# DDR (27 vs 38.4 GB/s), so it sits below it in the cascade rather than merging.
PL_CXL_DEV_DRAM         = "CXL Device DRAM"
PL_HOST_SSD             = "CXL Device NAND"
PREFILL_TOKENS          = 512
PREFILL_FLOP_MULTIPLIER = 15.0


# ── Timing helpers ────────────────────────────────────────────────────────────
def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def gpu_time_s(n):
    return transfer_time_s(n, GPU_HBM)


def ssd_time_s(n):
    """Backing store. On a CMM-H platform that is the device's own NAND, not a
    separate host drive.

    CMM-H is byte-addressable extended memory, not a filesystem. Once the model
    is mapped into the device's address space, a page absent from the 48 GB DRAM
    cache is served by the device's own NAND -- that is what the hybrid device
    is. The host NVMe is where the model file was read from once; it is not
    where steady-state overflow comes from.

    This previously used NVME_STREAM_BW at 7.6 GB/s for the steady-state path,
    describing a machine that re-reads the model file on every token, while
    every other simulator took its overflow from CMM-H NAND at 5.0 GB/s. The
    policy is unchanged -- fractional placement across GPU, host DRAM and the
    backing store, hot layers pinned first, transfer overlapped with compute.
    Only the provenance of the bytes is corrected, and that is a property of the
    platform rather than of the scheduler. Sec IV-B already describes these as
    CXL-adapted baselines; this makes FlexGen one.
    """
    return transfer_time_s(n, CXL_SSD_NAND)


# ── Build model ───────────────────────────────────────────────────────────────
layers      = build_layers(sequence_length=PREFILL_TOKENS)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

kv_cache_increment = {}
for L in layers:
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 8)
        kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * BYTES_PER_PARAM
    else:
        kv_cache_increment[L["name"]] = 0


# ── Placement ─────────────────────────────────────────────────────────────────
# FlexGen is a GPU system -- its title is "with a Single GPU" and its policy is
# a placement across GPU -> CPU -> disk. Modelling it without a GPU tier removed
# the top level of the hierarchy its LP exists to fill, and cost it the fastest
# memory on the machine. The cascade below is GPU first, then host DRAM, then
# disk, matching the paper's ordering.
placement = [PL_HOST_SSD] * len(layers)

# ── KV capacity reservation (audit A12) ──────────────────────────────────────
# The KV cache is resident for the whole session and occupies the tier it lives
# in, so those bytes are not available to hold weights. Placement previously
# ignored it in every baseline: at INT8 16H+32C that put 83.1 GB of state into
# 76 GB of memory. SemSched reserves it; the baselines did not, which understated
# their memory pressure and so overstated their throughput.
#
# Reserved at the generation mean, PREFILL + TOKENS/2, because the cache grows
# during decode -- reserving the final size starves the early steps and
# reserving the initial size over-commits the later ones.
_kv_resident = int(sum(kv_cache_increment.values()) * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS / 2.0))

# The LP's KV variables are (cg, cc, cd) -- GPU, CPU and DISK percentages -- so
# a cache that exceeds CPU memory is placed on the slower tiers and charged
# their bandwidth. It is never discarded. Reserving it from host DRAM and
# clipping at zero dropped 4.31 GiB at FP16 B=128: FlexGen paid the full
# capacity penalty and none of the transfer cost for the excess.
_kv_host = min(_kv_resident, host_dram_capacity_bytes)
_kv_cxl  = min(_kv_resident - _kv_host, cxl_dev_dram_capacity_bytes)
_kv_nand = max(0, _kv_resident - _kv_host - _kv_cxl)
_kv_frac = {"host": _kv_host / _kv_resident if _kv_resident else 1.0,
            "cxl":  _kv_cxl  / _kv_resident if _kv_resident else 0.0,
            "nand": _kv_nand / _kv_resident if _kv_resident else 0.0}

def kv_time_s(n):
    return (dram_time_s(n * _kv_frac["host"]) + cxl_time_s(n * _kv_frac["cxl"])
            + ssd_time_s(n * _kv_frac["nand"]))

gpu_free   = gpu_hbm_capacity_bytes
host_free  = max(0, host_dram_capacity_bytes - _kv_host)
cxl_free   = max(0, cxl_dev_dram_capacity_bytes - _kv_cxl)

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

# FlexGen's LP searches over NINE variables -- (wg, wc, wd) for weights,
# (cg, cc, cd) for the KV cache and (hg, hc, hd) for activations -- as
# INDEPENDENT percentages per tier (Sec 4.3, Eq. 1). Bundling a layer's KV into
# its weight footprint here made every decoder block fail the GPU test on the
# KV term alone, so the accelerator only ever took lm_head and final_norm and
# the tier gained 0%. Weights are placed on their own, each tier filled to
# capacity, matching the LP's structure rather than a per-layer all-or-nothing.
# FlexGen's LP does NOT place whole layers. Its variables (wg, wc, wd) are
# PERCENTAGES of the weight tensors on each tier (Sec 4.3, Eq. 1) -- a layer may
# be split 30/70 across GPU and disk. All-or-nothing placement forfeits that,
# stranding capacity whenever the next layer does not fit whole. Fractions are
# tracked per layer and each tier is filled exactly to capacity, which is what
# the LP's optimum looks like when the only constraint that binds is capacity.
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
    # FlexGen's LP fills each tier to capacity; the original `break` abandoned
    # placement at the first layer that did not fit, leaving fast memory unused.
    # Continue so the tier is actually filled.


# ── Phase 1: PREFILL ──────────────────────────────────────────────────────────
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s   = compute_time_s(L["flops"] * PREFILL_TOKENS * BATCH_SIZE)
    b = L["bytes"]
    fg, fh, fc = frac_gpu[i], frac_host[i], frac_cxl[i]
    fd = max(0.0, 1.0 - fg - fh - fc)
    mem_time = (gpu_time_s(b * fg) + dram_time_s(b * fh)
                + cxl_time_s(b * fc) + ssd_time_s(b * fd))
    prefill_latency += max(comp_s, mem_time)

# =========================================================================
# ADD THESE 5 LINES RIGHT HERE
# =========================================================================
total_read_stall_s         = 0.0
total_kv_write_stall_s     = 0.0
per_token_latency          = 0.0
per_token_read_stall_pcts  = []
per_token_write_stall_pcts = []

# ── Phase 2: DECODE — replace KV stall block entirely ────────────────────────
for token_step in range(TOKENS):
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

        # Growing-KV: attention layers read all prior K/V from cache.
        # KV lives in Host DRAM in FlexGen, so this is a host-DRAM read.
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

        # ── KV write stall ────────────────────────────────────────────────────
        # FlexGen §4: KV cache ALWAYS lives in Host DRAM.
        # Write is ALWAYS serialized after weight read — FlexGen has no
        # duplex capability. Bus separation (SSD vs DRAM) does NOT help
        # because FlexGen issues ops sequentially, not concurrently.
        kv_stall = 0.0
        if kv_inc_l > 0:
            kv_write_bytes          = kv_inc_l * BATCH_SIZE
            kv_stall                = kv_time_s(kv_write_bytes)   # always
            total_kv_write_stall_s += kv_stall
            step_write_stall_s     += kv_stall

        step_time_s += ltime + kv_stall   # serial: weight→compute→kv_write


    per_token_latency += step_time_s

    read_pct  = (step_read_stall_s  / step_time_s * 100) if step_time_s > 0 else 0.0
    write_pct = (step_write_stall_s / step_time_s * 100) if step_time_s > 0 else 0.0
    per_token_read_stall_pcts.append(read_pct)
    per_token_write_stall_pcts.append(write_pct)

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)


# ── Overall throughput ────────────────────────────────────────────────────────
total_model_bytes = sum(L["bytes"] for L in layers)
cold_load         = ssd_time_s(total_model_bytes)
pf_time           = prefill_latency
dec_time          = per_token_latency


# ── Reporting ─────────────────────────────────────────────────────────────────
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
