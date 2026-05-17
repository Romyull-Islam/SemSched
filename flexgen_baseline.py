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
from tiers import GiB, HOST_DRAM, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM, HOT_LAYERS_BY_NAME
from sim_cfg import (
    TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core,
    parallel_efficiency, host_dram_capacity_bytes, BATCH_SIZE
)

PL_HOST_DRAM            = "Host DRAM"
PL_HOST_SSD             = "Host SSD (NVMe)"
PREFILL_TOKENS          = 512
PREFILL_FLOP_MULTIPLIER = 15.0


# ── Timing helpers ────────────────────────────────────────────────────────────
def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def ssd_time_s(n):  return transfer_time_s(
    n, Tier("Host SSD", NVME_STREAM_BW, NVME_STREAM_LAT_S))


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
placement = [PL_HOST_SSD] * len(layers)
host_free  = host_dram_capacity_bytes

for n in HOT_LAYERS_BY_NAME:
    idx = name_to_idx.get(n)
    if idx is not None:
        sz = layers[idx]["bytes"]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

for i, L in enumerate(layers):
    if placement[i] == PL_HOST_DRAM:
        continue
    sz = L["bytes"] + kv_cache_increment[L["name"]] * PREFILL_TOKENS
    if sz <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= sz
    else:
        break


# ── Phase 1: PREFILL ──────────────────────────────────────────────────────────
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s   = compute_time_s(L["flops"] * PREFILL_FLOP_MULTIPLIER)
    mem_time = dram_time_s(L["bytes"]) if placement[i] == PL_HOST_DRAM \
               else ssd_time_s(L["bytes"])
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
        comp_s   = compute_time_s(L["flops"])
        mem_time = dram_time_s(L["bytes"]) if placement[i] == PL_HOST_DRAM \
                   else ssd_time_s(L["bytes"])

        # Growing-KV: attention layers read all prior K/V from cache.
        # KV lives in Host DRAM in FlexGen, so this is a host-DRAM read.
        kv_inc_l = kv_cache_increment[L["name"]]
        if kv_inc_l > 0:
            kv_positions_cached = PREFILL_TOKENS + token_step
            kv_read_bytes = kv_positions_cached * kv_inc_l * BATCH_SIZE
            mem_time += dram_time_s(kv_read_bytes)

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
            kv_stall                = dram_time_s(kv_write_bytes)  # always
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
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f}")
print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / (cold_load + pf_time + dec_time):.3f}")

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
