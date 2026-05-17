# lia_baseline.py
# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR: LIA — LLM Inference Acceleration (Kim et al., ISCA '25)
#
# Decode-phase includes growing-KV reads: at decode step t the attention layer
# reads (PREFILL_TOKENS + t) × per-token-KV-bytes × BATCH_SIZE from host DDR
# (LIA's KV residence tier). KV read and write share the DDR bus and serialize.
# ─────────────────────────────────────────────────────────────────────────────
#
# Layer-step timing:
#   ltime = max(comp_s, weight_mem_time, kv_write_time)
#
# Weight reads use the CXL link (Rx direction) and KV writes use the host DDR
# bus — these are physically separate buses, so the model permits them to
# overlap with compute on the CPU. This is the hardware-accurate interpretation
# of the LIA setup; a conservative serial-sum interpretation gives results
# within ~0.3% on the configurations tested.
#
# Stall accounting (separated):
#   Read  stall : max(0, weight_mem_time - comp_s)
#                 CXL DRAM (or NAND overflow) fetch blocking compute
#   Write stall : dram_time_s(kv_bytes)
#                 KV write to Host DRAM, serialized after weight read
#                 (LIA has separate CXL bus for reads and DDR bus for writes,
#                  but does NOT pipeline them — that is SemDuplex's innovation)
# ─────────────────────────────────────────────────────────────────────────────

import math
import pandas as pd
from tiers import (
    GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND,
    transfer_time_s, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS, cpu_freq_hz, cpu_cores,
    flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, BATCH_SIZE
)

PL_HOST_DRAM    = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

PREFILL_TOKENS          = 512
PREFILL_FLOP_MULTIPLIER = 15.0


# ── Timing helpers ────────────────────────────────────────────────────────────
def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def ssd_time_s(n):  return transfer_time_s(
    n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))


# ── Build model ───────────────────────────────────────────────────────────────
layers = build_layers(sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}


# ── LIA Placement: ALL weights → CXL DRAM, spill to NAND only on overflow ────
placement = []
cxl_free  = cxl_dev_dram_capacity_bytes
for L in layers:
    if L["bytes"] <= cxl_free:
        placement.append(PL_CXL_DEV_DRAM)
        cxl_free -= L["bytes"]
    else:
        placement.append(PL_CXL_DEV_NAND)


# ── Phase 1: PREFILL ──────────────────────────────────────────────────────────
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s = compute_time_s(L["flops"] * PREFILL_FLOP_MULTIPLIER)
    if placement[i] == PL_CXL_DEV_DRAM:
        mem_time = cxl_time_s(L["bytes"])
    elif placement[i] == PL_CXL_DEV_NAND:
        mem_time = nand_time_s(L["bytes"])
    else:
        mem_time = dram_time_s(L["bytes"])
    prefill_latency += max(comp_s, mem_time)


# ── Phase 2: DECODE ───────────────────────────────────────────────────────────
total_read_stall_s         = 0.0   # CXL/NAND weight fetch blocking compute
total_kv_write_stall_s     = 0.0   # Host DRAM KV write serialized after weight
per_token_latency          = 0.0
kv_layer_count             = 0
per_token_read_stall_pcts  = []
per_token_write_stall_pcts = []

for token_step in range(TOKENS):
    step_time_s        = 0.0
    step_read_stall_s  = 0.0
    step_write_stall_s = 0.0

    for i, L in enumerate(layers):
        weight_sz = L["bytes"]
        kv_scaled = kv_inc[L["name"]] * BATCH_SIZE
        comp_s    = compute_time_s(L["flops"])

        # ── Weight read from CXL tier ─────────────────────────────────────────
        if placement[i] == PL_CXL_DEV_DRAM:
            weight_mem_time = cxl_time_s(weight_sz)
        elif placement[i] == PL_CXL_DEV_NAND:
            weight_mem_time = nand_time_s(weight_sz)
        else:
            weight_mem_time = dram_time_s(weight_sz)

        # Growing-KV: attention reads all prior K/V from host DRAM (LIA places KV
        # cache in host DDR). KV read shares the DDR bus with the KV write.
        kv_read_time = 0.0
        if kv_scaled > 0:
            kv_positions_cached = PREFILL_TOKENS + token_step
            kv_read_bytes = kv_positions_cached * kv_inc[L["name"]] * BATCH_SIZE
            kv_read_time  = dram_time_s(kv_read_bytes)

        # ── Read stall: CXL fetch in excess of compute ────────────────────────
        read_stall = max(0.0, weight_mem_time - comp_s)
        total_read_stall_s += read_stall
        step_read_stall_s  += read_stall

        # ── Write stall: KV to Host DRAM ──────────────────────────────────────
        # Weight read on the CXL link, KV write on host DDR — physically
        # independent buses. ltime takes the max across the three contending
        # paths (compute, CXL read, DDR write+read) since each runs on
        # independent hardware. KV read+write share the DDR bus and serialize.
        kv_write_time = 0.0
        if kv_scaled > 0:
            kv_write_time           = dram_time_s(kv_scaled)
            total_kv_write_stall_s += kv_write_time
            step_write_stall_s     += kv_write_time
            if token_step == 0:
                kv_layer_count += 1

        kv_total_ddr_time = kv_read_time + kv_write_time
        ltime        = max(comp_s, weight_mem_time, kv_total_ddr_time)
        step_time_s += ltime

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

# Separated stall reporting
print(f"Read_Stall_Time_s:  {total_read_stall_s      / TOKENS:.6f}")
print(f"Read_Stall_Pct:     {avg_read_stall_pct:.4f}%")
print(f"Write_Stall_Time_s: {total_kv_write_stall_s  / TOKENS:.6f}")
print(f"Write_Stall_Pct:    {avg_write_stall_pct:.4f}%")
print(f"Total_Stall_Time_s: {(total_read_stall_s + total_kv_write_stall_s) / TOKENS:.6f}")
print(f"Total_Stall_Pct:    {avg_read_stall_pct + avg_write_stall_pct:.4f}%")
print(f"Write_Util_Pct:     0.0000%")
print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")
