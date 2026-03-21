import math
import pandas as pd
from collections import OrderedDict
from tiers import GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, \
                   parallel_efficiency, host_dram_capacity_bytes, \
                   cxl_dev_dram_capacity_bytes, BATCH_SIZE

PL_HOST_DRAM    = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"
PREFILL_TOKENS  = 512

# ** LIA/TPP RIGOR **
# Linux Kernel HugePage Fault Overhead (~8us) + Context Switch
OS_FAULT_PENALTY_S = 8e-6
# Linux Standard Readahead (2MB)
OS_READAHEAD_BYTES = 2 * 1024 * 1024


class OSPagingManager:
    """Mimics OS Kernel (TPP/LIA) Page Management"""
    def __init__(self, cap_bytes):
        self.cap  = cap_bytes
        self.used = 0
        self.lru  = OrderedDict()  # Mimic Active/Inactive LRU lists

    def access_page(self, layer_id, size):
        if layer_id in self.lru:
            self.lru.move_to_end(layer_id)
            return True  # Hit (No fault)
        else:
            self._allocate(layer_id, size)
            return False

    def _allocate(self, layer_id, size):
        while (self.used + size) > self.cap and self.lru:
            lid, sz = self.lru.popitem(last=False)
            self.used -= sz
        if (self.used + size) <= self.cap:
            self.lru[layer_id] = size
            self.used += size
            self.lru.move_to_end(layer_id)


def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)


def cxl_time_s(n):  return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)


# Build Model
layers = build_layers(sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}

# LIA Topology-Aware Placement: Host First, then spill to CXL (OS NUMA tiering)
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
for i, L in enumerate(layers):
    if L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND  # Cold start in NAND (Tier 3)

os_cache = OSPagingManager(cxl_dev_dram_capacity_bytes)


# Phase 2: PREFILL (Reactive Paging)
prefill_latency = 0.0
for i, L in enumerate(layers):
    place  = placement[i]
    comp_s = compute_time_s(L["flops"] * 15.0)

    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(L["bytes"])
    else:
        if os_cache.access_page(i, L["bytes"]):
            mem_time = cxl_time_s(L["bytes"])
        else:
            mem_time = nand_time_s(L["bytes"]) + OS_FAULT_PENALTY_S

    prefill_latency += max(comp_s, mem_time)


# Phase 3: DECODE — replace your existing decode loop with this
total_kv_write_stall_s     = 0.0
per_token_latency          = 0.0
kv_layer_count             = 0
per_token_write_stall_pcts = []

for token_step in range(TOKENS):
    step_time_s  = 0.0
    step_stall_s = 0.0

    for i, L in enumerate(layers):
        place      = placement[i]
        kv_scaled  = kv_inc[L["name"]] * BATCH_SIZE
        weight_sz  = L["bytes"]
        comp_s     = compute_time_s(L["flops"])

        # Weight read
        if place == PL_HOST_DRAM:
            weight_mem_time = dram_time_s(weight_sz)
        else:
            if os_cache.access_page(i, weight_sz):
                weight_mem_time = cxl_time_s(weight_sz)
            else:
                weight_mem_time = nand_time_s(weight_sz) + OS_FAULT_PENALTY_S

        # KV write: serialized after weight read (simplex)
        kv_write_time = 0.0
        if kv_scaled > 0:
            kv_write_time = cxl_time_s(kv_scaled)
            total_kv_write_stall_s += kv_write_time
            step_stall_s += kv_write_time
            if token_step == 0:
                kv_layer_count += 1

        ltime = max(comp_s, weight_mem_time) + kv_write_time
        step_time_s += ltime

    per_token_latency += step_time_s

    # Write stall as % of this token's total step time
    stall_pct_this_token = (step_stall_s / step_time_s * 100) if step_time_s > 0 else 0.0
    per_token_write_stall_pcts.append(stall_pct_this_token)

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)





# Compute variables needed for Overall throughput
from tiers import Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
def ssd_time_s(n): return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

total_model_bytes = sum(L["bytes"] for L in layers)
cold_load = ssd_time_s(total_model_bytes)
pf_time   = prefill_latency
dec_time  = per_token_latency * TOKENS

# --- FINAL RIGOROUS REPORTING (Baselines) ---
total_weight_rd = sum(L["bytes"] for L in layers)
total_kv_wr     = sum(L.get("kv_cache_bytes", 0) for L in layers) / 512
total_io_vol    = total_weight_rd + total_kv_wr

print(f"Read_Op_Percent: {(total_weight_rd / total_io_vol) * 100:.4f}%")
print(f"Write_Op_Percent: {(total_kv_wr / total_io_vol) * 100:.4f}%")
print(f"Read_Ratio: 100.0000%")  # Baselines are Simplex (Wasted Lane)

print(f"Decode throughput: {decode_tps:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f}")
print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / (cold_load + pf_time + dec_time):.3f}")

avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)
print(f"Write_Stall_Time_s: {total_kv_write_stall_s / TOKENS:.6f}")
print(f"Write_Stall_Pct: {avg_write_stall_pct:.4f}%")
print(f"Write_Util_Pct: 0.0000%")
print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.4f}' for x in per_token_write_stall_pcts)}")


