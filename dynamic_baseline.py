# Dynamic replacement baseline: instead of keeping a fixed reserve, the device
# DRAM is managed as a cache over the NAND with Belady's replacement rule,
# which the known decode order makes possible. Before each sub-layer runs, the
# next K device-class sub-layers are prefetched, evicting whatever is needed
# furthest in the future, but never a sub-layer needed sooner than the one being
# fetched. On CMM-H this is what the host's prefetch and eviction hints
# (Soltaniyeh et al., HotStorage'25) would implement.
#
# The unit is the sub-layer (attention or MLP), the granularity TierSched
# places at, so the cache can keep part of a block. GPU HBM and host DRAM are
# filled statically in model order with the embedding table last, as in the
# other baselines; everything else is device-class and goes through the cache,
# except an embedding table that did not fit, whose B looked-up rows per step
# are read from NAND.
#
#   DYN_PREFETCH_DEPTH  lookahead K (default 32)
#   DYN_KV_TIER         "cxl" or "host" (default "cxl")

import os

from tiers import kv_reserve_level, kv_step_split
from tiers import kv_growth_spill_time_s, kv_cascade, kv_tail
from tiers import (GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, GPU_HBM,
                   transfer_time_s)
from model_cfg import decomposed_build_layers, DEFAULT_MODEL_CFG, bytes_read, embedding_last
from pipeline import pipelined_time_s
from sim_cfg import (
    gpu_hbm_capacity_bytes,
    TOKENS, cpu_freq_hz, cpu_cores,
    flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, BATCH_SIZE
)

# KV reserved at the generation mean or its peak, whichever is faster.
KV_LEVEL = kv_reserve_level(__file__)

PREFETCH_DEPTH = int(os.environ.get("DYN_PREFETCH_DEPTH", "32"))
KV_TIER = os.environ.get("DYN_KV_TIER", "cxl")
# "belady": the host steers the cache with prefetch and eviction hints.
# "lru": CMM-H's default host-agnostic mode, where the device fills its cache on
# demand and evicts the least recently used page; the cache does not prefetch.
POLICY = os.environ.get("DYN_POLICY", "belady")
WARMUP_STEPS = 3

TIER_BW  = {"GPU HBM": GPU_HBM.bw_Bps, "Host DRAM": HOST_DRAM.bw_Bps,
            "CXL Device DRAM": CXL_DRAM.bw_Bps,
            "CXL Device NAND": CXL_SSD_NAND.bw_Bps}
TIER_LAT = {"GPU HBM": GPU_HBM.chunk_latency_s,
            "Host DRAM": HOST_DRAM.chunk_latency_s,
            "CXL Device DRAM": CXL_DRAM.chunk_latency_s,
            "CXL Device NAND": CXL_SSD_NAND.chunk_latency_s}

PL_GPU_HBM, PL_HOST_DRAM = "GPU HBM", "Host DRAM"
PL_CXL_DEV_DRAM, PL_CXL_DEV_NAND = "CXL Device DRAM", "CXL Device NAND"
PL_DEVICE_CLASS = "device-class"     # resident or not is decided by the cache

PREFILL_TOKENS = 512


def compute_time_s(flops):
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)


layers = decomposed_build_layers(DEFAULT_MODEL_CFG(), sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}

# KV is reserved at the generation mean on its tier, as in the other baselines,
# spilling to the other DRAM tier and then to NAND.
_kv_resident = int(sum(kv_inc.values()) * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS * KV_LEVEL))
_kv_order = ([("host", host_dram_capacity_bytes), ("cxl", cxl_dev_dram_capacity_bytes)]
             if KV_TIER == "host" else
             [("cxl", cxl_dev_dram_capacity_bytes), ("host", host_dram_capacity_bytes)])
_kv_split, _kv_took = kv_cascade(_kv_resident, _kv_order)
_kv_host, _kv_dev = _kv_took.get("host", 0.0), _kv_took.get("cxl", 0.0)

gpu_free = gpu_hbm_capacity_bytes
host_free = max(0, host_dram_capacity_bytes - _kv_host)
cache_cap = max(0, cxl_dev_dram_capacity_bytes - _kv_dev)

placement = [None] * len(layers)
for i in embedding_last(layers):
    L = layers[i]
    if L["bytes"] <= gpu_free:
        placement[i] = PL_GPU_HBM
        gpu_free -= L["bytes"]
    elif L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    elif L["kind"] == "Embedding":
        placement[i] = PL_CXL_DEV_NAND
    else:
        placement[i] = PL_DEVICE_CLASS

dev_idx = [i for i, p in enumerate(placement) if p == PL_DEVICE_CLASS]
n_dev = len(dev_idx)
pos = {i: k for k, i in enumerate(dev_idx)}


def belady_hits(steps):
    """Simulate the device cache over `steps` decode steps of the device-class
    sequence. Returns, per step, the sub-layers served from device DRAM. A
    sub-layer is a hit only if it stayed resident since its previous use; one
    fetched in the meantime, by prefetch or on demand, is a NAND read."""
    resident = {}
    fresh = set()            # fetched from NAND and not yet used
    last = {}                # position of each sub-layer's last use, for LRU
    window = PREFETCH_DEPTH if POLICY == "belady" else 0
    used = 0
    per_step = []
    for s in range(steps):
        hits = set()
        for k in range(n_dev):
            p = s * n_dev + k

            def next_use(i):
                return p + ((pos[i] - k) % n_dev)

            # Prefetch the window p..p+K, nearest first.
            for d in range(window + 1):
                x = dev_idx[(k + d) % n_dev]
                need = layers[x]["bytes"]
                if x in resident or need > cache_cap:
                    continue
                while used + need > cache_cap:
                    if POLICY == "lru":
                        victim = min(resident, key=lambda i: last.get(i, -1))
                    else:
                        victim = max(resident, key=next_use)
                        if next_use(victim) <= p + d:
                            break
                    used -= resident.pop(victim)
                    fresh.discard(victim)
                if used + need <= cache_cap:
                    resident[x] = need
                    used += need
                    fresh.add(x)
                    last[x] = p
            cur = dev_idx[k]
            if cur in resident and cur not in fresh:
                hits.add(cur)
            fresh.discard(cur)
            last[cur] = p
        per_step.append(hits)
    return per_step


_hits = belady_hits(WARMUP_STEPS + 1)[-1]    # steady state: repeats every step


# Prefill: compute-bound; device-class weights come from NAND unless resident.
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s = compute_time_s(L["flops"] * PREFILL_TOKENS * BATCH_SIZE)
    p = placement[i]
    tier = (PL_CXL_DEV_DRAM if i in _hits else PL_CXL_DEV_NAND) if p == PL_DEVICE_CLASS else p
    mem = transfer_time_s(bytes_read(L, PREFILL_TOKENS * BATCH_SIZE),
                          {PL_GPU_HBM: GPU_HBM, PL_HOST_DRAM: HOST_DRAM,
                           PL_CXL_DEV_DRAM: CXL_DRAM,
                           PL_CXL_DEV_NAND: CXL_SSD_NAND}[tier])
    prefill_latency += max(comp_s, mem)


# Decode
per_token_latency = 0.0
nand_bytes = sum(layers[i]["bytes"] for i in dev_idx if i not in _hits)
for token_step in range(TOKENS):
    units = []
    _kvf = kv_step_split(_kv_split, _kv_resident, sum(kv_inc.values()) * BATCH_SIZE
                         * (PREFILL_TOKENS + token_step))
    for i, L in enumerate(layers):
        p = placement[i]
        tier = (PL_CXL_DEV_DRAM if i in _hits else PL_CXL_DEV_NAND) if p == PL_DEVICE_CLASS else p
        _by = {tier: float(bytes_read(L, BATCH_SIZE))}
        kv_scaled = kv_inc[L["name"]] * BATCH_SIZE
        if kv_scaled > 0:
            kv_read = (PREFILL_TOKENS + token_step) * kv_scaled
            for _t, _f in ((PL_HOST_DRAM, _kvf.get("host", 0.0)),
                           (PL_CXL_DEV_DRAM, _kvf.get("cxl", 0.0)),
                           (PL_CXL_DEV_NAND, _kvf.get("nand", 0.0))):
                if _f > 0:
                    _by[_t] = _by.get(_t, 0.0) + kv_read * _f
        units.append((_by, compute_time_s(L["flops"] * BATCH_SIZE)))
    # Prefetched bytes live in the cache itself: eviction makes the room, so the
    # in-flight bound is the cache plus any host DRAM left free.
    step = pipelined_time_s(units, PREFETCH_DEPTH, TIER_BW, TIER_LAT,
                            inflight_budget=cache_cap + host_free)
    step += kv_growth_spill_time_s(
        sum(kv_inc.values()) * BATCH_SIZE * (PREFILL_TOKENS + token_step + 1),
        _kv_resident, kv_tail(_kv_split),
        headroom=({"gpu": gpu_free, "host": host_free}
                  .get(list(_kv_split)[-1], 0.0) if _kv_split else 0.0))
    per_token_latency += step

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)
print(f"Dynamic cache ({POLICY}): {n_dev} device-class sub-layers, {len(_hits)} hits per step, "
      f"{nand_bytes / GiB:.2f} GB from NAND per step, K={PREFETCH_DEPTH}, KV {KV_TIER}")
print(f"Decode throughput: {decode_tps:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.6f}")
