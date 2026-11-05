# ITME baseline (Jang et al., "ITME: Inference Tiered Memory Expansion with
# Disaggregated CXL-Hybrid Memories", arXiv 2606.12556).
#
# ITME keeps the weights in a CXL-hybrid memory (a DRAM cache in front of NAND,
# the organization of CMM-H) and streams them layer by layer to the compute
# engine through a circular buffer in the device DRAM and a staging buffer in
# host DRAM. GPU and host memory are left to the working KV cache and the
# weights that are not streamed. From the paper:
#
#   - The streamed weights are the FFNs. Fig. 2b feeds only FFN(L) from the
#     device, and the 1344 MB transfer unit (Sec. 2.3) and the 105 GB and
#     10.5 GB offloaded footprints (Sec. 3.1, 6.4) are exactly the FP16 FFN
#     bytes of one Llama-3.1 70B layer and of all of Llama-3.1 70B and 8B. The
#     other weights stay with the compute engine: in GPU memory, or in host
#     DRAM on the CPU-only platform. Any that do not fit are streamed too.
#   - The device DRAM holds a circular buffer of BUFFER_LAYERS slots, one FFN
#     layer each, and a slot is refilled with the next layer as soon as its
#     layer has been read (Sec. 4.3). Fig. 7 draws three slots, and Sec. 6.4
#     calls a 4 GB buffer (3 x 1344 MB) "a three-depth pipeline".
#   - Host DRAM holds pinned store and load staging buffers, sized as a
#     fraction of host memory (Sec. 4.1); the evaluation gives them 30 GB of a
#     256 GB host (Sec. 5, Figs. 10-11). The rest of host DRAM holds spillover
#     KV, not weights (Fig. 1b).
#   - The KV cache fills the engine's memory after its weights (ITME is built
#     on vLLM, Sec. 4), then host DRAM, then the CXL-hybrid memory (Table 1,
#     Sec. 2.2), and what none of them holds is read from NAND.
#   - The device DRAM is a hardware cache with LRU replacement (Sec. 3.1). The
#     streamed weights stay in it only if all of them fit, as the 8B model's
#     do (Sec. 6.4); a larger cyclic stream evicts each layer before its next
#     use, so every step reads all of them from NAND.
#
# On CMM-H the device is local on the CXL link rather than behind RDMA; NAND
# reads pass through its DRAM, which pipeline.py charges.
#
#   ITME_BUFFER_LAYERS      circular-buffer slots in device DRAM (default 3)
#   ITME_HOST_STAGING_FRAC  share of host DRAM held by the staging buffers
#                           (default 30/256)

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
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes,
    cxl_ssd_capacity_bytes, BATCH_SIZE
)

# KV reserved at the generation mean or its peak, whichever is faster.
KV_LEVEL = kv_reserve_level(__file__)

BUFFER_LAYERS = int(os.environ.get("ITME_BUFFER_LAYERS", "3"))
HOST_STAGING_FRAC = float(os.environ.get("ITME_HOST_STAGING_FRAC", str(30 / 256)))
assert BUFFER_LAYERS >= 1 and 0.0 <= HOST_STAGING_FRAC <= 1.0

PL_GPU_HBM, PL_HOST_DRAM = "GPU HBM", "Host DRAM"
PL_CXL_DEV_DRAM, PL_CXL_DEV_NAND = "CXL Device DRAM", "CXL Device NAND"
TIER = {PL_GPU_HBM: GPU_HBM, PL_HOST_DRAM: HOST_DRAM,
        PL_CXL_DEV_DRAM: CXL_DRAM, PL_CXL_DEV_NAND: CXL_SSD_NAND}
TIER_BW = {k: t.bw_Bps for k, t in TIER.items()}
TIER_LAT = {k: t.chunk_latency_s for k, t in TIER.items()}
KV_PL = {"gpu": PL_GPU_HBM, "host": PL_HOST_DRAM, "cxl": PL_CXL_DEV_DRAM,
         "nand": PL_CXL_DEV_NAND}

PREFILL_TOKENS = 512


def compute_time_s(flops):
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)


layers = decomposed_build_layers(DEFAULT_MODEL_CFG(), sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}

# Staging, reserved before anything is placed. A ring slot holds one FFN layer.
slot = max(L["bytes"] for L in layers if L["kind"] == "MLP")
BUFFER_LAYERS = min(BUFFER_LAYERS, max(1, int(cxl_dev_dram_capacity_bytes // slot)))
ring_bytes = BUFFER_LAYERS * slot
host_staging = HOST_STAGING_FRAC * host_dram_capacity_bytes

# A slot is refilled once its layer has been read, so with N slots the read of
# FFN L+N-1 is issued when FFN L-1 is done (Fig. 7: L+2 while L computes, for
# N = 3). Units alternate attention and FFN, which puts it 2N-1 units ahead.
PREFETCH_DEPTH = 2 * BUFFER_LAYERS - 1

# Reads in flight wait in the ring and the host staging buffers, and nowhere
# else: no resident weight or KV is placed in either.
_ITME_STAGING = ring_bytes + host_staging

# Every weight but the FFNs stays with the compute engine, in model order with
# the embedding table last; the rest is streamed from the device.
gpu = gpu_hbm_capacity_bytes > 0
PL_ENGINE = PL_GPU_HBM if gpu else PL_HOST_DRAM
engine_free = gpu_hbm_capacity_bytes if gpu else host_dram_capacity_bytes - host_staging
placement = [None] * len(layers)
for i in embedding_last(layers):
    L = layers[i]
    if L["kind"] != "MLP" and L["bytes"] <= engine_free:
        placement[i] = PL_ENGINE
        engine_free -= L["bytes"]
streamed = [i for i, p in enumerate(placement) if p is None]

# KV is reserved at the generation mean on what the engine's weights leave,
# then on host DRAM, then on the device DRAM outside the ring.
_kv_resident = int(sum(kv_inc.values()) * BATCH_SIZE
                   * (PREFILL_TOKENS + TOKENS * KV_LEVEL))
_kv_split, _kv_took = kv_cascade(_kv_resident, [
    ("gpu", engine_free if gpu else 0.0),
    ("host", host_dram_capacity_bytes - host_staging if gpu else engine_free),
    ("cxl", cxl_dev_dram_capacity_bytes - ring_bytes)])

# The streamed weights are served from the device DRAM cache only if all of
# them fit beside the ring and the KV.
streamed_bytes = sum(layers[i]["bytes"] for i in streamed)
cache_room = cxl_dev_dram_capacity_bytes - ring_bytes - _kv_took.get("cxl", 0.0)
cached = streamed_bytes <= cache_room
for i in streamed:
    placement[i] = PL_CXL_DEV_DRAM if cached else PL_CXL_DEV_NAND

def _resident(pl):
    return sum(L["bytes"] for L, p in zip(layers, placement) if p == pl)


# Room the KV cache can grow into before it displaces weights: what its last
# tier still has free.
_free = {"gpu": (engine_free if gpu else 0.0) - _kv_took.get("gpu", 0.0),
         "host": (host_dram_capacity_bytes - host_staging - _resident(PL_HOST_DRAM)
                  if gpu else engine_free) - _kv_took.get("host", 0.0),
         "cxl": cache_room - (streamed_bytes if cached else 0.0)}
_kv_headroom = max(0.0, _free.get(list(_kv_split)[-1], 0.0)) if _kv_split else 0.0

# No tier holds more than it has.
_used = {
    PL_GPU_HBM: (_resident(PL_GPU_HBM) + _kv_took.get("gpu", 0.0), gpu_hbm_capacity_bytes),
    PL_HOST_DRAM: (host_staging + _resident(PL_HOST_DRAM) + _kv_took.get("host", 0.0),
                   host_dram_capacity_bytes),
    PL_CXL_DEV_DRAM: (ring_bytes + _resident(PL_CXL_DEV_DRAM) + _kv_took.get("cxl", 0.0),
                      cxl_dev_dram_capacity_bytes),
    PL_CXL_DEV_NAND: (streamed_bytes + _kv_took.get("nand", 0.0), cxl_ssd_capacity_bytes),
}
for _t, (_u, _cap) in _used.items():
    assert _u <= _cap + 1, f"{_t} holds {_u / GiB:.2f} GB of {_cap / GiB:.2f} GB"


# Prefill: compute-bound; each layer is read once from where it is placed.
prefill_latency = 0.0
for i, L in enumerate(layers):
    comp_s = compute_time_s(L["flops"] * PREFILL_TOKENS * BATCH_SIZE)
    mem = transfer_time_s(bytes_read(L, PREFILL_TOKENS * BATCH_SIZE), TIER[placement[i]])
    prefill_latency += max(comp_s, mem)


# Decode
per_token_latency = 0.0
for token_step in range(TOKENS):
    units = []
    for i, L in enumerate(layers):
        _by = {placement[i]: float(bytes_read(L, BATCH_SIZE))}
        if kv_inc[L["name"]] > 0:
            kv_read = (PREFILL_TOKENS + token_step) * kv_inc[L["name"]] * BATCH_SIZE
            for _k, _f in kv_step_split(_kv_split, _kv_resident, sum(kv_inc.values())
                                        * BATCH_SIZE * (PREFILL_TOKENS + token_step)).items():
                _by[KV_PL[_k]] = _by.get(KV_PL[_k], 0.0) + kv_read * _f
        units.append((_by, compute_time_s(L["flops"] * BATCH_SIZE)))
    step = pipelined_time_s(units, PREFETCH_DEPTH, TIER_BW, TIER_LAT,
                            inflight_budget=_ITME_STAGING)
    step += kv_growth_spill_time_s(
        sum(kv_inc.values()) * BATCH_SIZE * (PREFILL_TOKENS + token_step + 1),
        _kv_resident, kv_tail(_kv_split), headroom=_kv_headroom)
    per_token_latency += step

decode_tps = BATCH_SIZE / (per_token_latency / TOKENS)
print(f"ITME: ring {BUFFER_LAYERS} x {slot / GiB:.2f} GB, host staging "
      f"{host_staging / GiB:.2f} GB, depth {PREFETCH_DEPTH} units; "
      f"{_resident(PL_ENGINE) / GiB:.2f} GB resident in {PL_ENGINE}, "
      f"{streamed_bytes / GiB:.2f} GB streamed from "
      f"{'device DRAM' if cached else 'NAND'}; KV "
      + ", ".join(f"{k} {v / GiB:.2f} GB" for k, v in _kv_took.items()))
print("Capacity: " + ", ".join(f"{t} {u / GiB:.2f}/{c / GiB:.2f} GB"
                               for t, (u, c) in _used.items()))
print(f"Decode throughput: {decode_tps:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.6f}")
