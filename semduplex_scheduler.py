# SemSched simulator: hybrid-CXL LLM inference with
#   - sub-layer placement (attention and MLP placed separately, order set by
#     PLACEMENT_ORDER) and a per-configuration search over the staging reserve
#     and the KV-cache tier,
#   - optional Phase-1 NAND staging into device DRAM (STAGE_RESERVE_FRAC),
#   - K=32 lookahead, sparsity-ranked prefetching on a 16-thread I/O pool,
#   - a two-queue full-duplex CXL link (cxl_link.py): independent Rx and Tx lanes
#     (27 GB/s each, 505 ns per transaction). A write that fits under the read
#     window costs nothing; one that overruns it is charged the slip as stall.
#     Uwrite is measured from the queue (tx_busy_total / t_end).
#
# Prefill throughput is per sequence (PREFILL_TOKENS / pf_time_val), the
# convention of the baselines (FlexGen, LIA, LLMFlash).
#
# Decode:
#   - growing KV: step t reads (PREFILL_TOKENS + t) x per-token KV bytes x
#     BATCH_SIZE from the KV tier;
#   - below ADAPTIVE_BATCH_THRESHOLD, the per-layer pass scales MLP transfer
#     bytes by the active fraction 1 - (1 - sparsity)^B (LLMFlash sliding window).

import math
import pandas as pd
import sys
import traceback
from collections import OrderedDict, deque
from enum import Enum

from cxl_link import CXLLink   # two-queue full-duplex bus model

from tiers import kv_growth_spill_time_s
from pipeline import pipelined_time_s
from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, GPU_HBM, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, IO_CHUNK_BYTES
)
from model_cfg import build_layers, BYTES_PER_PARAM, DEFAULT_MODEL_CFG
from sim_cfg import (
    gpu_hbm_capacity_bytes,
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes,
    BATCH_SIZE
)

GiB = 1024**3
PL_GPU_HBM      = "GPU HBM"
PL_HOST_DRAM    = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

IO_THREAD_POOL_SIZE   = 16
PREFETCH_QUEUE_DEPTH  = 32
TRAFFIC_WINDOW_SIZE   = 16
PREFILL_CHUNK_SIZE    = 64
ENABLE_PREFILL_WARMUP = True
PREFILL_TOKENS        = 512
PREFILL_FLOP_MULTIPLIER = 15.0

DUPLEX_PENALTY = 1.15

# Tier for the KV cache: "cxl" | "host" | "gpu". Its capacity is reserved before
# placement, KV reads and writes are timed at its bandwidth, and overflow spills
# to the next slower tier. The decode setup searches the KV tier over _kv_opts
# together with the staging reserve, so this value only seeds that search.
KV_TIER = "cxl"          # "cxl" | "host" | "gpu"

# Order in which sub-layers claim fast memory: "size-desc" (default) |
# "semantic" | "sequential" | "random". "semantic" is the type and sparsity
# priority order in fill(); size-desc beats it in the ablation (Sec. V-D).
import os as _os
PLACEMENT_ORDER = _os.environ.get("SEMSCHED_PLACEMENT_ORDER", "size-desc")

# Fraction of the staging reserve that Phase-1 warmup may pin for the session;
# the rest is the prefetcher's in-flight budget. Default 0: staging loses this
# split in every configuration (Sec. V-D).
STAGE_RESERVE_FRAC = float(_os.environ.get("SEMSCHED_STAGE_FRAC", "0.0"))

# Bandwidth and per-chunk latency of each placement tier, keyed by the label
# `frac` uses. The pipeline engine needs these to run tiers concurrently.
TIER_BW  = {PL_GPU_HBM:      GPU_HBM.bw_Bps,
            PL_HOST_DRAM:    HOST_DRAM.bw_Bps,
            PL_CXL_DEV_DRAM: CXL_DRAM.bw_Bps,
            PL_CXL_DEV_NAND: CXL_SSD_NAND.bw_Bps}
TIER_LAT = {PL_GPU_HBM:      GPU_HBM.chunk_latency_s,
            PL_HOST_DRAM:    HOST_DRAM.chunk_latency_s,
            PL_CXL_DEV_DRAM: CXL_DRAM.chunk_latency_s,
            PL_CXL_DEV_NAND: CXL_SSD_NAND.chunk_latency_s}

# Below this batch size, MLP transfers are scaled by the LLMFlash active
# fraction 1 - (1 - p)^B. At larger B the fraction is close to 1 and full layers
# are transferred. The B=4 crossover is observed in the LLMFlash baseline runs.
ADAPTIVE_BATCH_THRESHOLD = 4

OVERLAP_ATTENTION = 0.35
OVERLAP_MLP       = 0.65
OVERLAP_NORM      = 0.45
OVERLAP_OTHER     = 0.45

# Overlap factor for a streamed sub-layer.
SEMANTIC_STREAM_OVERLAP = 0.85


def decomposed_build_layers(sequence_length):
    standard_layers = build_layers(DEFAULT_MODEL_CFG, sequence_length=sequence_length)
    new_layers = []
    for L in standard_layers:
        if L["kind"] == "DecoderBlock":
            aname  = f"{L['name']}_attn"
            aflops = L["flops"] // 3
            abytes = int(L["bytes"] * 0.33)
            new_layers.append({
                "name":           aname,
                "kind":           "Attention",
                "bytes":          abytes,
                "flops":          aflops,
                "head_dim":       L.get("head_dim", 128),
                "kv_heads":       L.get("kv_heads", 8),
                "kv_cache_bytes": L.get("kv_cache_bytes", 0),
                "sparsity":       0.05,  # attention is structurally dense
            })
            mname  = f"{L['name']}_mlp"
            mflops = L["flops"] - aflops
            mbytes = L["bytes"] - abytes
            new_layers.append({
                "name":           mname,
                "kind":           "MLP",
                "bytes":          mbytes,
                "flops":          mflops,
                "kv_cache_bytes": 0,
                "sparsity":       L.get("sparsity", 0.54),  # inherited from the block
            })
        else:
            new_layers.append(L)
    return new_layers


class LayerType(Enum):
    ATTENTION = "attention"
    MLP       = "mlp"
    NORM      = "norm"
    EMBEDDING = "embedding"
    OUTPUT    = "output"


def classify_layer_type(layer_dict):
    name = layer_dict["name"].lower()
    if "attn" in name or "attention" in name: return LayerType.ATTENTION
    elif "mlp" in name or "ffn" in name:      return LayerType.MLP
    elif "norm" in name:                       return LayerType.NORM
    elif "embed" in name:                      return LayerType.EMBEDDING
    elif "lm_head" in name:                    return LayerType.OUTPUT
    return LayerType.NORM


def compute_layer_sparsity(layer_idx, total_decoder_blocks, layer_type):
    if layer_type not in [LayerType.ATTENTION, LayerType.MLP]: return 0.0
    norm_pos = layer_idx / max(1, total_decoder_blocks - 1)
    if layer_type == LayerType.ATTENTION:
        return 0.05
    elif layer_type == LayerType.MLP:
        if norm_pos < 0.15:   return 0.13 + (0.50 - 0.13) * (norm_pos / 0.15)
        elif norm_pos < 0.75: return 0.50 + (0.90 - 0.50) * ((norm_pos - 0.15) / 0.6)
        else:                 return 0.90 + (0.956 - 0.90) * ((norm_pos - 0.75) / 0.25)
    return 0.5


def get_overlap_factor(layer_type, is_streaming=False):
    if is_streaming: return SEMANTIC_STREAM_OVERLAP
    return {
        LayerType.ATTENTION: OVERLAP_ATTENTION,
        LayerType.MLP:       OVERLAP_MLP,
        LayerType.NORM:      OVERLAP_NORM,
    }.get(layer_type, OVERLAP_OTHER)


class DuplexTrafficMonitor:
    def __init__(self, window_size=10):
        self.read_history  = deque(maxlen=window_size)
        self.write_history = deque(maxlen=window_size)
        self.total_read_bytes  = 0
        self.total_write_bytes = 0
        self.total_write_time_s  = 0.0
        self.total_decode_time_s = 0.0
        self.write_stall_time_s = 0.0
        self.write_stall_count  = 0


    def record_read(self, b):
        self.read_history.append(b)
        self.total_read_bytes += b

    def record_write(self, b):
        self.write_history.append(b)
        self.total_write_bytes += b
        self.total_write_time_s += transfer_time_s(b, CXL_DRAM)

    def record_layer_time(self, t):
        self.total_decode_time_s += t

    def get_read_ratio(self):
        r = sum(self.read_history)
        w = sum(self.write_history)
        return r / (r + w) if (r + w) > 0 else 0.5

    def needs_read_injection(self):
        return self.get_read_ratio() < 0.45

    def needs_write_injection(self):
        return self.get_read_ratio() > 0.6

    # Write time as a percentage of decode time.
    def get_write_util_pct(self):
        if self.total_decode_time_s <= 0:
            return 0.0
        return (self.total_write_time_s / self.total_decode_time_s) * 100




class AttentionGuidedCache:
    def __init__(self, cap_bytes):
        self.cap              = cap_bytes
        self.used             = 0
        self.cache            = OrderedDict()
        self.attention_scores = {}
        self.pinned           = set()
        self.session_pinned   = set()

    def set_attention_score(self, lid, score):
        self.attention_scores[lid] = score
        if score > 0.8: self.pinned.add(lid)
        else:           self.pinned.discard(lid)

    def pin_for_session(self, lid): self.session_pinned.add(lid)

    def _evict(self, need):
        if need <= 0: return
        cands = [(lid, sz, self.attention_scores.get(lid, 0.5))
                 for lid, sz in self.cache.items()
                 if lid not in self.pinned and lid not in self.session_pinned]
        cands.sort(key=lambda x: x[2])
        freed = 0
        for lid, sz, score in cands:
            self.used -= self.cache.pop(lid)
            freed += sz
            if freed >= need: break

    def add(self, lid, sz):
        if sz > self.cap: return False
        need = max(0, (self.used + sz) - self.cap)
        if need > 0: self._evict(need)
        if (self.used + sz) <= self.cap:
            self.used += sz
            self.cache[lid] = sz
            return True
        return False

    def contains(self, lid, sz): return self.cache.get(lid, 0) >= sz


class DuplexScheduler:
    def __init__(self, io_pool_size=12):
        self.io_pool_size          = io_pool_size
        self.read_threads          = io_pool_size // 2
        self.write_threads         = io_pool_size // 2
        self.pending_kv_writebacks = 0

    def adjust_thread_allocation(self, tmon):
        ratio = tmon.get_read_ratio()
        if ratio > 0.6:    self.read_threads, self.write_threads = 1, self.io_pool_size - 1
        elif ratio < 0.45: self.read_threads, self.write_threads = self.io_pool_size - 1, 1
        else:              self.read_threads, self.write_threads = self.io_pool_size // 2, self.io_pool_size // 2

    def should_activate_duplex(self, sparsity):
        from model_cfg import QUANT
        if QUANT not in ["int4", "int8"]: return False
        return sparsity > 0.20

    def schedule_complementary_ops(self, layer_type, has_kv_cache, tmon, sparsity):
        if not self.should_activate_duplex(sparsity): return False, False
        if has_kv_cache and layer_type == LayerType.ATTENTION:
            return (True, True) if tmon.needs_read_injection() else (False, True)
        return (True, True) if tmon.needs_write_injection() and self.pending_kv_writebacks > 0 else (True, False)


class IOThread:
    def __init__(self, i):
        self.id           = i
        self.busy_until   = 0.0
        self.current_task = None


def tier_of(p, staged=True):
    """Map a placement label to its Tier.

    A NAND-resident layer is read at CXL DRAM speed only if Phase 1 staged it
    into the device cache; otherwise it is read from NAND.
    """
    if p == PL_GPU_HBM:   return GPU_HBM
    if p == PL_HOST_DRAM: return HOST_DRAM
    if p == PL_CXL_DEV_NAND and not staged:
        return CXL_SSD_NAND
    return CXL_DRAM


def split_time_s(nbytes, fr, staged=True):
    """Time to move `nbytes` of a layer split across tiers by the map `fr`.

    Placement fills each tier exactly, so a sub-layer may straddle a boundary.
    Each share is timed at the tier that holds it; for a single-tier layer this
    reduces to transfer_time_s(nbytes, tier_of(...)).
    """
    return sum(transfer_time_s(nbytes * f, tier_of(t, staged))
               for t, f in fr.items() if f > 0)


def semantic_aware_placement(layers, host_cap, cxl_cap, total_decoder_blocks,
                             gpu_cap=0):
    place    = [None] * len(layers)
    ltypes   = [classify_layer_type(L) for L in layers]
    sparsity = {}
    kv_inc   = {}

    for i, L in enumerate(layers):
        dec_idx = 0
        if "decoder_" in L["name"]:
            try: dec_idx = int(L["name"].split("_")[1])
            except: pass
        sparsity[i] = compute_layer_sparsity(dec_idx, total_decoder_blocks, ltypes[i])
        if ltypes[i] == LayerType.ATTENTION:
            kv_inc[L["name"]] = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * BYTES_PER_PARAM
        else:
            kv_inc[L["name"]] = 0

    def size_of(i, L):
        # Attention layers must also hold their growing KV increment.
        return L["bytes"] + (kv_inc[L["name"]] if ltypes[i] == LayerType.ATTENTION else 0)

    # Bytes of each layer not yet assigned to a tier, and the per-tier fraction
    # of each layer. Every tier is filled exactly: a layer that straddles a
    # capacity boundary is split across it.
    rem  = [float(L["bytes"]) for L in layers]
    frac = [dict() for _ in layers]

    def take(i, tier_label, free, extra=0.0):
        """Move as much of layer i into `tier_label` as fits. Returns new free."""
        if free <= 0 or rem[i] <= 0:
            return free
        # The KV increment is indivisible, so it is charged only when the whole
        # layer fits.
        want = rem[i] + (extra if rem[i] == layers[i]["bytes"] else 0.0)
        got  = min(want, free)
        if got <= 0:
            return free
        moved = min(got, rem[i])
        frac[i][tier_label] = frac[i].get(tier_label, 0.0) + moved / layers[i]["bytes"]
        rem[i] -= moved
        return free - got

    def fill(tier_label, cap):
        """Fill one tier in PLACEMENT_ORDER; return the unused bytes.

        The "semantic" order: output head (touched every step), attention with
        its KV (latency-critical), embeddings, MLP with sparsity > 0.60 by
        descending sparsity, then whatever dense MLP still fits.
        """
        free = cap
        if free <= 0:
            return free
        if PLACEMENT_ORDER != "semantic":
            # Same tiers, capacities and bytes; only the order in which
            # sub-layers claim fast memory differs.
            if PLACEMENT_ORDER == "sequential":
                order = list(range(len(layers)))
            elif PLACEMENT_ORDER == "size-desc":
                order = sorted(range(len(layers)),
                               key=lambda i: layers[i]["bytes"], reverse=True)
            elif PLACEMENT_ORDER == "random":
                import random as _r
                order = list(range(len(layers)))
                _r.Random(20260814).shuffle(order)
            else:
                raise SystemExit(f"unknown SEMSCHED_PLACEMENT_ORDER: {PLACEMENT_ORDER}")
            for i in order:
                extra = (size_of(i, layers[i]) - layers[i]["bytes"]
                         if ltypes[i] == LayerType.ATTENTION else 0.0)
                free = take(i, tier_label, free, extra)
            return free
        for i, L in enumerate(layers):                       # 0: output head
            if ltypes[i] == LayerType.OUTPUT:
                free = take(i, tier_label, free)
        for i, L in enumerate(layers):                       # 1: attention + KV
            if ltypes[i] == LayerType.ATTENTION:
                free = take(i, tier_label, free, size_of(i, L) - L["bytes"])
        for i, L in enumerate(layers):                       # 2: embeddings
            if ltypes[i] == LayerType.EMBEDDING:
                free = take(i, tier_label, free)
        cands = sorted([(i, L, sparsity[i]) for i, L in enumerate(layers)
                        if ltypes[i] == LayerType.MLP],
                       key=lambda x: x[2], reverse=True)
        for i, L, sp in cands:                               # 3: sparse MLP
            if sp > 0.60:
                free = take(i, tier_label, free)
        for i, L in enumerate(layers):                        # 3.5: dense MLP
            if ltypes[i] == LayerType.MLP:
                free = take(i, tier_label, free)
        return free

    # Fill tiers fastest first; whatever is left goes to NAND. With gpu_cap = 0
    # this is the three-tier placement.
    fill(PL_GPU_HBM, gpu_cap)
    fill(PL_HOST_DRAM, host_cap)
    fill(PL_CXL_DEV_DRAM, cxl_cap)                            # 4: CXL DRAM

    for i, L in enumerate(layers):                            # 5: NAND
        if rem[i] > 0:
            frac[i][PL_CXL_DEV_NAND] = (frac[i].get(PL_CXL_DEV_NAND, 0.0)
                                        + rem[i] / L["bytes"])
            rem[i] = 0.0

    # Spread the NAND residue through execution order (Sec. IV-C). A K-deep
    # lookahead can only overlap a transfer with work that runs before it, and
    # the fill order tends to leave the residue in one contiguous band.
    # Sub-layers in a (type, size) group are interchangeable, so swapping their
    # splits changes positions but not bytes.
    _groups = {}
    for i, L in enumerate(layers):
        _groups.setdefault((ltypes[i], L["bytes"]), []).append(i)
    for _members in _groups.values():
        if len(_members) < 3:
            continue
        _on_nand = [i for i in _members if frac[i].get(PL_CXL_DEV_NAND, 0) > 0]
        if not (0 < len(_on_nand) < len(_members)):
            continue        # all or nothing on NAND: no arrangement to choose
        # Even stride across the group, so each NAND read has fast-tier work
        # before it to hide behind.
        _step = len(_members) / len(_on_nand)
        _target = [_members[min(len(_members) - 1, int(k * _step + _step / 2))]
                   for k in range(len(_on_nand))]
        _target = sorted(set(_target))
        if len(_target) != len(_on_nand) or set(_target) == set(_on_nand):
            continue
        _donor = [i for i in _on_nand if i not in _target]
        _recv  = [i for i in _target if i not in _on_nand]
        for _d, _r in zip(_donor, _recv):
            frac[_d], frac[_r] = frac[_r], frac[_d]

    # `place` names the tier holding the largest share. It drives the per-layer
    # control flow (staging, prefetch, duplex lane selection), which cannot be
    # fractional; `frac` carries the split that timing uses.
    place = [max(f, key=f.get) for f in frac]

    return place, ltypes, kv_inc, sparsity, frac

def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    return flops / (cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency)


def combine_sublayer_stats(rows):
    combined = []
    skip     = False
    for i in range(len(rows)):
        if skip:
            skip = False
            continue
        row = rows[i]
        if row["Name"].endswith("_attn") and i + 1 < len(rows):
            nxt = rows[i + 1]
            if nxt["Name"] == row["Name"].replace("_attn", "") + "_mlp":
                combined.append({
                    "Layer":        len(combined) + 1,
                    "Name":         row["Name"].replace("_attn", ""),
                    "Type":         "decoder_block",
                    "Sparsity":     f"Attn:{row['Sparsity']}/MLP:{nxt['Sparsity']}",
                    "Placement":    f"Attn:{row['Placement']}/MLP:{nxt['Placement']}",
                    "Served_From":  f"Attn:{row['Served_From']}/MLP:{nxt['Served_From']}",
                    "Layer_Time_s": row["Layer_Time_s"] + nxt["Layer_Time_s"],
                    "Read_Ratio":   row["Read_Ratio"]
                })
                skip = True
                continue
        combined.append(row)
    return combined


def run_prefill_chunked(layers, place, ltypes, sparsity, inc,
                        cache, tmon, sched, threads, seq_len, frac):
    chunks = math.ceil(seq_len / PREFILL_CHUNK_SIZE)
    lat, nand_link_free_at = 0.0, 0.0
    stats = {"bytes_prefetched": 0, "warmup_time": 0}

    # Phase 1: stage NAND-resident layers into device DRAM, up to the cache
    # capacity (none if ENABLE_PREFILL_WARMUP is False). Staging uses the NAND->DRAM DMA path, independent of compute, so
    # it overlaps prefill: a layer waits only if its staging has not finished
    # when execution reaches it. stage_finish[i] is that completion time on a
    # single bandwidth-limited NAND channel.
    stage_finish = {}
    t_stage      = 0.0
    _stage_budget = cache.cap if ENABLE_PREFILL_WARMUP else 0
    for i, L in enumerate(layers):
        if place[i] == PL_CXL_DEV_NAND:
            bw_term  = L["bytes"] / CXL_SSD_NAND.bw_Bps
            # Per-chunk latency is serial, not divided across the I/O pool,
            # since no baseline models concurrent staging I/O either.
            lat_term = (math.ceil(L["bytes"] / IO_CHUNK_BYTES)
                        * CXL_SSD_NAND.chunk_latency_s)
            if L["bytes"] > _stage_budget:
                continue          # does not fit: stays NAND-resident for decode
            _stage_budget -= L["bytes"]
            t_stage += bw_term + lat_term
            stage_finish[i] = t_stage
            cache.add(i, L["bytes"])
            cache.pin_for_session(i)
            stats["bytes_prefetched"] += L["bytes"]
    stats["warmup_time"] = t_stage

    for i, L in enumerate(layers):
        sz        = L["bytes"]
        # Prefill runs seq_len tokens for each of B sequences: FLOPs =
        # 2*P*seq_len*B. No sparsity discount, since prefill fires all neurons.
        eff_flops = L["flops"] * seq_len * BATCH_SIZE
        mem = split_time_s(sz, frac[i], i in cache.session_pinned)

        comp_chunk = compute_time_s(eff_flops, cpu_cores) / chunks
        # Overlap can hide transfer time but not compute: every FLOP still runs.
        full_comp  = comp_chunk * chunks
        # Transfer of chunk k+1 overlaps compute of chunk k, so the cost is the
        # slower stream plus one chunk of the faster one to fill the pipe.
        pipe_time  = (max(mem, full_comp) + min(mem, full_comp) / chunks
                      if chunks > 1 else max(mem, comp_chunk))

        # Wait for this layer's asynchronous staging only if it has not finished.
        start = max(lat, stage_finish.get(i, 0.0))
        # No duplex penalty in prefill; write contention is modeled only in
        # decode, by the CXLLink queue.
        lat = start + pipe_time

        tmon.record_read(sz)
    return lat, stats


def ssd_cold_time_s(n):
    return transfer_time_s(
        n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# Per-token latency and write-stall sweep over all simulators (not run by default).
def track_write_stalls():
    print("\n" + "=" * 70)
    print("TRACKING PER-TOKEN LATENCY & STALLS FOR ALL SIMULATORS")
    print("=" * 70)
    
    results = []
    
    # The tight memory config (16 GB host + 32 GB CXL) at B=128 exposes NAND
    # stalls at every precision.
    target_batch = 128
    h_gb, c_gb = 16, 32 
    quants_to_test = ["fp32", "fp16", "int8", "int4"]
    
    update_memory_config(h_gb, c_gb)
    update_batch_config(target_batch)
    
    for quant in quants_to_test:
        print(f"\n--- Model: Qwen 72B | Quant: {quant.upper()} | Batch: {target_batch} ---")
        update_model_config("Qwen2_5_72BCfg", quant)
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(
                    ["python", sim], capture_output=True, text=True, timeout=900)
                
                mets = parse_metrics(ret.stdout, sim)
                tps = mets["TPS"]
                
                # Per-token latency (compute + stall), from TPS = batch / latency.
                latency_s = target_batch / tps if tps > 0 else 0.0

                # Prefer the simulator's own single-token latency if it prints one.
                exact_lat_match = re.search(r"Single-token decode latency:\s*([\d\.]+)s", ret.stdout)
                if exact_lat_match:
                    latency_s = float(exact_lat_match.group(1))
                    
                results.append({
                    "Simulator": sim_name,
                    "Model": "72B",
                    "Quant": quant,
                    "MemConfig": f"{h_gb}H+{c_gb}C",
                    "BatchSize": target_batch,
                    "TPS": tps,
                    "Per_Token_Latency_s": latency_s,
                    "Write_Util_Pct": mets.get("Write_Util_Pct", 0.0),
                    "Bus_Mode": mets.get("Bus_Mode", "Simplex")
                })
                print(f"  {sim_name:<12}: TPS={tps:>8.4f} | Per-Token Latency={latency_s:>8.4f}s")
                
            except Exception as e:
                print(f"  {sim_name:<12}: ERROR — {e}")

    # Restore default configurations after the sweep
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")
    update_batch_config(1)

    # Save to a separate CSV
    df = pd.DataFrame(results)
    csv_filename = "token_stalls_latency.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n>>> Successfully saved per-token stall data to {csv_filename}")




def run_semantic_duplex_simulation():
    try:
        seq_len    = PREFILL_TOKENS
        layers     = decomposed_build_layers(sequence_length=seq_len)
        num_blocks = sum(1 for L in layers
                         if "decoder_" in L["name"] and "_attn" in L["name"])

        # KV capacity is reserved from its tier before placement. KV grows, and
        # at step t the weights can use C - kv(P + t), so placement (a one-shot
        # decision) is given the time-averaged reservation kv(P + T/2).
        _kv_per_tok = sum(2 * L.get("kv_heads", 8) * L.get("head_dim", 128)
                          * BYTES_PER_PARAM
                          for L in layers
                          if classify_layer_type(L) == LayerType.ATTENTION) * BATCH_SIZE
        _kv_mean = int(_kv_per_tok * (PREFILL_TOKENS + TOKENS / 2.0))

        _caps = {"gpu":  gpu_hbm_capacity_bytes,
                 "host": host_dram_capacity_bytes,
                 "cxl":  cxl_dev_dram_capacity_bytes}
        # Unreduced capacities: every _kv_place call starts from these, not
        # from a dict an earlier call already reduced.
        _caps0 = dict(_caps)
        _kv_tiers = {"gpu": GPU_HBM, "host": HOST_DRAM, "cxl": CXL_DRAM}

        def _kv_place(tier):
            """Charge the cache to `tier`, spilling down when it does not fit.

            Returns (split, caps): the fraction of every KV access served by each
            tier, and the capacity left for placement. A faster KV tier evicts
            an equal mass of weights, so the tier is searched with the staging
            reserve rather than fixed.
            """
            order = {"gpu": ["gpu", "host", "cxl"],
                     "host": ["host", "cxl"],
                     "cxl": ["cxl"]}[tier]
            caps = dict(_caps0)
            split, rem = {}, _kv_mean
            for t in order:
                take = min(rem, caps[t])
                if take > 0:
                    split[t] = take / _kv_mean if _kv_mean else 0.0
                    caps[t] -= take
                    rem -= take
            if rem > 0:   # nowhere left: the tail stays on the slowest KV tier
                split[order[-1]] = split.get(order[-1], 0.0) + rem / _kv_mean
                caps[order[-1]] = 0
            return split, caps

        _kv_split, _caps = _kv_place(KV_TIER)

        def _kv_time(n):
            """Time to move n KV bytes, split across the tiers that hold them."""
            return sum(transfer_time_s(n * f, _kv_tiers[t])
                       for t, f in _kv_split.items() if f > 0)

        # Staging reserve. Bytes held back from placement become in-flight room
        # for the K-deep prefetcher, so NAND transfers can overlap the other
        # tiers' reads instead of serializing behind them; the cost is more NAND
        # residency. The optimum depends on tier bandwidths, model size and
        # capacity, so it is searched per configuration.
        _dev_total = _caps["cxl"]

        _host_total = _caps["host"]

        def _plan(dev_res, host_res):
            """Place with dev_res and host_res bytes held back as staging room,
            and time one decode step.

            Host DRAM is off the CXL link, so a buffer there lets device and NAND
            transfers run ahead without competing for the link, but the weights
            it displaces then cross the link.
            """
            pl, lt, kvi, sp, fr = semantic_aware_placement(
                layers, max(0, _host_total - host_res),
                max(0, _dev_total - dev_res), num_blocks,
                gpu_cap=_caps["gpu"])
            # A representative step: mid-generation, so the KV read is the one
            # placement was sized for.
            _mid = PREFILL_TOKENS + TOKENS / 2.0
            us = []
            for _i, _L in enumerate(layers):
                _by = {t: _L["bytes"] * f for t, f in fr[_i].items()}
                if kvi[_L["name"]] > 0:
                    _kb = _mid * kvi[_L["name"]] * BATCH_SIZE
                    for _t, _f in _kv_split.items():
                        _lbl = {"gpu": PL_GPU_HBM, "host": PL_HOST_DRAM,
                                "cxl": PL_CXL_DEV_DRAM}[_t]
                        _by[_lbl] = _by.get(_lbl, 0.0) + _kb * _f
                us.append((_by, compute_time_s(_L["flops"] * BATCH_SIZE, cpu_cores)))
            return pipelined_time_s(us, PREFETCH_QUEUE_DEPTH, TIER_BW, TIER_LAT,
                                    inflight_budget=dev_res + host_res), \
                   (pl, lt, kvi, sp, fr)

        # KV tier and reserves are searched jointly: where the cache sits changes
        # which tier is the bottleneck, and so how much staging room is worth
        # holding back. Reserve fractions run to 0.50 because leaving bytes on
        # NAND, to use an otherwise idle bus, can need a large reserve on a big
        # device.
        _fr = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20,
               0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
        _kv_opts = ["cxl", "host"] + (["gpu"] if gpu_hbm_capacity_bytes else [])
        def build_units(fr, kvi, kv_split, token_step, pinned=()):
            """Per-step unit list, shared by the plan evaluator and the decode
            loop so that the search times exactly what decode runs."""
            us = []
            for _i, _L in enumerate(layers):
                _by = {}
                for t_, f_ in fr[_i].items():
                    # a layer Phase-1 staged into the device cache reads from
                    # device DRAM rather than NAND for the whole session
                    tt = (PL_CXL_DEV_DRAM
                          if t_ == PL_CXL_DEV_NAND and _i in pinned else t_)
                    _by[tt] = _by.get(tt, 0.0) + _L["bytes"] * f_
                if kvi[_L["name"]] > 0:
                    _kb = (PREFILL_TOKENS + token_step) * kvi[_L["name"]] * BATCH_SIZE
                    for _t2, _f2 in kv_split.items():
                        _lbl = {"gpu": PL_GPU_HBM, "host": PL_HOST_DRAM,
                                "cxl": PL_CXL_DEV_DRAM}[_t2]
                        _by[_lbl] = _by.get(_lbl, 0.0) + _kb * _f2
                us.append((_by, compute_time_s(_L["flops"] * BATCH_SIZE, cpu_cores)))
            return us

        # Two-stage search. The one-step estimate misorders near-tied plans, so
        # it only shortlists; finalists are timed exactly over all TOKENS steps
        # with growing KV and the realized best is kept. The search also runs
        # with the device clamped to each smaller standard size (32, 48 GB), so
        # a larger device's candidates include every smaller device's plan and
        # decode time cannot get worse with more capacity.
        _cands = []
        _dev_caps = sorted({cxl_dev_dram_capacity_bytes}
                           | {c * GiB for c in (32, 48)
                              if c * GiB < cxl_dev_dram_capacity_bytes})
        for _devcap in _dev_caps:
            _caps0["cxl"] = _devcap
            for _kvt in _kv_opts:
                _kv_split, _caps = _kv_place(_kvt)
                _dev_total, _host_total = _caps["cxl"], _caps["host"]
                for _df in _fr:
                    for _hf in _fr:
                        _dr, _hr = _dev_total * _df, _host_total * _hf
                        _t, _p = _plan(_dr, _hr)
                        _cands.append((_t, _kvt, _dr, _hr, _p, _devcap))
        # Shortlist per (device cap, KV tier) group, not globally: the one-step
        # estimate misranks NAND-heavy plans against all-resident ones, and a
        # global top-N could drop a whole group.
        _cands.sort(key=lambda x: x[0])
        _bygroup = {}
        for _cd in _cands:
            _bygroup.setdefault((_cd[5], _cd[1]), []).append(_cd)
        _cands = [c for g in _bygroup.values() for c in g[:6]]
        _cands.sort(key=lambda x: x[0])

        def _exact(kvt, dr, hr, plan):
            """Realized decode time for a finalist: every step, growing KV."""
            kv_split, caps = _kv_place(kvt)
            pl, lt, kvi, sp, fr = plan
            tot = 0.0
            for _tk in range(TOKENS):
                us = build_units(fr, kvi, kv_split, _tk)
                lat = pipelined_time_s(us, PREFETCH_QUEUE_DEPTH, TIER_BW,
                                       TIER_LAT, inflight_budget=dr + hr)
                lat += kv_growth_spill_time_s(
                    _kv_per_tok * (PREFILL_TOKENS + _tk + 1), _kv_mean,
                    _kv_tiers[max(kv_split, key=kv_split.get)])
                tot += lat
            return tot

        _best_t, _best, _prefetch_reserve = None, None, 0.0
        _best_kv = KV_TIER
        for _t0, _kvt, _dr, _hr, _p, _devcap in _cands:
            _caps0["cxl"] = _devcap          # _kv_place inside _exact reads it
            _tr = _exact(_kvt, _dr, _hr, _p)
            if _best_t is None or _tr < _best_t:
                _best_t, _best, _best_kv = _tr, _p, _kvt
                _prefetch_reserve, _dev_res, _host_res = _dr + _hr, _dr, _hr
                _best_devcap = _devcap
        _caps0["cxl"] = _best_devcap
        # Re-establish the winning cache placement: _kv_split and _caps are read
        # by _kv_time and by the decode loop.
        _kv_split, _caps = _kv_place(_best_kv)
        _dev_total, _host_total = _caps["cxl"], _caps["host"]
        place, ltypes, inc, sparsity, frac = _best
        _dev_for_weights = max(0, _dev_total - _dev_res)

        tmon    = DuplexTrafficMonitor()
        _placed_in_dev = sum(L["bytes"] for i, L in enumerate(layers)
                             if place[i] == PL_CXL_DEV_DRAM)
        # Phase-1 staging room comes only from the staging reserve, never from
        # device DRAM that placed weights or the KV cache already occupy. It is
        # the share STAGE_RESERVE_FRAC grants (0 by default). Bytes Phase 1 pins
        # stay resident for the session, so decode deducts them from the
        # prefetcher's in-flight budget.
        _stage_room = _prefetch_reserve * STAGE_RESERVE_FRAC
        _cache_cap = _stage_room
        cache   = AttentionGuidedCache(_cache_cap)
        sched   = DuplexScheduler(IO_THREAD_POOL_SIZE)
        threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

        # Two-queue CXL link, 27 GB/s and 505 ns per transaction, as CXL_DRAM in
        # tiers.py.
        link = CXLLink(bw_gbps=27.0, txn_overhead_s=505e-9)
        queue_uwrite_samples = []   # per-token Uwrite measured from the queue

        pf_time_val, pf_stats = run_prefill_chunked(
            layers, place, ltypes, sparsity, inc,
            cache, tmon, sched, threads, seq_len, frac)

        # Decode loop.
        total_spar_flops           = 0
        per_token_write_stall_pcts = []
        rows                       = []
        lat                        = 0.0
        # Throughput uses the mean step time over all tokens, as in
        # FlexGen/LIA/LLMFlash.
        total_decode_lat           = 0.0

        for token_step in range(TOKENS):
            lat               = 0.0
            # One builder for evaluator and decode; see build_units.
            units             = build_units(frac, inc, _kv_split, token_step,
                                            cache.session_pinned)
            step_stall_s      = 0.0
            step_time_s       = 0.0
            nand_link_free_at = 0.0
            link.reset()  # fresh queue state per token (matches `lat` semantics)
            fetched           = set()

            # Each decode step starts with idle I/O threads.
            for t in threads:
                t.busy_until   = 0.0
                t.current_task = None

            for idx, L in enumerate(layers):
                sz        = L["bytes"]
                has_kv    = inc[L["name"]] > 0
                # Each of the B sequences runs the layer. No sparsity discount:
                # sparsity guides placement here. Compute skipping belongs to
                # the LLMFlash baseline.
                eff_flops = L["flops"] * BATCH_SIZE

                # Low-batch MLP streaming: only the union of the B sequences'
                # active neurons, 1 - (1 - p)^B of the layer, is transferred
                # (LLMFlash sliding window).
                if ltypes[idx] == LayerType.MLP and BATCH_SIZE < ADAPTIVE_BATCH_THRESHOLD:
                    p = sparsity[idx]            # MLP layer sparsity
                    active_fraction = 1.0 - (1.0 - p) ** BATCH_SIZE
                    sz = int(sz * active_fraction)

                # Accumulate sparsity savings only on first token
                if token_step == 0:
                    total_spar_flops += int(L["flops"] * sparsity[idx])

                sched.adjust_thread_allocation(tmon)

                # Prefetch NAND-resident sub-layers among the next
                # PREFETCH_QUEUE_DEPTH, attention first, then by sparsity.
                pf_cands = []
                for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
                    fi = idx + i
                    if fi < len(layers) and place[fi] == PL_CXL_DEV_NAND \
                            and fi not in fetched:
                        p_score = 2.0 if ltypes[fi] == LayerType.ATTENTION else sparsity[fi]
                        pf_cands.append((fi, p_score))
                pf_cands.sort(key=lambda x: x[1], reverse=True)

                for t in threads:
                    if t.busy_until <= lat and pf_cands:
                        fi, score = pf_cands.pop(0)
                        fetched.add(fi)
                        dur   = transfer_time_s(layers[fi]["bytes"], CXL_SSD_NAND)
                        start = max(lat, t.busy_until, nand_link_free_at)
                        t.busy_until      = start + dur
                        nand_link_free_at = t.busy_until
                        t.current_task    = (fi, layers[fi]["bytes"])
                        cache.set_attention_score(fi, score)

                # A NAND layer not yet in the cache waits for its prefetch.
                src       = place[idx]
                raw_stall = 0
                if place[idx] == PL_CXL_DEV_NAND and not cache.contains(idx, sz):
                    src         = "CXL NAND (Stall)"
                    stall_until = 0
                    for th in threads:
                        if th.current_task and th.current_task[0] == idx:
                            stall_until = th.busy_until
                            break
                    if stall_until == 0:
                        stall_until = lat + transfer_time_s(sz, CXL_SSD_NAND)
                    # raw_stall only enters the per-layer report row. The NAND
                    # transfer is charged once, by `mem` below, and overlapped
                    # with compute via max(comp, mem), as in the baselines.
                    raw_stall = max(0, stall_until - lat)

                comp  = compute_time_s(eff_flops, cpu_cores)
                mem   = split_time_s(
                    sz, frac[idx], idx in cache.session_pinned)

                # Growing KV: attention reads all PREFILL_TOKENS + token_step
                # cached positions, timed at the KV tier(s) and serialized with
                # the weight read in `mem`.
                if has_kv:
                    kv_positions_cached = PREFILL_TOKENS + token_step
                    kv_read_bytes = kv_positions_cached * inc[L["name"]] * BATCH_SIZE
                    kv_read_time  = _kv_time(kv_read_bytes)
                    mem += kv_read_time

                ltime = max(comp, mem)

                tmon.record_read(sz)

                # Duplex write accounting. The CXL link has independent Rx
                # (read) and Tx (write) lanes. A KV write is scheduled on Tx
                # with the end of this layer's read window as its deadline, and
                # only the slip past the deadline is charged as stall.
                ltime_base = ltime  # snapshot before any KV adjustments

                # Track Rx-lane usage for CXL-resident reads (Host DRAM reads
                # do not touch the CXL link).
                if place[idx] in (PL_CXL_DEV_DRAM, PL_CXL_DEV_NAND):
                    link.schedule_read(lat, sz)

                if has_kv:
                    kv_write_scaled = inc[L["name"]] * BATCH_SIZE
                    tmon.record_write(kv_write_scaled)
                    sched.pending_kv_writebacks += kv_write_scaled

                    if place[idx] == PL_GPU_HBM:
                        # KV write lands in HBM; no CXL link involvement.
                        pass

                    elif place[idx] == PL_HOST_DRAM:
                        # Off the CXL link, so no Tx overlap: the KV write is
                        # serialized, timed at the tier holding the cache.
                        kv_write_t = _kv_time(kv_write_scaled)
                        exposed_stall = kv_write_t
                        tmon.write_stall_time_s += exposed_stall
                        tmon.write_stall_count  += 1
                        ltime        += exposed_stall
                        step_stall_s += exposed_stall

                    elif place[idx] in (PL_CXL_DEV_DRAM, PL_CXL_DEV_NAND):
                        # Tx-lane write, concurrent with the Rx read.
                        exposed = link.schedule_write_background(
                            lat, kv_write_scaled, deadline=lat + ltime_base)
                        if exposed > 0:
                            tmon.write_stall_time_s += exposed
                            tmon.write_stall_count  += 1
                            ltime        += exposed
                            step_stall_s += exposed
                        # otherwise the write hid under the read window

                lat         += ltime
                step_time_s += ltime
                tmon.record_layer_time(ltime)

                # Completed prefetches enter the cache.
                for t in threads:
                    if t.current_task and t.busy_until <= lat:
                        cache.add(t.current_task[0], t.current_task[1])
                        t.current_task = None

                # Per-layer report rows come from the first token.
                if token_step == 0:
                    rows.append({
                        "Layer":        idx + 1,
                        "Name":         L["name"],
                        "Type":         ltypes[idx].value,
                        "Sparsity":     f"{sparsity[idx]:.1%}",
                        "Placement":    place[idx],
                        "Served_From":  src,
                        "Layer_Time_s": ltime + raw_stall,
                        "Read_Ratio":   f"{tmon.get_read_ratio():.2%}"
                    })

            # Write stall as a percentage of this token's step time.
            stall_pct_this_token = (step_stall_s / step_time_s * 100) \
                                   if step_time_s > 0 else 0.0
            per_token_write_stall_pcts.append(stall_pct_this_token)

            queue_uwrite_samples.append(link.write_utilization_pct())
            # Step time comes from the pipeline model: tiers run concurrently
            # and a transfer may be issued up to PREFETCH_QUEUE_DEPTH sub-layers
            # early. The serial `lat` above only feeds the reported statistics
            # (per-layer rows, write stalls, Uwrite).
            lat = pipelined_time_s(units, PREFETCH_QUEUE_DEPTH, TIER_BW,
                                   TIER_LAT,
                                   inflight_budget=max(0.0, _prefetch_reserve
                                                       - pf_stats["bytes_prefetched"]))
            lat += kv_growth_spill_time_s(
                _kv_per_tok * (PREFILL_TOKENS + token_step + 1),
                _kv_mean, _kv_tiers[max(_kv_split, key=_kv_split.get)])
            total_decode_lat += lat

        avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)

        total_layers = len(rows)
        misses       = sum(1 for r in rows if "Stall" in r["Served_From"])
        hit_rate     = ((total_layers - misses) / total_layers) * 100

        print(f"Cache_Hit_Rate: {hit_rate:.4f}%")
        comb = combine_sublayer_stats(rows)
        print(pd.DataFrame(comb).to_string())
        mean_lat = total_decode_lat / TOKENS
        print(f"\nSingle-token decode latency: {mean_lat:.6f}s")

        print(f"Warmup time: {pf_stats['warmup_time']:.6f}")
        print(f"Warmup staged bytes: {pf_stats['bytes_prefetched']}")
        print(f"Decode throughput: {BATCH_SIZE / mean_lat:.6f} t/s")
        # Per-sequence prefill TPS, the convention of flexgen_baseline.py and
        # lia_baseline.py.
        print(f"Prefill throughput: {PREFILL_TOKENS / pf_time_val:.6f} t/s")

        total_model_size = sum(L["bytes"] for L in layers)
        total_time       = ssd_cold_time_s(total_model_size) + pf_time_val + total_decode_lat
        print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / total_time:.3f} t/s")
        print(f"Sparsity-based FLOP savings: {total_spar_flops:,}")

        weight_rd    = sum(L["bytes"] for L in layers)
        kv_wr        = sum(L.get("kv_cache_bytes", 0) for L in layers) / 512
        total_io_vol = weight_rd + kv_wr

        print(f"Read_Op_Percent: {(weight_rd / total_io_vol) * 100:.4f}%")
        print(f"Write_Op_Percent: {(kv_wr / total_io_vol) * 100:.4f}%")

        measured_ratio = tmon.get_read_ratio() * 100
        final_util_ratio = (weight_rd / total_io_vol) * 100 \
                           if measured_ratio >= 100.0 and kv_wr > 0 \
                           else measured_ratio

        print(f"Read_Ratio: {final_util_ratio:.4f}%")
        print(f"Write_Util_Pct: {tmon.get_write_util_pct():.4f}%")

        # Queue-measured Uwrite, averaged over tokens.
        queue_uwrite_avg = (sum(queue_uwrite_samples) / len(queue_uwrite_samples)) \
                           if queue_uwrite_samples else 0.0
        print(f"Queue_Uwrite_Pct: {queue_uwrite_avg:.4f}%")

        total_decode_layers = len([L for L in layers if inc[L["name"]] > 0])
        stall_freq_pct      = (tmon.write_stall_count / (total_decode_layers * TOKENS) * 100) \
                              if total_decode_layers > 0 else 0.0

        print(f"Write_Stall_Time_s: {tmon.write_stall_time_s / TOKENS:.6f}")
        print(f"Write_Stall_Pct: {avg_write_stall_pct:.4f}%")
        print(f"Write_Stall_Count: {tmon.write_stall_count}")
        print(f"Write_Stall_Freq_Pct: {stall_freq_pct:.4f}%")
        print(f"Per_Token_Write_Stall_Pcts: {','.join(f'{x:.6f}' for x in per_token_write_stall_pcts)}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    run_semantic_duplex_simulation()
    #track_write_stalls()
