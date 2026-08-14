# semduplex_scheduler.py
# ------------------------------------------------------------
# SIMULATOR: Semantic-Aware Duplex CXL Scheduler (SemSched)
# ------------------------------------------------------------
#
# Models hybrid-CXL LLM inference with:
#   - sub-layer semantic placement (Attention / MLP placed by type & sparsity)
#   - 16-thread parallel NAND warmup before decode
#   - deep-lookahead (K=32) sparsity-ranked prefetching
#   - two-queue full-duplex CXL link model (cxl_link.py) for honest
#     read/write concurrency accounting:
#       - independent Rx and Tx lanes (27 GB/s each, 505 ns overhead),
#       - writes fitting under the read window are truly free (no penalty),
#       - writes exceeding the window expose their slip as actual stall,
#       - Uwrite is *measured* from the queue (tx_busy_total / t_end),
#         not derived from post-hoc byte accounting.
#
# Prefill throughput is reported per-sequence (PREFILL_TOKENS / pf_time_val)
# to match the convention of the baselines (FlexGen, LIA, LLMFlash).
#
# Decode-phase modeling:
#   - Growing KV: at decode step t, attention reads (PREFILL_TOKENS + t) ×
#     per-token-KV-bytes × BATCH_SIZE from CXL DRAM (linear-in-context KV growth).
#   - Adaptive low-batch streaming: when BATCH_SIZE < ADAPTIVE_BATCH_THRESHOLD,
#     MLP transfer bytes are scaled by active_fraction = 1 - (1 - sparsity)^B
#     (mimicking the LLMFlash sliding-window policy for batch-1 efficiency).
# ------------------------------------------------------------

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

# Which tier holds the KV cache. This is a placement decision, not a constant of
# the hardware, and the baselines disagree about it: FlexGen and LIA put KV in
# host DRAM, CXLAimPod in the device, LLM-in-a-Flash in a unified pool. SemSched
# has always used the device, on the argument that weights are read-only and so
# NAND-safe while KV is written every step and can never spill -- so the tier
# weights can least afford to lose is the one KV should occupy.
#
# That argument is about *safety*, not speed, and it was never measured. The
# knob exists so it can be. A tier is charged both ways: capacity is reserved
# from it before placement, and every KV read and write is timed at its
# bandwidth. If the cache does not fit, the remainder spills to the next slower
# tier and is timed there -- the rule now applied to every simulator.
KV_TIER = "cxl"          # "cxl" | "host" | "gpu"

# Which order placement considers sub-layers in. "semantic" is the paper's
# contribution and the default, so the shipped behaviour is unchanged. The
# alternatives exist so the contribution can be ABLATED: the measured advantage
# has so far been attributed to the staging reserve and the prefetch depth,
# neither of which is semantic, and the ordering itself had never been tested
# against a null. If semantic ties sequential, the paper's headline mechanism is
# not what produces the result and the framing has to change.
import os as _os
PLACEMENT_ORDER = _os.environ.get("SEMSCHED_PLACEMENT_ORDER", "semantic")

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

# Adaptive low-batch threshold. Below this batch size, MLP transfers are scaled
# by the LLMFlash-style active-fraction formula 1 - (1 - p)^B. At B >= threshold,
# the active fraction saturates near 1.0 and we fall back to full semantic
# placement transfer. Crossover at B=4 is observed in the LLMFlash baseline runs.
ADAPTIVE_BATCH_THRESHOLD = 4

OVERLAP_ATTENTION = 0.35
OVERLAP_MLP       = 0.65
OVERLAP_NORM      = 0.45
OVERLAP_OTHER     = 0.45

# AGGRESSIVE SUB-LAYER STREAMING OVERLAP (PhD Research Contribution)
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
                "sparsity":       0.05,  # Attention: structurally dense (paper §3.1)
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
                "sparsity":       L.get("sparsity", 0.54),  # FIX: propagate from parent
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

    # ── These 3 were accidentally deleted — restore them ──
    def get_read_ratio(self):
        r = sum(self.read_history)
        w = sum(self.write_history)
        return r / (r + w) if (r + w) > 0 else 0.5

    def needs_read_injection(self):
        return self.get_read_ratio() < 0.45

    def needs_write_injection(self):
        return self.get_read_ratio() > 0.6

    # ── New time-based metric ──
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

    A NAND-resident layer is read at CXL DRAM speed ONLY if Phase 1 actually
    staged it into the device cache. Previously this returned CXL_DRAM
    unconditionally, so 73 GB of NAND traffic was billed at 27 GB/s instead of
    5 GB/s every decode step -- a 5.4x under-charge that accounted for the
    reported speedup on its own.
    """
    if p == PL_GPU_HBM:   return GPU_HBM
    if p == PL_HOST_DRAM: return HOST_DRAM
    if p == PL_CXL_DEV_NAND and not staged:
        return CXL_SSD_NAND
    return CXL_DRAM


def split_time_s(nbytes, fr, staged=True):
    """Time to move `nbytes` of a layer split across tiers by the map `fr`.

    Placement fills each tier exactly, so a sub-layer may straddle a boundary.
    Each share is timed at the tier that actually holds it; with a single-tier
    layer this reduces to transfer_time_s(nbytes, tier_of(...)) exactly.
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

    # Bytes of each layer not yet assigned to a tier, and the split that results.
    # Placement used to be all-or-nothing: a sub-layer that did not fit whole was
    # passed over and the remaining capacity was left empty, so 0.91 GB of fast
    # memory sat unused at FP16 16H+48C while the bytes that could have occupied
    # it were read from NAND at 5 GB/s instead. Every tier is now filled exactly,
    # a layer splitting across the boundary when it straddles one.
    #
    # This changes the packing, not the policy. The semantic priority order below
    # is untouched -- output head, then attention with its KV, then embeddings,
    # then MLP by descending sparsity -- and what a split layer does is occupy
    # the tier its priority earned it, up to the capacity that tier has left.
    rem  = [float(L["bytes"]) for L in layers]
    frac = [dict() for _ in layers]

    def take(i, tier_label, free, extra=0.0):
        """Move as much of layer i into `tier_label` as fits. Returns new free."""
        if free <= 0 or rem[i] <= 0:
            return free
        # Attention layers must also hold their growing KV increment; that part
        # is indivisible, so it is only charged when the whole layer fits.
        want = rem[i] + (extra if rem[i] == layers[i]["bytes"] else 0.0)
        got  = min(want, free)
        if got <= 0:
            return free
        moved = min(got, rem[i])
        frac[i][tier_label] = frac[i].get(tier_label, 0.0) + moved / layers[i]["bytes"]
        rem[i] -= moved
        return free - got

    def fill(tier_label, cap):
        """Apply the semantic priority order into one tier; return unused bytes.

        The order is the contribution: output head first (touched every step),
        then attention with its KV (latency-critical), then embeddings, then MLP
        by descending activation sparsity (least bandwidth per useful FLOP),
        then whatever dense MLP still fits.
        """
        free = cap
        if free <= 0:
            return free
        if PLACEMENT_ORDER != "semantic":
            # Null orderings, for the ablation. Same tiers, same capacities, same
            # bytes -- only the sequence in which sub-layers claim fast memory.
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

    # The cascade. With an accelerator attached the whole ordering shifts up a
    # level: what would have been pinned in host DRAM goes to HBM, what would
    # have been in CXL DRAM moves to host, and NAND residency shrinks. With
    # gpu_cap = 0 this reduces exactly to the three-tier placement.
    fill(PL_GPU_HBM, gpu_cap)
    fill(PL_HOST_DRAM, host_cap)
    fill(PL_CXL_DEV_DRAM, cxl_cap)                            # 4: CXL DRAM

    for i, L in enumerate(layers):                            # 5: NAND
        if rem[i] > 0:
            frac[i][PL_CXL_DEV_NAND] = (frac[i].get(PL_CXL_DEV_NAND, 0.0)
                                        + rem[i] / L["bytes"])
            rem[i] = 0.0

    # ── Spread the NAND residue through execution order ──────────────────
    # Which bytes end up on NAND is fixed by capacity, but WHERE they sit in the
    # execution order is not, and a K-deep lookahead can only overlap a transfer
    # against work that runs before it. Filling MLP by descending sparsity puts
    # the leftover in one contiguous mid-model band -- at INT8 16H+32C, ten of
    # eleven NAND sub-layers landed at indices 30-48 of 163, so 5.7 GB of 5 GB/s
    # traffic saturates the backend in 12% of the step while the fast tiers idle
    # either side of it.
    #
    # Sub-layers of one kind are the same size, so moving the residue among them
    # is byte-neutral: the ledger is identical, only the positions change. That
    # makes this pure scheduling -- it cannot flatter the accounting, because it
    # does not touch it.
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

    # `place` names the tier holding the largest share. It drives the control
    # flow -- staging, prefetch, duplex lane selection -- which is per-layer and
    # cannot be fractional; `frac` carries the split that timing uses.
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

    # Phase 1: parallel NAND warmup (staging NAND-resident layers into device DRAM).
    # FIX (BigData 2026): the previous model divided the *whole* transfer time by
    # |T|=16, which divides the bandwidth term as well and implies ~80 GB/s from a
    # 5 GB/s device. Aggregate NAND bandwidth is a hard physical floor; a thread
    # pool can only amortize the per-chunk access latency. The resulting cost is
    # now charged into `lat` (previously it was computed and then discarded, so
    # staging was free -- see REVISION_PLAN.md Part 2.A).
    # Staging runs on the asynchronous I/O pool, i.e. on the NAND->device-DRAM
    # DMA path, which is a resource independent of the compute engine. It is
    # therefore OVERLAPPED with prefill compute, not serialized ahead of it: a
    # layer stalls only if its staging has not completed by the time execution
    # reaches it. `stage_finish[i]` is layer i's staging completion time on a
    # bandwidth-limited NAND channel (threads amortize per-chunk latency only).
    stage_finish = {}
    t_stage      = 0.0
    # ENABLE_PREFILL_WARMUP was declared but never read, so every ablation that
    # set it to False measured the warmup-ON configuration and reported it as
    # OFF. The flag now gates the staging loop it names.
    _stage_budget = cache.cap if ENABLE_PREFILL_WARMUP else 0
    for i, L in enumerate(layers):
        if place[i] == PL_CXL_DEV_NAND:
            bw_term  = L["bytes"] / CXL_SSD_NAND.bw_Bps
            # Serial per-chunk latency, NOT divided by the I/O thread pool.
            # No baseline models concurrent staging I/O, and FlexGen and
            # LLM-in-a-Flash both describe overlapped I/O in their own papers,
            # so crediting ourselves alone with it would be a strawman. The
            # division is inert in any case -- staging completes inside the
            # prefill slack, so stage_finish never binds and removing it changes
            # every reported figure by 0.000.
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
        # FIX (BigData 2026): prefill processes `seq_len` tokens for each of B
        # sequences, so FLOPs = 2*P*seq_len*B. The baselines previously used a
        # magic PREFILL_FLOP_MULTIPLIER=15 that SemSched did not apply at all,
        # giving SemSched a 15x compute advantage in prefill. Parameters remain
        # dense (no sparsity discount) -- prefill fires all neurons.
        eff_flops = L["flops"] * seq_len * BATCH_SIZE
        mem = split_time_s(sz, frac[i], i in cache.session_pinned)

        comp_chunk = compute_time_s(eff_flops, cpu_cores) / chunks
        # Chunked pipelining overlaps a chunk's compute with the next chunk's
        # transfer, but it can never take less time than the compute itself:
        # every FLOP is still executed. The unfloored form below charged
        # (chunks-1) compute chunks, one short of the total, which let prefill
        # finish ~1/chunks BELOW the 2*P*S*B compute floor -- impossible, and
        # worth about 10% at the chunk counts we use.
        full_comp  = comp_chunk * chunks
        # Chunked pipeline: transfer of chunk k+1 overlaps compute of chunk k,
        # so the cost is the slower stream plus one chunk of the faster one to
        # fill the pipe. The previous form, mem + (chunks-1)*comp_chunk, added
        # the FULL transfer to nearly-full compute and therefore only overlapped
        # when one term dominated; at mem ~= comp it charged close to their sum
        # -- 187.5 where 112.5 is correct at chunks=8 -- which produced a
        # non-monotonic dip to 0.85x exactly in the mixed regime the pipeline
        # exists to handle.
        pipe_time  = (max(mem, full_comp) + min(mem, full_comp) / chunks
                      if chunks > 1 else max(mem, comp_chunk))

        # Wait for this layer's asynchronous staging only if it has not finished.
        start = max(lat, stage_finish.get(i, 0.0))
        # The DUPLEX_PENALTY = 1.15 multiplier that used to be applied here is
        # gone. The decode path replaced it with the measured queue model
        # (see the Tx-lane scheduling below, which notes the replacement), but
        # prefill was never updated, so a blanket 15% survived on every layer
        # with sparsity > 0.20 -- and only at INT8/INT4, because
        # should_activate_duplex() gates on the quantization format. A data
        # format cannot affect link contention, no baseline carries any
        # equivalent term, and prefill writes one KV entry per sequence. It cost
        # 64.3 s of a 732.8 s INT8 prefill and nothing at FP16, which is what
        # made SemSched 11.5% above the compute floor at INT8 while every
        # baseline sat within 2.9% of it.
        lat = start + pipe_time

        tmon.record_read(sz)
    return lat, stats


def ssd_cold_time_s(n):
    return transfer_time_s(
        n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# ==========================================
# TRACK WRITE STALLS / PER-TOKEN LATENCY
# ==========================================
def track_write_stalls():
    print("\n" + "=" * 70)
    print("TRACKING PER-TOKEN LATENCY & STALLS FOR ALL SIMULATORS")
    print("=" * 70)
    
    results = []
    
    # We use the tight memory config (16GB Host + 32GB CXL) and B=128 
    # to properly expose the NAND stalls across the different precisions.
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
                
                # Recover the exact per-token execution time (Compute + Stall)
                # Since TPS = BatchSize / Per_Token_Latency
                latency_s = target_batch / tps if tps > 0 else 0.0
                
                # SemDuplex explicitly prints its single-token latency; let's grab it directly if available
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

        # The KV cache is resident in CXL device DRAM for the whole session, so
        # its bytes are NOT available to hold weights. Placement previously used
        # the full device capacity, so at 32H+64C it put 63.9 GB of MLP into the
        # same 64 GB the 20.6 GB KV cache also needs -- 32% over. KV belongs in
        # the slowest DRAM tier precisely because weights can least afford to
        # lose it: KV can never spill to NAND (it is written every step), while
        # weights are read-only and NAND-safe. Reserving it here is what makes
        # that placement honest.
        # KV GROWS. Reserving its final size starves the early steps of weight
        # capacity they could have used, and reserving its initial size
        # over-commits the device later. The device instead yields weight
        # capacity progressively as KV expands: at step t the space available
        # for weights is C - kv(P+t), so the time-averaged budget over the
        # generation is C - kv(P + T/2). Placement is a one-shot decision, so
        # the time-average is what it should see; the eviction that realises it
        # step by step is the runtime's job.
        _kv_per_tok = sum(2 * L.get("kv_heads", 8) * L.get("head_dim", 128)
                          * BYTES_PER_PARAM
                          for L in layers
                          if classify_layer_type(L) == LayerType.ATTENTION) * BATCH_SIZE
        _kv_mean = int(_kv_per_tok * (PREFILL_TOKENS + TOKENS / 2.0))

        # Charge the KV cache to whichever tier KV_TIER names, spilling down the
        # hierarchy when it does not fit. `_kv_split` is the fraction of every KV
        # access served by each tier, and is what the decode loop times against;
        # the capacities below are what placement is then allowed to use.
        _caps = {"gpu":  gpu_hbm_capacity_bytes,
                 "host": host_dram_capacity_bytes,
                 "cxl":  cxl_dev_dram_capacity_bytes}
        # The pristine capacities. _kv_place is called once per candidate tier by
        # the search below and must always subtract the cache from the FULL
        # device, never from a dict a previous call already reduced.
        _caps0 = dict(_caps)
        _kv_tiers = {"gpu": GPU_HBM, "host": HOST_DRAM, "cxl": CXL_DRAM}

        def _kv_place(tier):
            """Charge the cache to `tier`, spilling down when it does not fit.

            Returns (split, caps): the fraction of every KV access served by each
            tier, and what placement may then use. Which tier should hold the
            cache is not obvious and not constant -- putting it in HBM reads it
            66x faster but evicts an equal mass of weights, and the weights are
            read every step too. It is searched alongside the staging reserve
            rather than fixed, because the answer moves with quantization: at
            FP16 the device wins, at INT8 the host does.
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

        # ── Adaptive prefetch reservation ────────────────────────────────────
        # Every other policy fills each tier to 100%: FlexGen's LP maximises
        # residency by construction, and LIA, CXLAimPod and LLM-in-a-Flash place
        # greedily until full. That is optimal only if reads are served on
        # demand. With a K-deep lookahead it is not, because bytes fetched ahead
        # of consumption need somewhere to live, and a device filled to the brim
        # has nowhere -- which is why a K=32 queue over a full device buys 1.00x,
        # exactly what no prefetch at all buys.
        #
        # So hold some device DRAM back. The reserved bytes stop being resident,
        # which costs NAND traffic, and start being staging room, which lets the
        # NAND transfers overlap the host and accelerator reads instead of
        # serialising behind them. Neither effect dominates everywhere: too small
        # a reserve cannot cover a transfer, too large a one pushes more onto
        # NAND than overlap can hide. The optimum is interior and moves with the
        # tier bandwidths, the model size and the capacity, so it is searched per
        # configuration rather than tuned once.
        _dev_total = _caps["cxl"]

        _host_total = _caps["host"]

        def _plan(dev_res, host_res):
            """Place with bytes held back as staging, and time one decode step.

            Staging room need not come from the device. Host DRAM sits idle most
            of a step -- 0.42 s of 2.85 s at INT8 16H+32C -- and it is off the
            CXL link entirely, so a buffer there lets device and NAND transfers
            run ahead without competing for the link they are trying to escape.
            The weights it displaces do cross that link, so the trade is real in
            both directions and the balance is what the search resolves.
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

        # The search covers both placement decisions together, because they are
        # not separable: where the cache sits changes which tier is the
        # bottleneck, which changes how much staging room is worth holding back.
        _fr = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
        _kv_opts = ["cxl", "host"] + (["gpu"] if gpu_hbm_capacity_bytes else [])
        _best_t, _best, _prefetch_reserve = None, None, 0.0
        _best_kv = KV_TIER
        for _kvt in _kv_opts:
            _kv_split, _caps = _kv_place(_kvt)
            _dev_total, _host_total = _caps["cxl"], _caps["host"]
            for _df in _fr:
                for _hf in _fr:
                    _dr, _hr = _dev_total * _df, _host_total * _hf
                    _t, _p = _plan(_dr, _hr)
                    if _best_t is None or _t < _best_t:
                        _best_t, _best, _best_kv = _t, _p, _kvt
                        _prefetch_reserve, _dev_res, _host_res = _dr + _hr, _dr, _hr
        # Re-establish the winning cache placement: _kv_split and _caps are read
        # by _kv_time and by the decode loop, and the loop above left them on
        # whichever option it tried last.
        _kv_split, _caps = _kv_place(_best_kv)
        _dev_total, _host_total = _caps["cxl"], _caps["host"]
        place, ltypes, inc, sparsity, frac = _best
        _dev_for_weights = max(0, _dev_total - _dev_res)

        tmon    = DuplexTrafficMonitor()
        # The CXL device has ONE pool of DRAM. Placement has already consumed
        # part of it for PL_CXL_DEV_DRAM layers, so Phase 1 may only stage into
        # what is left. Handing the cache the full device capacity double-counts
        # it: at 16H+64C placement uses 63.02 GB of 64 GB, then staging asks for
        # a further 73.23 GB -- 213% of the device -- and every NAND-resident
        # layer is then billed at the 27 GB/s hit rate instead of 5 GB/s.
        # The KV cache is the other resident claim on that same pool, and it is
        # already subtracted from what placement was allowed to use
        # (_dev_for_weights). Subtracting only _placed_in_dev here handed Phase 1
        # the KV cache's own space as staging room: at INT8 16H+32C it staged
        # 5.13 GB into the 10.16 GB the KV cache occupies, so the one
        # configuration with any NAND residency read all of it at DRAM speed and
        # scored identically to 48C and 64C, which have none. Staging may only
        # use what is left of the WEIGHT budget.
        _placed_in_dev = sum(L["bytes"] for i, L in enumerate(layers)
                             if place[i] == PL_CXL_DEV_DRAM)
        _cache_cap = max(0, _dev_for_weights - _placed_in_dev)
        cache   = AttentionGuidedCache(_cache_cap)
        sched   = DuplexScheduler(IO_THREAD_POOL_SIZE)
        threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

        # Two-queue CXL link model:
        # CXL_DRAM bw=27 GB/s, lat=505ns — matches tiers.py for consistency.
        link = CXLLink(bw_gbps=27.0, txn_overhead_s=505e-9)
        queue_uwrite_samples = []   # collect per-token Uwrite from queue

        pf_time_val, pf_stats = run_prefill_chunked(
            layers, place, ltypes, sparsity, inc,
            cache, tmon, sched, threads, seq_len, frac)

        # ── Multi-token decode loop with per-token write stall tracking ────────
        total_spar_flops           = 0
        per_token_write_stall_pcts = []
        rows                       = []
        lat                        = 0.0
        # FIX (BigData 2026): accumulate every decode step so throughput is the
        # MEAN across tokens, matching FlexGen/LIA/LLMFlash. Previously SemSched
        # reported only the final step, which with growing-KV is its slowest.
        total_decode_lat           = 0.0

        for token_step in range(TOKENS):
            lat               = 0.0
            units             = []      # (bytes_by_tier, compute_s) per sub-layer
            step_stall_s      = 0.0
            step_time_s       = 0.0
            nand_link_free_at = 0.0
            link.reset()  # fresh queue state per token (matches `lat` semantics)
            fetched           = set()

            # Reset thread state each token — each decode step is independent
            for t in threads:
                t.busy_until   = 0.0
                t.current_task = None

            for idx, L in enumerate(layers):
                sz        = L["bytes"]
                has_kv    = inc[L["name"]] > 0
                # FIX (BigData 2026): (a) decode FLOPs scale with batch size --
                # each of the B sequences runs the layer; (b) the previous
                # (1 - sparsity) FLOP discount is removed. SemSched uses sparsity
                # for *placement*, not to skip compute (§III/§IV), and neither
                # FlexGen nor LIA received the discount. Sparsity-based compute
                # skipping belongs to LLMFlash, where it is now modeled.
                eff_flops = L["flops"] * BATCH_SIZE

                # Adaptive low-batch streaming for MLP layers.
                # At low batch, only `active_fraction` of MLP neurons fire, so transferring
                # the full MLP weight is wasteful. We mimic LLMFlash's sliding-window
                # approach by scaling MLP transfer bytes by 1 - (1 - p)^B (the union of
                # active sets across the batch). Above the threshold, semantic placement
                # is preferable so the scaling collapses back to ~1.0.
                if ltypes[idx] == LayerType.MLP and BATCH_SIZE < ADAPTIVE_BATCH_THRESHOLD:
                    p = sparsity[idx]            # MLP layer sparsity
                    active_fraction = 1.0 - (1.0 - p) ** BATCH_SIZE
                    sz = int(sz * active_fraction)

                # Accumulate sparsity savings only on first token
                if token_step == 0:
                    total_spar_flops += int(L["flops"] * sparsity[idx])

                sched.adjust_thread_allocation(tmon)

                # ── Aggressive Semantic Prefetcher ────────────────────────────
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

                # ── Serve layer from tier ─────────────────────────────────────
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
                    # DOUBLE-CHARGE REMOVED. stall_until is derived from the
                    # SAME NAND transfer that `mem` below charges in full, so
                    # adding (1-sigma) of it billed the transfer at 1.15x while
                    # every baseline bills it at 1.00x. That asymmetry, not any
                    # policy difference, was the whole +25% gap against the
                    # byte-accounting floor. The transfer is charged once, by
                    # `mem`, and overlapped with compute via max(comp, mem) --
                    # exactly as the baselines model it.
                    raw_stall = max(0, stall_until - lat)

                # ── Compute + memory time ─────────────────────────────────────
                comp  = compute_time_s(eff_flops, cpu_cores)
                mem   = split_time_s(
                    sz, frac[idx], idx in cache.session_pinned)

                # Growing-KV: attention layers read all prior K/V from cache.
                # At decode step `token_step`, total cached positions = PREFILL_TOKENS + token_step.
                # KV cache resides at CXL_DRAM in SemSched's design. The KV read uses
                # the same Rx lane as the weight read, so we serialize it into `mem`.
                if has_kv:
                    kv_positions_cached = PREFILL_TOKENS + token_step
                    kv_read_bytes = kv_positions_cached * inc[L["name"]] * BATCH_SIZE
                    kv_read_time  = _kv_time(kv_read_bytes)
                    mem += kv_read_time

                ltime = max(comp, mem)

                # Record this sub-layer for the pipeline engine: the bytes it
                # reads, per tier, and its compute. Weights follow `frac`; a
                # layer staged into the device cache reads from device DRAM
                # rather than NAND. The growing KV read is added at whichever
                # tier holds the cache.
                _by = {}
                for _t, _f in frac[idx].items():
                    _tt = (PL_CXL_DEV_DRAM
                           if _t == PL_CXL_DEV_NAND and idx in cache.session_pinned
                           else _t)
                    _by[_tt] = _by.get(_tt, 0.0) + sz * _f
                if has_kv:
                    _kvt = {"gpu": PL_GPU_HBM, "host": PL_HOST_DRAM,
                            "cxl": PL_CXL_DEV_DRAM}
                    for _t, _f in _kv_split.items():
                        _lbl = _kvt[_t]
                        _by[_lbl] = _by.get(_lbl, 0.0) + kv_read_bytes * _f
                units.append((_by, comp))

                tmon.record_read(sz)

                # ── Queue-based duplex write accounting ───────────────────────
                # The CXL link is modeled as two independent lanes (Rx for reads,
                # Tx for writes). For each CXL-resident read we register Rx-lane
                # busy time. For each KV write we schedule on Tx with deadline =
                # end of this layer's natural duration; if the write doesn't fit,
                # the queue returns the exposed slip which we charge as actual
                # stall (replaces the DUPLEX_PENALTY = 1.15 multiplier).
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
                        # Host DRAM is not on the CXL link; KV write serialized
                        # through host DDR.
                        # Timed at whatever tier holds the cache, not at a fixed
                        # CXL rate. The Tx-lane overlap below applies only to the
                        # device-resident case, which is what the duplex link
                        # models; writes are 0.017% of a step, so the branch that
                        # does not overlap costs nothing measurable either way.
                        kv_write_t = _kv_time(kv_write_scaled)
                        exposed_stall = kv_write_t
                        tmon.write_stall_time_s += exposed_stall
                        tmon.write_stall_count  += 1
                        ltime        += exposed_stall
                        step_stall_s += exposed_stall

                    elif place[idx] in (PL_CXL_DEV_DRAM, PL_CXL_DEV_NAND):
                        # Write on Tx lane, may run concurrently with Rx read.
                        # Deadline = end of this layer's natural read window.
                        exposed = link.schedule_write_background(
                            lat, kv_write_scaled, deadline=lat + ltime_base)
                        if exposed > 0:
                            tmon.write_stall_time_s += exposed
                            tmon.write_stall_count  += 1
                            ltime        += exposed
                            step_stall_s += exposed
                        # else: write fully hidden under read window → no stall
                # ─────────────────────────────────────────────────────────────

                lat         += ltime
                step_time_s += ltime
                tmon.record_layer_time(ltime)

                # ── Resolve completed prefetch tasks ──────────────────────────
                for t in threads:
                    if t.current_task and t.busy_until <= lat:
                        cache.add(t.current_task[0], t.current_task[1])
                        t.current_task = None

                # ── Record layer row only on first token ──────────────────────
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

            # ── Per-token: write stall as % of this token's total decode time ─
            stall_pct_this_token = (step_stall_s / step_time_s * 100) \
                                   if step_time_s > 0 else 0.0
            per_token_write_stall_pcts.append(stall_pct_this_token)

            # sample measured Tx-lane utilization from the queue
            queue_uwrite_samples.append(link.write_utilization_pct())
            # The step costs what the pipeline actually takes: tiers run
            # concurrently, and a transfer may be issued PREFETCH_QUEUE_DEPTH
            # sub-layers before it is needed. `lat` up to here was the serial
            # sum, which is the same number for any policy moving the same bytes.
            lat = pipelined_time_s(units, PREFETCH_QUEUE_DEPTH, TIER_BW,
                                   TIER_LAT, inflight_budget=_prefetch_reserve)
            lat += kv_growth_spill_time_s(
                _kv_per_tok * (PREFILL_TOKENS + token_step + 1),
                _kv_mean, _kv_tiers[max(_kv_split, key=_kv_split.get)])
            total_decode_lat += lat

        # ── Post-loop aggregates ───────────────────────────────────────────────
        avg_write_stall_pct = sum(per_token_write_stall_pcts) / len(per_token_write_stall_pcts)

        total_layers = len(rows)
        misses       = sum(1 for r in rows if "Stall" in r["Served_From"])
        hit_rate     = ((total_layers - misses) / total_layers) * 100

        print(f"Cache_Hit_Rate: {hit_rate:.4f}%")
        comb = combine_sublayer_stats(rows)
        print(pd.DataFrame(comb).to_string())
        mean_lat = total_decode_lat / TOKENS
        print(f"\nSingle-token decode latency: {mean_lat:.6f}s")

        # Phase-1 staging is a reported mechanism, so report it rather than
        # leaving it to be inferred from an ablation.
        print(f"Warmup time: {pf_stats['warmup_time']:.6f}")
        print(f"Warmup staged bytes: {pf_stats['bytes_prefetched']}")
        print(f"Decode throughput: {BATCH_SIZE / mean_lat:.6f} t/s")
        # Per-sequence prefill TPS (matches FlexGen/LIA/LLMFlash convention)
        # reporting convention (flexgen_baseline.py:155, lia_baseline.py:149).
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

        # queue-measured Uwrite (averaged across tokens)
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
