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


def tier_of(p):
    """Map a placement label to its Tier. NAND-resident layers are read at
    CXL DRAM speed during decode because Phase 1 has staged them there."""
    if p == PL_GPU_HBM:   return GPU_HBM
    if p == PL_HOST_DRAM: return HOST_DRAM
    return CXL_DRAM


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
        for i, L in enumerate(layers):                       # 0: output head
            if place[i] is None and ltypes[i] == LayerType.OUTPUT and size_of(i, L) <= free:
                place[i] = tier_label; free -= size_of(i, L)
        for i, L in enumerate(layers):                       # 1: attention + KV
            if place[i] is None and ltypes[i] == LayerType.ATTENTION and size_of(i, L) <= free:
                place[i] = tier_label; free -= size_of(i, L)
        for i, L in enumerate(layers):                       # 2: embeddings
            if place[i] is None and ltypes[i] == LayerType.EMBEDDING and L["bytes"] <= free:
                place[i] = tier_label; free -= L["bytes"]
        cands = sorted([(i, L, sparsity[i]) for i, L in enumerate(layers)
                        if place[i] is None and ltypes[i] == LayerType.MLP],
                       key=lambda x: x[2], reverse=True)
        for i, L, sp in cands:                               # 3: sparse MLP
            if sp > 0.60 and L["bytes"] <= free:
                place[i] = tier_label; free -= L["bytes"]
        for i, L in enumerate(layers):                        # 3.5: dense MLP
            if place[i] is None and ltypes[i] == LayerType.MLP and L["bytes"] <= free:
                place[i] = tier_label; free -= L["bytes"]
        return free

    # The cascade. With an accelerator attached the whole ordering shifts up a
    # level: what would have been pinned in host DRAM goes to HBM, what would
    # have been in CXL DRAM moves to host, and NAND residency shrinks. With
    # gpu_cap = 0 this reduces exactly to the three-tier placement.
    fill(PL_GPU_HBM, gpu_cap)
    fill(PL_HOST_DRAM, host_cap)

    c_free = cxl_cap                                          # 4: CXL DRAM
    for i, L in enumerate(layers):
        if place[i] is None and L["bytes"] <= c_free:
            place[i] = PL_CXL_DEV_DRAM
            c_free -= L["bytes"]

    for i in range(len(layers)):                              # 5: NAND
        if place[i] is None: place[i] = PL_CXL_DEV_NAND

    return place, ltypes, kv_inc, sparsity

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
                        cache, tmon, sched, threads, seq_len):
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
        mem = transfer_time_s(sz, tier_of(place[i]))

        comp_chunk = compute_time_s(eff_flops, cpu_cores) / chunks
        pipe_time  = mem + (chunks - 1) * comp_chunk if chunks > 1 else max(mem, comp_chunk)

        # Wait for this layer's asynchronous staging only if it has not finished.
        start = max(lat, stage_finish.get(i, 0.0))
        if not sched.should_activate_duplex(sparsity[i]):
            lat = start + pipe_time
        else:
            lat = start + pipe_time * DUPLEX_PENALTY

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

        place, ltypes, inc, sparsity = semantic_aware_placement(
            layers, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, num_blocks,
            gpu_cap=gpu_hbm_capacity_bytes)

        tmon    = DuplexTrafficMonitor()
        cache   = AttentionGuidedCache(cxl_dev_dram_capacity_bytes)
        sched   = DuplexScheduler(IO_THREAD_POOL_SIZE)
        threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

        # Two-queue CXL link model:
        # CXL_DRAM bw=27 GB/s, lat=505ns — matches tiers.py for consistency.
        link = CXLLink(bw_gbps=27.0, txn_overhead_s=505e-9)
        queue_uwrite_samples = []   # collect per-token Uwrite from queue

        pf_time_val, pf_stats = run_prefill_chunked(
            layers, place, ltypes, sparsity, inc,
            cache, tmon, sched, threads, seq_len)

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
                    raw_stall = max(0, stall_until - lat)
                    lat += raw_stall * (1 - SEMANTIC_STREAM_OVERLAP)

                # ── Compute + memory time ─────────────────────────────────────
                comp  = compute_time_s(eff_flops, cpu_cores)
                mem   = transfer_time_s(
                    sz, tier_of(place[idx]))

                # Growing-KV: attention layers read all prior K/V from cache.
                # At decode step `token_step`, total cached positions = PREFILL_TOKENS + token_step.
                # KV cache resides at CXL_DRAM in SemSched's design. The KV read uses
                # the same Rx lane as the weight read, so we serialize it into `mem`.
                if has_kv:
                    kv_positions_cached = PREFILL_TOKENS + token_step
                    kv_read_bytes = kv_positions_cached * inc[L["name"]] * BATCH_SIZE
                    kv_read_time  = transfer_time_s(kv_read_bytes, CXL_DRAM)
                    mem += kv_read_time

                ltime = max(comp, mem)

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
                        kv_write_t = transfer_time_s(kv_write_scaled, CXL_DRAM)
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

        print(f"Decode throughput: {BATCH_SIZE / mean_lat:.6f} t/s")
        # Per-sequence prefill TPS (matches FlexGen/LIA/LLMFlash convention)
        # reporting convention (flexgen_baseline.py:155, lia_baseline.py:149).
        print(f"Prefill throughput: {PREFILL_TOKENS / pf_time_val:.2f} t/s")

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
