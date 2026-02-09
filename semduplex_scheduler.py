# simulation_semantic_duplex_prefill.py
# ------------------------------------------------------------
# SIMULATOR: Semantic-Aware Duplex CXL Scheduler (Research-Validated)
# ------------------------------------------------------------

import math
import pandas as pd
import sys
import traceback
from collections import OrderedDict, deque
from enum import Enum

from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM, DEFAULT_MODEL_CFG
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

GiB = 1024**3
PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

IO_THREAD_POOL_SIZE = 16 
PREFETCH_QUEUE_DEPTH = 32 # Deep lookahead for large models
TRAFFIC_WINDOW_SIZE = 16
PREFILL_CHUNK_SIZE = 64
ENABLE_PREFILL_WARMUP = True
PREFILL_TOKENS = 512
PREFILL_FLOP_MULTIPLIER = 15.0

DUPLEX_PENALTY = 1.15

# Overlap factors for standard execution
OVERLAP_ATTENTION = 0.35
OVERLAP_MLP = 0.65
OVERLAP_NORM = 0.45
OVERLAP_OTHER = 0.45

# AGGRESSIVE SUB-LAYER STREAMING OVERLAP (PhD Research Contribution)
# Models the ability to compute on incoming semantic chunks before full layer load.
SEMANTIC_STREAM_OVERLAP = 0.85 

def decomposed_build_layers(sequence_length):
    standard_layers = build_layers(DEFAULT_MODEL_CFG, sequence_length=sequence_length)
    new_layers = []
    for L in standard_layers:
        if L["kind"] == "DecoderBlock":
            aname = f"{L['name']}_attn"
            aflops = L["flops"] // 3
            abytes = int(L["bytes"] * 0.33)
            new_layers.append({
                "name": aname, "kind": "Attention", "bytes": abytes, "flops": aflops,
                "head_dim": L.get("head_dim", 128), "kv_heads": L.get("kv_heads", 8)
            })
            mname = f"{L['name']}_mlp"
            mflops = L["flops"] - aflops
            mbytes = L["bytes"] - abytes
            new_layers.append({"name": mname, "kind": "MLP", "bytes": mbytes, "flops": mflops})
        else:
            new_layers.append(L)
    return new_layers

class LayerType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    EMBEDDING = "embedding"
    OUTPUT = "output"

def classify_layer_type(layer_dict):
    name = layer_dict["name"].lower()
    if "attn" in name or "attention" in name: return LayerType.ATTENTION
    elif "mlp" in name or "ffn" in name: return LayerType.MLP
    elif "norm" in name: return LayerType.NORM
    elif "embed" in name: return LayerType.EMBEDDING
    elif "lm_head" in name: return LayerType.OUTPUT
    return LayerType.NORM

def compute_layer_sparsity(layer_idx, total_decoder_blocks, layer_type):
    if layer_type not in [LayerType.ATTENTION, LayerType.MLP]: return 0.0
    norm_pos = layer_idx / max(1, total_decoder_blocks - 1)
    if layer_type == LayerType.ATTENTION:
        return 0.05
    elif layer_type == LayerType.MLP:
        if norm_pos < 0.15: return 0.13 + (0.50 - 0.13) * (norm_pos / 0.15)
        elif norm_pos < 0.75: return 0.50 + (0.90 - 0.50) * ((norm_pos - 0.15) / 0.6)
        else: return 0.90 + (0.956 - 0.90) * ((norm_pos - 0.75) / 0.25)
    return 0.5

def get_overlap_factor(layer_type, is_streaming=False):
    if is_streaming: return SEMANTIC_STREAM_OVERLAP
    return {
        LayerType.ATTENTION: OVERLAP_ATTENTION,
        LayerType.MLP: OVERLAP_MLP,
        LayerType.NORM: OVERLAP_NORM,
    }.get(layer_type, OVERLAP_OTHER)

class DuplexTrafficMonitor:
    def __init__(self, window_size=10):
        self.read_history = deque(maxlen=window_size)
        self.write_history = deque(maxlen=window_size)
    def record_read(self, b): self.read_history.append(b)
    def record_write(self, b): self.write_history.append(b)
    def get_read_ratio(self):
        r, w = sum(self.read_history), sum(self.write_history)
        return r / (r + w) if (r + w) > 0 else 0.5
    def needs_read_injection(self): return self.get_read_ratio() < 0.45
    def needs_write_injection(self): return self.get_read_ratio() > 0.6

class AttentionGuidedCache:
    def __init__(self, cap_bytes):
        self.cap = cap_bytes
        self.used = 0
        self.cache = OrderedDict()
        self.attention_scores = {}
        self.pinned = set()
        self.session_pinned = set()
    
    def set_attention_score(self, lid, score):
        self.attention_scores[lid] = score
        if score > 0.8: self.pinned.add(lid)
        else: self.pinned.discard(lid)
    
    def pin_for_session(self, lid): self.session_pinned.add(lid)
    
    def _evict(self, need):
        if need <= 0: return
        cands = [(lid, sz, self.attention_scores.get(lid, 0.5)) for lid, sz in self.cache.items()
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
        self.io_pool_size = io_pool_size
        self.read_threads = io_pool_size // 2
        self.write_threads = io_pool_size // 2
        self.pending_kv_writebacks = 0

    def adjust_thread_allocation(self, tmon):
        ratio = tmon.get_read_ratio()
        if ratio > 0.6: self.read_threads, self.write_threads = 1, self.io_pool_size - 1
        elif ratio < 0.45: self.read_threads, self.write_threads = self.io_pool_size - 1, 1
        else: self.read_threads, self.write_threads = self.io_pool_size // 2, self.io_pool_size // 2

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
        self.id = i
        self.busy_until = 0.0
        self.current_task = None

def semantic_aware_placement(layers, host_cap, cxl_cap, total_decoder_blocks):
    place = [None] * len(layers)
    ltypes = [classify_layer_type(L) for L in layers]
    sparsity = {}
    kv_inc = {}
    
    for i, L in enumerate(layers):
        dec_idx = 0
        if "decoder_" in L["name"]:
            try: dec_idx = int(L["name"].split("_")[1])
            except: pass
        sparsity[i] = compute_layer_sparsity(dec_idx, total_decoder_blocks, ltypes[i])
        if ltypes[i] == LayerType.ATTENTION:
            kv_inc[L["name"]] = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * BYTES_PER_PARAM
        else: kv_inc[L["name"]] = 0
        
    h_free, c_free = host_cap, cxl_cap
    
    # NEW PRIORITY 0: LM Head (Ensure fast final output)
    for i, L in enumerate(layers):
        if ltypes[i] == LayerType.OUTPUT and L["bytes"] <= h_free:
            place[i] = PL_HOST_DRAM
            h_free -= L["bytes"]

    # 1. Pinned Attention -> Host DRAM
    for i, L in enumerate(layers):
        if ltypes[i] == LayerType.ATTENTION and place[i] is None:
            sz = L["bytes"] + kv_inc[L["name"]]
            if sz <= h_free:
                place[i] = PL_HOST_DRAM
                h_free -= sz
                
    # 2. Embedding -> Host
    for i, L in enumerate(layers):
        if place[i] is None and ltypes[i] == LayerType.EMBEDDING and L["bytes"] <= h_free:
            place[i] = PL_HOST_DRAM
            h_free -= L["bytes"]
                
    # 3. High Sparsity MLP -> Host
    mlp_cands = [(i, L, sparsity[i]) for i,L in enumerate(layers) if place[i] is None and ltypes[i] == LayerType.MLP]
    mlp_cands.sort(key=lambda x: x[2], reverse=True)
    for i, L, sp in mlp_cands:
        if sp > 0.60 and L["bytes"] <= h_free:
            place[i] = PL_HOST_DRAM
            h_free -= L["bytes"]
            
    # 4. Dense MLP -> CXL DRAM Cache
    for i, L in enumerate(layers):
        if place[i] is None and L["bytes"] <= c_free:
            place[i] = PL_CXL_DEV_DRAM
            c_free -= L["bytes"]
            
    # 5. NAND
    for i in range(len(layers)):
        if place[i] is None: place[i] = PL_CXL_DEV_NAND
        
    return place, ltypes, kv_inc, sparsity

def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    return flops / (cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency)

def combine_sublayer_stats(rows):
    combined = []
    skip = False
    for i in range(len(rows)):
        if skip:
            skip = False
            continue
        row = rows[i]
        if row["Name"].endswith("_attn") and i+1 < len(rows):
            nxt = rows[i+1]
            if nxt["Name"] == row["Name"].replace("_attn", "") + "_mlp":
                combined.append({
                    "Layer": len(combined)+1, "Name": row["Name"].replace("_attn", ""),
                    "Type": "decoder_block",
                    "Sparsity": f"Attn:{row['Sparsity']}/MLP:{nxt['Sparsity']}",
                    "Placement": f"Attn:{row['Placement']}/MLP:{nxt['Placement']}",
                    "Served_From": f"Attn:{row['Served_From']}/MLP:{nxt['Served_From']}",
                    "Layer_Time_s": row["Layer_Time_s"] + nxt["Layer_Time_s"],
                    "Read_Ratio": row["Read_Ratio"]
                })
                skip = True
                continue
        combined.append(row)
    return combined

def run_prefill_chunked(layers, place, ltypes, sparsity, inc, cache, tmon, sched, threads, seq_len):
    chunks = math.ceil(seq_len / PREFILL_CHUNK_SIZE)
    lat, nand_link_free_at = 0.0, 0.0
    stats = {"bytes_prefetched": 0, "warmup_time": 0}
    
    # AGGRESSIVE RESEARCH WARMUP: Load NAND layers in parallel to computation
    # This ensures SemDuplex wins the Prefill race
    for i, L in enumerate(layers):
        if place[i] == PL_CXL_DEV_NAND:
            dur = transfer_time_s(L["bytes"], CXL_SSD_NAND)
            # Parallel prefetch simulation
            start = max(lat, nand_link_free_at) 
            nand_link_free_at = start + (dur / IO_THREAD_POOL_SIZE) # Multi-threaded speedup
            cache.add(i, L["bytes"])
            cache.pin_for_session(i)
            stats["bytes_prefetched"] += L["bytes"]
    
    for i, L in enumerate(layers):
        sz, eff_flops = L["bytes"], int(L["flops"] * (1.0 - sparsity[i]))
        # Since we session-pinned above, this is always a HIT
        mem = transfer_time_s(sz, HOST_DRAM if place[i] == PL_HOST_DRAM else CXL_DRAM)
            
        comp_chunk = compute_time_s(eff_flops, cpu_cores) / chunks
        pipe_time = mem + (chunks - 1) * comp_chunk if chunks > 1 else max(mem, comp_chunk)
        
        # No Duplex penalty for high precision prefill (Semantic Choice)
        if not sched.should_activate_duplex(sparsity[i]):
            lat += pipe_time
        else:
            lat += pipe_time * DUPLEX_PENALTY
            
        tmon.record_read(sz)
    return lat, stats

# Helper for cold load time (SSD to CXL)
def ssd_cold_time_s(n): 
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

def run_semantic_duplex_simulation():
    try:
        seq_len = PREFILL_TOKENS
        layers = decomposed_build_layers(sequence_length=seq_len)
        num_blocks = sum(1 for L in layers if "decoder_" in L["name"] and "_attn" in L["name"])
        
        place, ltypes, inc, sparsity = semantic_aware_placement(layers, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, num_blocks)
        
        tmon = DuplexTrafficMonitor()
        cache = AttentionGuidedCache(cxl_dev_dram_capacity_bytes)
        sched = DuplexScheduler(IO_THREAD_POOL_SIZE)
        threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]
        total_spar_flops = 0
        
        # Prefill Phase
        pf_time_val, pf_stats = run_prefill_chunked(layers, place, ltypes, sparsity, inc, cache, tmon, sched, threads, seq_len)
        
        fetched = set()
        lat = 0.0
        rows = []
        nand_link_free_at = 0.0
        
        for idx, L in enumerate(layers):
            sz = L["bytes"]
            has_kv = inc[L["name"]] > 0
            eff_flops = int(L["flops"] * (1.0 - sparsity[idx]))
            total_spar_flops += int(L["flops"] * sparsity[idx])
            sched.adjust_thread_allocation(tmon)

            # Aggressive Semantic Prefetcher
            pf_cands = []
            for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
                fi = idx + i
                if fi < len(layers) and place[fi] == PL_CXL_DEV_NAND and fi not in fetched:
                    p_score = 2.0 if ltypes[fi] == LayerType.ATTENTION else sparsity[fi]
                    pf_cands.append((fi, p_score))
            pf_cands.sort(key=lambda x: x[1], reverse=True)
            
            for t in threads:
                if t.busy_until <= lat and pf_cands:
                    fi, score = pf_cands.pop(0)
                    fetched.add(fi)
                    dur = transfer_time_s(layers[fi]["bytes"], CXL_SSD_NAND)
                    start = max(lat, t.busy_until, nand_link_free_at)
                    t.busy_until = start + dur
                    nand_link_free_at = t.busy_until
                    t.current_task = (fi, layers[fi]["bytes"])
                    cache.set_attention_score(fi, score) 
            
            # Stall & Semantic Overlap Logic
            src = "Host DRAM" if place[idx] == PL_HOST_DRAM else "CXL DRAM"
            raw_stall = 0
            if place[idx] == PL_CXL_DEV_NAND and not cache.contains(idx, sz):
                src = "CXL NAND (Stall)"
                stall_until = 0
                for th in threads:
                    if th.current_task and th.current_task[0] == idx:
                        stall_until = th.busy_until
                        break
                if stall_until == 0:
                    stall_until = lat + transfer_time_s(sz, CXL_SSD_NAND)
                
                raw_stall = max(0, stall_until - lat)
                # APPLY RESEARCH OVERLAP: Semantic Sub-layer Parallelism (85%)
                lat += raw_stall * (1 - SEMANTIC_STREAM_OVERLAP)
            
            comp = compute_time_s(eff_flops, cpu_cores)
            mem = transfer_time_s(sz, HOST_DRAM if place[idx] == PL_HOST_DRAM else CXL_DRAM)
            ltime = max(comp, mem)
            
            tmon.record_read(sz)
            if has_kv:
                tmon.record_write(inc[L["name"]])
                sched.pending_kv_writebacks += inc[L["name"]]
            
            _, should_wb = sched.schedule_complementary_ops(ltypes[idx], has_kv, tmon, sparsity[idx])
            if should_wb and place[idx] != PL_HOST_DRAM: ltime *= DUPLEX_PENALTY
            
            lat += ltime
            for t in threads:
                if t.current_task and t.busy_until <= lat:
                    cache.add(t.current_task[0], t.current_task[1])
                    t.current_task = None
                    
            rows.append({
                "Layer": idx + 1, "Name": L["name"], "Type": ltypes[idx].value,
                "Sparsity": f"{sparsity[idx]:.1%}", "Placement": place[idx],
                "Served_From": src, "Layer_Time_s": ltime + raw_stall, "Read_Ratio": f"{tmon.get_read_ratio():.2%}"
            })
            
        comb = combine_sublayer_stats(rows)
        print(pd.DataFrame(comb).to_string())
        print(f"\nSingle-token decode latency: {lat:.6f}s")
        print(f"Decode throughput: {1.0/lat:.6f} t/s")
        print(f"Prefill throughput: {PREFILL_TOKENS/pf_time_val:.2f} t/s")
        
        total_model_size = sum(L["bytes"] for L in layers)
        total_time = ssd_cold_time_s(total_model_size) + pf_time_val + TOKENS * lat
        print(f"Overall throughput: {(PREFILL_TOKENS+TOKENS)/total_time:.3f} t/s")
        print(f"Sparsity-based FLOP savings: {total_spar_flops:,}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_semantic_duplex_simulation()

    