# filename: simulation_semantic_duplex.py
# ------------------------------------------------------------
# SIMULATOR: Semantic-Aware Duplex CXL Scheduler
#
# DESCRIPTION:
# This simulation exploits transformer layer semantics and CXL full-duplex
# architecture to maximize bandwidth utilization. It differentiates between
# attention (irregular) and MLP (sequential) layers to balance read/write traffic.
#
# OPTIMIZATION IMPROVEMENTS:
# 1. Non-blocking partial NAND fetch (40% overlap pipeline)
# 2. Increased I/O concurrency (12-thread pool)
# 3. Extended prefetch window to maintain continuous pipeline
# 4. Adaptive duplex tuning around optimal read ratio (50-55%)
# ------------------------------------------------------------

import math
import pandas as pd
from collections import OrderedDict, deque
from enum import Enum

from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM, decomposed_build_layers
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

GiB = 1024**3

# Canonical labels
PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

# Tunables
IO_THREAD_POOL_SIZE = 12
PREFETCH_QUEUE_DEPTH = 24
TRAFFIC_WINDOW_SIZE = 16

# Layer Types
class LayerType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    EMBEDDING = "embedding"
    OUTPUT = "output"

def classify_layer_type(layer_dict):
    name = layer_dict["name"].lower()
    kind = layer_dict.get("kind", "")

    # Classify based on name explicitly for decomposed sublayers
    if "attn" in name or "attention" in name or "self_attn" in name:
        return LayerType.ATTENTION
    elif "mlp" in name or "ffn" in name or "feed_forward" in name:
        return LayerType.MLP
    elif "norm" in name or "layernorm" in name or "rmsnorm" in name:
        return LayerType.NORM
    elif "embed" in name:
        return LayerType.EMBEDDING
    elif "lm_head" in name or "output" in name:
        return LayerType.OUTPUT
    else:
        # Fallback default
        return LayerType.NORM


# ---------------------------
# Duplex Traffic Monitor
# ---------------------------
class DuplexTrafficMonitor:
    def __init__(self, window_size=10):
        self.read_history = deque(maxlen=window_size)
        self.write_history = deque(maxlen=window_size)
        self.total_reads = 0
        self.total_writes = 0

    def record_read(self, bytes_read):
        self.read_history.append(bytes_read)
        self.total_reads += bytes_read

    def record_write(self, bytes_written):
        self.write_history.append(bytes_written)
        self.total_writes += bytes_written

    def get_read_ratio(self):
        reads, writes = sum(self.read_history), sum(self.write_history)
        total = reads + writes
        if total == 0:
            return 0.5
        return reads / total

    def needs_read_injection(self):
        return self.get_read_ratio() < 0.45

    def needs_write_injection(self):
        return self.get_read_ratio() > 0.6


# ---------------------------
# Attention-Guided Cache
# ---------------------------
class AttentionGuidedCache:
    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.used = 0
        self.cache = OrderedDict()
        self.attention_scores = {}
        self.pinned = set()

    def set_attention_score(self, lid, score):
        self.attention_scores[lid] = score
        if score > 0.3:
            self.pinned.add(lid)
        else:
            self.pinned.discard(lid)

    def _evict_by_attention(self, needed_bytes):
        if needed_bytes <= 0:
            return
        candidates = [(lid, sz, self.attention_scores.get(lid, 0.5))
                      for lid, sz in self.cache.items()
                      if lid not in self.pinned]
        candidates.sort(key=lambda x: x[2])
        freed = 0
        to_evict = []
        for lid, sz, score in candidates:
            to_evict.append(lid)
            freed += sz
            if freed >= needed_bytes:
                break
        for lid in to_evict:
            self.used -= self.cache.pop(lid)

    def add(self, lid, size_b):
        if size_b > self.capacity:
            return False
        need = max(0, self.used + size_b - self.capacity)
        if need > 0:
            self._evict_by_attention(need)
        if self.used + size_b <= self.capacity:
            self.used += size_b
            self.cache[lid] = size_b
            return True
        return False

    def contains(self, lid, req_size):
        return self.cache.get(lid, 0) >= req_size


# ---------------------------
# Duplex Scheduler
# ---------------------------
class DuplexScheduler:
    def __init__(self, io_pool_size=12):
        self.io_pool_size = io_pool_size
        self.read_threads = io_pool_size // 2
        self.write_threads = io_pool_size // 2
        self.pending_kv_writebacks = 0

    def adjust_thread_allocation(self, tmon):
        ratio = tmon.get_read_ratio()
        if ratio > 0.6:
            self.read_threads = 1
            self.write_threads = self.io_pool_size - 1
        elif ratio < 0.45:
            self.read_threads = self.io_pool_size - 1
            self.write_threads = 1
        else:
            self.read_threads = self.io_pool_size // 2
            self.write_threads = self.io_pool_size // 2

    def schedule_complementary_ops(self, layer_type, has_kv_cache, tmon):
        if has_kv_cache and layer_type == LayerType.ATTENTION:
            if tmon.needs_read_injection():
                return True, True
            return False, True
        else:
            if tmon.needs_write_injection() and self.pending_kv_writebacks > 0:
                return True, True
            return True, False


# ---------------------------
# I/O Thread
# ---------------------------
class IOThread:
    def __init__(self, tid):
        self.id = tid
        self.busy_until = 0.0
        self.current_task = None



# Update semantic_aware_placement to use new layer list w/ decomposed sublayers
def semantic_aware_placement(layers, host_cap, cxl_cap, seq_len):
    placement = [None] * len(layers)
    layer_types = [classify_layer_type(L) for L in layers]
    kv_inc = {}

    # Update: KV cache increments based on attention layers only
    for L in layers:
        if LayerType.ATTENTION == classify_layer_type(L):
            # Use typical head_dim and kv_heads if keys exist; fallback values otherwise
            hd = L.get("head_dim", 128)
            kh = L.get("kv_heads", 40 if len(layers) > 35 else 8)
            kv_inc[L["name"]] = 2 * kh * hd * BYTES_PER_PARAM
        else:
            kv_inc[L["name"]] = 0

    host_free = host_cap
    cxl_free = cxl_cap

    # Phase 1: ATTENTION layers in Host DRAM
    for i, L in enumerate(layers):
        if layer_types[i] == LayerType.ATTENTION:
            sz = L["bytes"] + kv_inc[L["name"]] * seq_len
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 2: OUTPUT layers in Host DRAM
    for i, L in enumerate(layers):
        if placement[i] is None and layer_types[i] == LayerType.OUTPUT:
            sz = L["bytes"] + kv_inc[L["name"]] * seq_len
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 3: EMBEDDING layers in Host DRAM
    for i, L in enumerate(layers):
        if placement[i] is None and layer_types[i] == LayerType.EMBEDDING:
            sz = L["bytes"] + kv_inc[L["name"]] * seq_len
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 4: Fill remaining Host DRAM with others (mostly MLP/NORM)
    for i, L in enumerate(layers):
        if placement[i] is None:
            sz = L["bytes"] + kv_inc[L["name"]] * seq_len
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 5: Place remaining in CXL DRAM
    for i, L in enumerate(layers):
        if placement[i] is None:
            sz = L["bytes"] + kv_inc[L["name"]] * seq_len
            if sz <= cxl_free:
                placement[i] = PL_CXL_DEV_DRAM
                cxl_free -= sz

    # Phase 6: Spill remainder to NAND
    for i in range(len(layers)):
        if placement[i] is None:
            placement[i] = PL_CXL_DEV_NAND

    return placement, layer_types, kv_inc


# ---------------------------
# Helper Timing Functions
# ---------------------------
def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0:
        return 0.0
    flops_per_sec = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_sec

def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n): return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def ssd_cold_time_s(n): return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))
def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"





def combine_sublayer_stats(rows):
    combined_rows = []
    skip_next = False

    for i in range(len(rows)):
        if skip_next:
            skip_next = False
            continue
        row = rows[i]
        if row["Name"].endswith("_attn") and i + 1 < len(rows):
            next_row = rows[i+1]
            prefix = row["Name"].replace("_attn", "")
            if next_row["Name"] == prefix + "_mlp":
                combined_row = {
                    "Layer": len(combined_rows) + 1,
                    "Name": prefix,
                    "Type": "decoder_block",
                    "Placement": f"Attn: {row['Placement']}, MLP: {next_row['Placement']}",
                    "Served_From": f"Attn: {row['Served_From']}, MLP: {next_row['Served_From']}",
                    "Layer_Time_s": row["Layer_Time_s"] + next_row["Layer_Time_s"],
                    "Read_Ratio": row["Read_Ratio"]
                }
                combined_rows.append(combined_row)
                skip_next = True
                continue
        combined_rows.append(row)
    return combined_rows




def run_semantic_duplex_simulation():
    seq_len = 512
    layers = decomposed_build_layers(sequence_length=seq_len)
    place, ltypes, inc = semantic_aware_placement(layers, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, seq_len)
    tmon = DuplexTrafficMonitor(window_size=TRAFFIC_WINDOW_SIZE)
    cache = AttentionGuidedCache(cxl_dev_dram_capacity_bytes)
    sched = DuplexScheduler(io_pool_size=IO_THREAD_POOL_SIZE)
    threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]
    queue, fetched = deque(), set()
    latency = 0.0

    stats = {"compute_stall_s": 0, "bytes_prefetched": 0, "bytes_from_nand_miss": 0,
             "complementary_ops_injected": 0, "kv_writebacks_injected": 0}
    rows = []

    for idx, L in enumerate(layers):
        sz = L["bytes"] + inc[L["name"]] * seq_len
        layer_type = ltypes[idx]
        has_kv = inc[L["name"]] > 0
        sched.adjust_thread_allocation(tmon)

        for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
            fidx = idx + i
            if fidx < len(layers) and place[fidx] == PL_CXL_DEV_NAND and fidx not in fetched:
                queue.append(fidx)
                fetched.add(fidx)

        for t in threads:
            if t.busy_until <= latency and queue:
                l2f = queue.popleft()
                Lf = layers[l2f]
                fsize = Lf["bytes"] + inc[Lf["name"]] * seq_len
                ftime = cxlssd_time_s(fsize)
                t.busy_until = latency + ftime
                t.current_task = (l2f, fsize)

        should_prefetch, should_wb = sched.schedule_complementary_ops(layer_type, has_kv, tmon)

        # Compute or stall
        if place[idx] == PL_HOST_DRAM:
            ltime = max(compute_time_s(L["flops"], cpu_cores), dram_time_s(sz))
            src = "Host DRAM"
        elif place[idx] == PL_CXL_DEV_DRAM:
            ltime = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
            src = "CXL DRAM"
        else:
            if cache.contains(idx, sz):
                ltime = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
                src = "CXL DRAM (prefetched)"
            else:
                src = "CXL NAND (partial overlap)"
                stall_until = float('inf')
                for t in threads:
                    if t.current_task and t.current_task[0] == idx:
                        stall_until = t.busy_until
                        break
                if stall_until == float('inf'):
                    thr = min(threads, key=lambda th: th.busy_until)
                    stall_until = thr.busy_until + cxlssd_time_s(sz)
                    stats["bytes_from_nand_miss"] += sz

                overlap_factor = 0.45
                raw_stall = max(0, stall_until - latency)
                effective_stall = raw_stall * (1 - overlap_factor)
                stats["compute_stall_s"] += effective_stall
                latency += effective_stall
                ltime = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))

        # Record reads/writes
        tmon.record_read(sz)
        if has_kv:
            kv_sz = inc[L["name"]] * seq_len
            tmon.record_write(kv_sz)
            sched.pending_kv_writebacks += kv_sz

        if should_wb:
            r, w = tmon.total_reads, tmon.total_writes
            target = 0.52
            needed_total = (r / target) - r
            need_extra = max(0, needed_total - w)
            wb_size = min(need_extra, sz * 1.0)
            if wb_size > 0:
                tmon.record_write(wb_size)
                stats["kv_writebacks_injected"] += 1
                stats["complementary_ops_injected"] += 1

        if should_prefetch and idx + 1 < len(layers):
            nxt = idx + 1
            nxt_sz = layers[nxt]["bytes"] + inc[layers[nxt]["name"]] * seq_len
            if place[nxt] == PL_CXL_DEV_NAND and not cache.contains(nxt, nxt_sz):
                cache.add(nxt, nxt_sz)

        if layer_type == LayerType.ATTENTION:
            cache.set_attention_score(idx, 1.0 - (idx / len(layers)))

        latency += ltime
        for t in threads:
            if t.current_task and t.busy_until <= latency:
                tidx, tb = t.current_task
                if cache.add(tidx, tb):
                    stats["bytes_prefetched"] += tb
                t.current_task = None

        rows.append({
            "Layer": idx + 1, "Name": L["name"], "Type": layer_type.value,
            "Placement": place[idx], "Served_From": src,
            "Layer_Time_s": ltime, "Read_Ratio": f"{tmon.get_read_ratio():.2%}"
        })

    combined_rows = combine_sublayer_stats(rows)
    df = pd.DataFrame(combined_rows)
    print(df.to_string())

    total_model_bytes = sum(L["bytes"] for L in layers)
    cold_load_s = ssd_cold_time_s(total_model_bytes)
    throughput = 1.0 / latency if latency > 0 else 0.0
    total_time = cold_load_s + TOKENS * latency

    # Calculate model stats
    total_kv_cache_bytes = sum(inc[L["name"]] * seq_len for L in layers)
    per_token_kv_update_bytes = total_kv_cache_bytes / seq_len

    host_layers_count = sum(1 for i, p in enumerate(place) if p == PL_HOST_DRAM)
    cxl_dram_layers_count = sum(1 for i, p in enumerate(place) if p == PL_CXL_DEV_DRAM)
    cxl_nand_layers_count = sum(1 for i, p in enumerate(place) if p == PL_CXL_DEV_NAND)

    host_bytes = sum(layers[i]["bytes"] + inc[layers[i]["name"]] * seq_len for i, p in enumerate(place) if p == PL_HOST_DRAM)
    cxl_dram_bytes = sum(layers[i]["bytes"] + inc[layers[i]["name"]] * seq_len for i, p in enumerate(place) if p == PL_CXL_DEV_DRAM)
    cxl_nand_bytes = sum(layers[i]["bytes"] + inc[layers[i]["name"]] * seq_len for i, p in enumerate(place) if p == PL_CXL_DEV_NAND)

    mlp_layers_count = sum(1 for i, t in enumerate(ltypes) if t == LayerType.MLP)
    mlp_in_host = sum(1 for i, t in enumerate(ltypes) if t == LayerType.MLP and place[i] == PL_HOST_DRAM)
    mlp_in_cxl = sum(1 for i, t in enumerate(ltypes) if t == LayerType.MLP and place[i] == PL_CXL_DEV_DRAM)
    mlp_in_nand = sum(1 for i, t in enumerate(ltypes) if t == LayerType.MLP and place[i] == PL_CXL_DEV_NAND)

    print(f"\n{'='*80}")
    print("Summary (Semantic-Aware Duplex CXL Scheduler with Partial Overlap):")
    print(f"{'='*80}")
    print(f"Cold-load time: {cold_load_s:.6f}s")
    print(f"Single-token latency: {latency:.6f}s")
    print(f"Tokens/sec: {throughput:.6f}")
    print(f"Total time (T={TOKENS}): {total_time:.6f}s")

    print("\nModel Size & Placement:")
    print(f"  Dtype: FP32, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes / GiB:.1f} GiB")
    print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
    print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
    print(f"  Per-token KV cache update: {per_token_kv_update_bytes:,} bytes ({fmt_bytes(per_token_kv_update_bytes)})")
    print(f"  Host DRAM layers: {host_layers_count}")
    print(f"  CXL DRAM layers: {cxl_dram_layers_count}")
    print(f"  CXL NAND layers: {cxl_nand_layers_count}")
    print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_bytes(host_bytes)})")
    print(f"  CXL Device DRAM: {cxl_dram_bytes:,} bytes ({fmt_bytes(cxl_dram_bytes)})")
    print(f"  CXL Device NAND: {cxl_nand_bytes:,} bytes ({fmt_bytes(cxl_nand_bytes)})")

    print(f"MLP layers: {mlp_layers_count} total")
    print(f"  MLP in Host DRAM: {mlp_in_host}")
    print(f"  MLP in CXL DRAM: {mlp_in_cxl}")
    print(f"  MLP in CXL NAND: {mlp_in_nand}")

    print(f"\nCompute stalls: {stats['compute_stall_s']:.6f}s")
    print(f"Bytes prefetched: {stats['bytes_prefetched']:,}")
    print(f"Bytes from NAND miss: {stats['bytes_from_nand_miss']:,}")
    print(f"Final Duplex read ratio: {tmon.get_read_ratio():.2%}")
    print(f"Complementary ops injected: {stats['complementary_ops_injected']}")
    print(f"KV writebacks injected: {stats['kv_writebacks_injected']}")

if __name__ == "__main__":
    run_semantic_duplex_simulation()