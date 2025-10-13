# filename: simulation_semantic_duplex.py
# ------------------------------------------------------------
# SIMULATOR: Semantic-Aware Duplex CXL Scheduler
#
# DESCRIPTION:
# This simulation exploits transformer layer semantics and CXL full-duplex
# architecture to maximize bandwidth utilization. It differentiates between
# attention layers (irregular access, KV cache updates) and MLP layers
# (sequential access, weight loading) to achieve balanced read-write traffic.
#
# KEY INNOVATIONS:
# 1. Layer-Type Classification: Identifies attention vs MLP layers
# 2. Duplex Traffic Balancing: Maintains 50-55% read ratio for optimal CXL bandwidth
# 3. Complementary Operation Injection: Overlaps reads with writes
# 4. Attention-Guided KV Cache Eviction: Uses attention scores instead of LRU
# ------------------------------------------------------------
import math
import pandas as pd
from collections import OrderedDict, deque
from enum import Enum


from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)


GiB = 1024**3


# Canonical labels
PL_HOST_DRAM        = "Host DRAM"
PL_CXL_DEV_DRAM     = "CXL Device DRAM"
PL_CXL_DEV_NAND     = "CXL Device NAND"


# ---------------------------
# Tunables
# ---------------------------
IO_THREAD_POOL_SIZE = 4
PREFETCH_QUEUE_DEPTH = 8
TRAFFIC_WINDOW_SIZE = 10


# ---------------------------
# PART 1: Layer Type Classification
# ---------------------------
class LayerType(Enum):
    """Semantic layer type classification for transformer components"""
    ATTENTION = "attention"      # Irregular access, KV cache read/write
    MLP = "mlp"                  # Sequential access, weight read-heavy
    NORM = "norm"                # Lightweight, read-only
    EMBEDDING = "embedding"      # Static lookup
    OUTPUT = "output"            # Final projection


def classify_layer_type(layer_dict):
    """
    Classify transformer layer by examining the layer dictionary structure.

    DecoderBlock layers contain both attention and MLP, so we treat them as
    hybrid layers that have BOTH characteristics:
    - Heavy KV cache updates (attention component)
    - Large weight matrices (MLP component)
    """
    name = layer_dict["name"]
    kind = layer_dict.get("kind", "")
    name_lower = name.lower()

    # Check the "kind" field first (more reliable)
    if kind == "DecoderBlock":
        # DecoderBlocks have both attention and MLP
        # For semantic scheduling, treat them as ATTENTION type
        # because they have KV cache (write operations)
        return LayerType.ATTENTION

    # Fallback to name-based detection
    if any(x in name_lower for x in ["attn", "attention", "self_attn"]):
        return LayerType.ATTENTION
    elif any(x in name_lower for x in ["mlp", "ffn", "feed_forward"]):
        return LayerType.MLP
    elif any(x in name_lower for x in ["norm", "layernorm", "rmsnorm"]):
        return LayerType.NORM
    elif "embed" in name_lower:
        return LayerType.EMBEDDING
    elif any(x in name_lower for x in ["lm_head", "output"]):
        return LayerType.OUTPUT
    else:
        return LayerType.NORM  # Default to lightweight


# ---------------------------
# PART 2: Duplex Traffic Monitor
# ---------------------------
class DuplexTrafficMonitor:
    """
    Monitors read/write traffic balance on CXL link.
    CXL full-duplex achieves 55-61% bandwidth gain at 50-55% read ratio.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.read_history = deque(maxlen=window_size)
        self.write_history = deque(maxlen=window_size)
        self.total_reads = 0
        self.total_writes = 0

    def record_read(self, bytes_read):
        """Track read operations (weight loading, KV cache reads)"""
        self.read_history.append(bytes_read)
        self.total_reads += bytes_read

    def record_write(self, bytes_written):
        """Track write operations (KV cache updates, writebacks)"""
        self.write_history.append(bytes_written)
        self.total_writes += bytes_written

    def get_read_ratio(self):
        """
        Calculate recent read ratio. Target: 50-55% for optimal duplex.
        Returns ratio in range [0, 1]
        """
        recent_reads = sum(self.read_history)
        recent_writes = sum(self.write_history)
        total = recent_reads + recent_writes

        if total == 0:
            return 0.5  # Neutral
        return recent_reads / total

    def needs_read_injection(self):
        """Check if traffic is write-heavy (needs more reads)"""
        ratio = self.get_read_ratio()
        return ratio < 0.45  # Below optimal range

    def needs_write_injection(self):
        """Check if traffic is read-heavy (needs more writes)"""
        ratio = self.get_read_ratio()
        return ratio > 0.60  # Above optimal range


# ---------------------------
# PART 3: Attention-Guided Cache Manager
# ---------------------------
class AttentionGuidedCache:
    """
    KV cache manager that uses attention scores instead of LRU.
    Evicts KV entries with low attention weights first.
    """
    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.used = 0
        self.cache = OrderedDict()  # layer_id -> bytes
        self.attention_scores = {}   # layer_id -> float (0-1)
        self.pinned = set()

    def set_attention_score(self, layer_id, score):
        """
        Set attention importance score for a layer.
        Higher score = more important to keep cached.
        """
        self.attention_scores[layer_id] = score

    def _evict_by_attention(self, needed_bytes):
        """
        Evict layers with lowest attention scores first.
        This is more intelligent than LRU for transformers.
        """
        if needed_bytes <= 0:
            return

        # Sort by attention score (lowest first)
        candidates = [(lid, sz, self.attention_scores.get(lid, 0.5)) 
                     for lid, sz in self.cache.items() 
                     if lid not in self.pinned]
        candidates.sort(key=lambda x: x[2])  # Sort by score

        freed = 0
        to_evict = []
        for lid, sz, score in candidates:
            to_evict.append(lid)
            freed += sz
            if freed >= needed_bytes:
                break

        for lid in to_evict:
            self.used -= self.cache.pop(lid)

    def add(self, layer_id, size_bytes):
        """Add layer to cache, evicting low-attention entries if needed"""
        if size_bytes > self.capacity:
            return False

        needed = max(0, (self.used + size_bytes) - self.capacity)
        if needed > 0:
            self._evict_by_attention(needed)

        if (self.used + size_bytes) <= self.capacity:
            self.used += size_bytes
            self.cache[layer_id] = size_bytes
            return True
        return False

    def contains(self, layer_id, required_size):
        """Check if layer is fully cached"""
        return self.cache.get(layer_id, 0) >= required_size

    @property
    def free_bytes(self):
        """Get available cache space"""
        return max(0, self.capacity - self.used)


# ---------------------------
# PART 4: Duplex Scheduler Core
# ---------------------------
class DuplexScheduler:
    """
    Core scheduling logic that balances read/write operations.

    Strategy:
    - DecoderBlock layers (have both attention + MLP): Inject complementary ops
    - Monitor KV cache updates and balance with weight prefetching
    """
    def __init__(self, io_pool_size=4):
        self.io_pool_size = io_pool_size
        self.read_threads = io_pool_size // 2
        self.write_threads = io_pool_size // 2
        self.pending_kv_writebacks = 0  # Accumulated KV cache to writeback

    def adjust_thread_allocation(self, traffic_monitor):
        """
        Dynamically allocate I/O threads based on traffic imbalance.
        More write threads when read-heavy, more read threads when write-heavy.
        """
        read_ratio = traffic_monitor.get_read_ratio()

        if read_ratio > 0.60:  # Too read-heavy
            self.read_threads = 1
            self.write_threads = self.io_pool_size - 1
        elif read_ratio < 0.45:  # Too write-heavy
            self.read_threads = self.io_pool_size - 1
            self.write_threads = 1
        else:  # Balanced
            self.read_threads = self.io_pool_size // 2
            self.write_threads = self.io_pool_size // 2

    def schedule_complementary_ops(self, current_layer_type, has_kv_cache, traffic_monitor):
        """
        Inject complementary operations to maintain duplex balance.

        Returns: (should_prefetch, should_writeback_kv)
        """
        # For layers with KV cache, they generate write traffic
        if has_kv_cache and current_layer_type == LayerType.ATTENTION:
            # This layer will update KV cache (write)
            # Check if we need to inject reads
            if traffic_monitor.needs_read_injection():
                return True, True   # Prefetch AND writeback
            return False, True      # Just writeback

        # For layers without KV cache, they're read-only
        else:
            # Check if we need to inject writes
            if traffic_monitor.needs_write_injection() and self.pending_kv_writebacks > 0:
                return True, True   # Prefetch AND writeback pending KV
            return True, False      # Just prefetch

    def accumulate_kv_writeback(self, kv_bytes):
        """Track KV cache that needs to be written back"""
        self.pending_kv_writebacks += kv_bytes

    def consume_kv_writeback(self, kv_bytes):
        """Mark KV cache as written back"""
        self.pending_kv_writebacks = max(0, self.pending_kv_writebacks - kv_bytes)


# ---------------------------
# PART 5: I/O Thread Pool
# ---------------------------
class IOThread:
    def __init__(self, thread_id):
        self.id = thread_id
        self.busy_until = 0.0
        self.current_task = None


# ---------------------------
# PART 6: Semantic-Aware Placement Strategy
# ---------------------------
def semantic_aware_placement(layers, host_cap, cxl_dram_cap, sequence_length):
    """
    Place layers based on their semantic type and access patterns.

    Strategy:
    - DecoderBlock layers (attention) → Host DRAM (latency-sensitive, have KV cache)
    - Other layers → Fill Host DRAM, then CXL DRAM, then NAND
    """
    placement = [None] * len(layers)
    layer_types = [classify_layer_type(L) for L in layers]

    # Calculate KV cache increments
    kv_cache_increment = {}
    for L in layers:
        if L["kind"] == "DecoderBlock":
            head_dim = L.get("head_dim", 128)
            kv_heads = L.get("kv_heads", 40 if len(layers) > 35 else 8)
            kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * BYTES_PER_PARAM
        else:
            kv_cache_increment[L["name"]] = 0

    host_free = host_cap
    cxl_free = cxl_dram_cap

    # Phase 1: Place all ATTENTION (DecoderBlock) layers in Host DRAM (priority)
    for idx, L in enumerate(layers):
        if layer_types[idx] == LayerType.ATTENTION:
            sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
            if sz <= host_free:
                placement[idx] = PL_HOST_DRAM
                host_free -= sz

    # Phase 2: Place critical output layers in Host DRAM
    for idx, L in enumerate(layers):
        if placement[idx] is None and layer_types[idx] == LayerType.OUTPUT:
            sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
            if sz <= host_free:
                placement[idx] = PL_HOST_DRAM
                host_free -= sz

    # Phase 3: Fill remaining Host DRAM with any remaining layers
    for idx, L in enumerate(layers):
        if placement[idx] is None:
            sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
            if sz <= host_free:
                placement[idx] = PL_HOST_DRAM
                host_free -= sz

    # Phase 4: Fill CXL DRAM with remaining layers
    for idx, L in enumerate(layers):
        if placement[idx] is None:
            sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
            if sz <= cxl_free:
                placement[idx] = PL_CXL_DEV_DRAM
                cxl_free -= sz

    # Phase 5: Spill to NAND
    for idx in range(len(layers)):
        if placement[idx] is None:
            placement[idx] = PL_CXL_DEV_NAND

    return placement, layer_types, kv_cache_increment


# ---------------------------
# PART 7: Helper Functions
# ---------------------------
def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s


def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))


# ---------------------------
# PART 8: Main Simulation Loop
# ---------------------------
def run_semantic_duplex_simulation():
    """Main simulation with semantic awareness and duplex scheduling"""

    # Build model
    sequence_length = 512
    layers = build_layers(sequence_length=sequence_length)

    # Semantic-aware placement
    placement, layer_types, kv_cache_increment = semantic_aware_placement(
        layers, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, sequence_length
    )

    # Initialize components
    traffic_monitor = DuplexTrafficMonitor(window_size=TRAFFIC_WINDOW_SIZE)
    cache_manager = AttentionGuidedCache(cxl_dev_dram_capacity_bytes)
    scheduler = DuplexScheduler(io_pool_size=IO_THREAD_POOL_SIZE)
    io_threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

    # Prefetch queue
    fetch_queue = deque()
    fetched_or_queued = set()

    # Simulation state
    per_token_latency = 0.0
    rows = []
    stats = {
        "compute_stall_s": 0,
        "bytes_prefetched": 0,
        "bytes_from_nand_miss": 0,
        "complementary_ops_injected": 0,
        "kv_writebacks_injected": 0
    }

    # Main loop: process each layer
    for exec_idx, L in enumerate(layers):
        sz = L["bytes"] + kv_cache_increment[L["name"]]
        layer_type = layer_types[exec_idx]
        has_kv_cache = kv_cache_increment[L["name"]] > 0

        # Adjust thread allocation based on traffic
        scheduler.adjust_thread_allocation(traffic_monitor)

        # Lookahead prefetching
        for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
            future_idx = exec_idx + i
            if future_idx < len(layers) and placement[future_idx] == PL_CXL_DEV_NAND and future_idx not in fetched_or_queued:
                fetch_queue.append(future_idx)
                fetched_or_queued.add(future_idx)

        # Assign fetch tasks to idle I/O threads
        for thread in io_threads:
            if thread.busy_until <= per_token_latency and fetch_queue:
                layer_to_fetch_idx = fetch_queue.popleft()
                layer_to_fetch = layers[layer_to_fetch_idx]
                fetch_size = layer_to_fetch["bytes"] + kv_cache_increment[layer_to_fetch["name"]]

                fetch_time = cxlssd_time_s(fetch_size)
                thread.busy_until = per_token_latency + fetch_time
                thread.current_task = (layer_to_fetch_idx, fetch_size)

        # Determine complementary operations
        should_prefetch, should_writeback = scheduler.schedule_complementary_ops(
            layer_type, has_kv_cache, traffic_monitor
        )

        # Simulate layer execution
        layer_time = 0.0
        served_from = ""

        if placement[exec_idx] == PL_HOST_DRAM:
            layer_time = max(compute_time_s(L["flops"], cpu_cores), dram_time_s(sz))
            served_from = "Host DRAM"
            traffic_monitor.record_read(sz)

        elif placement[exec_idx] == PL_CXL_DEV_DRAM:
            layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
            served_from = "CXL DRAM (resident)"
            traffic_monitor.record_read(sz)

        else:  # CXL NAND
            if cache_manager.contains(exec_idx, sz):
                layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
                served_from = "CXL DRAM (prefetched)"
                traffic_monitor.record_read(sz)
            else:
                # Cache miss - check if being fetched
                served_from = "CXL NAND (stall)"
                stall_until = float('inf')
                for thread in io_threads:
                    if thread.current_task and thread.current_task[0] == exec_idx:
                        stall_until = thread.busy_until
                        break

                if stall_until == float('inf'):
                    # Not queued, fetch immediately
                    idle_thread = min(io_threads, key=lambda th: th.busy_until)
                    fetch_time = cxlssd_time_s(sz)
                    stall_until = idle_thread.busy_until + fetch_time
                    stats["bytes_from_nand_miss"] += sz

                stall_time = max(0, stall_until - per_token_latency)
                stats["compute_stall_s"] += stall_time
                per_token_latency += stall_time

                layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
                traffic_monitor.record_read(sz)

        # Handle KV cache updates (writes) for layers with KV cache
        if has_kv_cache:
            kv_update_size = kv_cache_increment[L["name"]]
            traffic_monitor.record_write(kv_update_size)
            scheduler.accumulate_kv_writeback(kv_update_size)

        # Inject complementary writeback operations if needed
        if should_writeback and scheduler.pending_kv_writebacks > 0:
            # Writeback pending KV cache to balance read-heavy traffic
            writeback_size = min(scheduler.pending_kv_writebacks, sz)  # Match read size
            traffic_monitor.record_write(writeback_size)
            scheduler.consume_kv_writeback(writeback_size)
            stats["kv_writebacks_injected"] += 1
            stats["complementary_ops_injected"] += 1

        # Prefetch next layer if needed
        if should_prefetch and exec_idx + 1 < len(layers):
            next_idx = exec_idx + 1
            next_sz = layers[next_idx]["bytes"] + kv_cache_increment[layers[next_idx]["name"]]
            if placement[next_idx] == PL_CXL_DEV_NAND and not cache_manager.contains(next_idx, next_sz):
                if cache_manager.add(next_idx, next_sz):
                    pass  # Already tracked in I/O threads

        # Update attention scores (simulated - normally from attention weights)
        if layer_type == LayerType.ATTENTION:
            # Recent attention layers have higher importance
            score = 1.0 - (exec_idx / len(layers))
            cache_manager.set_attention_score(exec_idx, score)

        per_token_latency += layer_time

        # Complete I/O thread tasks
        for thread in io_threads:
            if thread.current_task and thread.busy_until <= per_token_latency:
                task_idx, task_bytes = thread.current_task
                if cache_manager.add(task_idx, task_bytes):
                    stats["bytes_prefetched"] += task_bytes
                thread.current_task = None

        rows.append({
            "Layer": exec_idx + 1,
            "Name": L["name"],
            "Type": layer_type.value,
            "Placement": placement[exec_idx],
            "Served_From": served_from,
            "Layer_Time_s": layer_time,
            "Read_Ratio": f"{traffic_monitor.get_read_ratio():.2%}"
        })

    # Output results
    df = pd.DataFrame(rows)
    df = df.reindex(columns=['Layer', 'Name', 'Type', 'Placement', 'Served_From', 'Layer_Time_s', 'Read_Ratio'])
    print(df.to_string())

    # Calculate summary statistics
    total_model_bytes = sum(L["bytes"] for L in layers)
    total_kv_cache_bytes = sum(L.get("kv_cache_bytes", 0) for L in layers)
    cold_load_s = ssd_cold_time_s(total_model_bytes)
    throughput = 1.0 / per_token_latency if per_token_latency > 0 else 0.0
    total_time_for_all_tokens = cold_load_s + (TOKENS * per_token_latency)

    model_dtype_bits = int(BYTES_PER_PARAM * 8)
    host_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) 
                     for i, p in enumerate(placement) if p == PL_HOST_DRAM)
    cxl_dram_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) 
                         for i, p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
    cxl_nand_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) 
                         for i, p in enumerate(placement) if p == PL_CXL_DEV_NAND)

    print(f"\n{'='*80}")
    print(f"Summary (Semantic-Aware Duplex CXL Scheduler):")
    print(f"{'='*80}")
    print(f"One-time cold SSD load: {cold_load_s:.6f} s")
    print(f"Single-token Latency: {per_token_latency:.6f} s")
    print(f"Estimated Tokens/sec: {throughput:.6f}")
    print(f"Total time for T={TOKENS}: {total_time_for_all_tokens:.6f} s")

    print(f"\nModel Size & Placement:")
    print(f"  Dtype: FP{model_dtype_bits}, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes/GiB:.1f} GiB")
    print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
    print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
    print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_bytes(host_bytes)})")
    print(f"  CXL Device DRAM: {cxl_dram_bytes:,} bytes ({fmt_bytes(cxl_dram_bytes)})")
    print(f"  CXL Device NAND: {cxl_nand_bytes:,} bytes ({fmt_bytes(cxl_nand_bytes)})")

    print(f"\nDuplex Optimization:")
    print(f"  Final Read Ratio: {traffic_monitor.get_read_ratio():.2%}")
    print(f"  Target Range: 50-55% (optimal for CXL full-duplex)")
    print(f"  Total Reads: {traffic_monitor.total_reads:,} bytes")
    print(f"  Total Writes: {traffic_monitor.total_writes:,} bytes")
    print(f"  Complementary Ops Injected: {stats['complementary_ops_injected']}")
    print(f"  KV Cache Writebacks: {stats['kv_writebacks_injected']}")

    print(f"\nSemantic Placement:")
    attn_in_host = sum(1 for i, t in enumerate(layer_types) if t == LayerType.ATTENTION and placement[i] == PL_HOST_DRAM)
    total_attn = sum(1 for t in layer_types if t == LayerType.ATTENTION)
    total_decoder = sum(1 for L in layers if L["kind"] == "DecoderBlock")
    print(f"  DecoderBlock layers (Attention+MLP) in Host DRAM: {attn_in_host}/{total_attn}")
    print(f"  Total DecoderBlock layers: {total_decoder}")

    print(f"\nPerformance Breakdown:")
    print(f"  Total compute stall time: {stats['compute_stall_s']:.6f} s")
    print(f"  Total bytes prefetched: {stats['bytes_prefetched']:,} bytes")
    print(f"  Total bytes read from NAND on miss: {stats['bytes_from_nand_miss']:,} bytes")

    return per_token_latency, stats


if __name__ == "__main__":
    run_semantic_duplex_simulation()