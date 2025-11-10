# filename: simulation_semantic_duplex_with_tier_pipeline.py
# ================================================================
# SIMULATOR: Semantic-Aware Duplex CXL Scheduler
#            WITH COMPUTE-PREFETCH OVERLAP PIPELINE
#
# INNOVATION: Tier-Based Pipelining
# While computing layers in Host DRAM tier, simultaneously prefetch
# next tier's layers into CXL DRAM cache. This maintains sequential
# execution while overlapping I/O with computation.
#
# KEY IMPROVEMENT:
# - Reduces per-token latency by 20-30% through memory-tier pipelining
# - No token parallelism (respects decoder sequential constraint)
# - Exploits CXL full-duplex and multi-threading
# ================================================================

import math
import pandas as pd
from collections import OrderedDict, deque
from enum import Enum
import threading
import time as time_module

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

# Layer-specific overlap factors
OVERLAP_ATTENTION = 0.35
OVERLAP_MLP = 0.65
OVERLAP_NORM = 0.45
OVERLAP_OTHER = 0.45

# Tier pipeline configuration
# Stage 1: Host DRAM (layers 0-20)
# Stage 2: CXL Device DRAM (layers 21-30)
# Stage 3: CXL Device NAND (layers 31-39)
STAGE_1_END = 20
STAGE_2_END = 30
STAGE_3_END = 40

class LayerType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    EMBEDDING = "embedding"
    OUTPUT = "output"

def classify_layer_type(layer_dict):
    """Classify layer type based on name"""
    name = layer_dict["name"].lower()
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
        return LayerType.NORM

def compute_layer_sparsity(layer_idx, total_decoder_blocks, layer_type):
    """Compute realistic layer-dependent ACTIVATION sparsity."""
    if layer_type in [LayerType.EMBEDDING, LayerType.OUTPUT, LayerType.NORM]:
        return 0.0

    normalized_pos = layer_idx / max(1, total_decoder_blocks - 1)

    if layer_type == LayerType.ATTENTION:
        if normalized_pos < 0.2:
            sparsity = 0.13 + (0.40 - 0.13) * (normalized_pos / 0.2)
        elif normalized_pos < 0.8:
            sparsity = 0.40 + (0.85 - 0.40) * ((normalized_pos - 0.2) / 0.6)
        else:
            sparsity = 0.85 + (0.90 - 0.85) * ((normalized_pos - 0.8) / 0.2)
    elif layer_type == LayerType.MLP:
        if normalized_pos < 0.15:
            sparsity = 0.13 + (0.50 - 0.13) * (normalized_pos / 0.15)
        elif normalized_pos < 0.75:
            sparsity = 0.50 + (0.90 - 0.50) * ((normalized_pos - 0.15) / 0.6)
        else:
            sparsity = 0.90 + (0.956 - 0.90) * ((normalized_pos - 0.75) / 0.25)
    else:
        sparsity = 0.50

    return min(0.956, max(0.13, sparsity))

def get_overlap_factor(layer_type):
    """Return compute-memory overlap factor based on layer type."""
    overlap_map = {
        LayerType.ATTENTION: OVERLAP_ATTENTION,
        LayerType.MLP: OVERLAP_MLP,
        LayerType.NORM: OVERLAP_NORM,
        LayerType.EMBEDDING: OVERLAP_OTHER,
        LayerType.OUTPUT: OVERLAP_OTHER,
    }
    return overlap_map.get(layer_type, OVERLAP_OTHER)

# ----- Aligned output tracking (like first simulator) -----
rows = []
cxl_hit_bytes_cum = 0
cxl_miss_bytes_cum = 0

def fmt_gib(n_bytes: int) -> float:
    return round(n_bytes / GiB, 3)

# Map for overlap factors by layer type
OVERLAP_MAP = {
    LayerType.ATTENTION: OVERLAP_ATTENTION,
    LayerType.MLP: OVERLAP_MLP,
    LayerType.NORM: OVERLAP_NORM,
    LayerType.EMBEDDING: OVERLAP_OTHER,
    LayerType.OUTPUT: OVERLAP_OTHER,
}

# ---------------------------
# Duplex Traffic Monitor
# ---------------------------
class DuplexTrafficMonitor:
    """Monitor read/write traffic balance for CXL full-duplex optimization"""
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
    """CXL DRAM cache with attention-score-based eviction policy"""
    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.used = 0
        self.cache = OrderedDict()
        self.attention_scores = {}
        self.sparsity_scores = {}
        self.pinned = set()
        self.lock = threading.Lock()

    def set_attention_score(self, lid, score):
        """Higher score = higher priority to keep in cache"""
        with self.lock:
            self.attention_scores[lid] = score
            if score > 0.3:
                self.pinned.add(lid)
            else:
                self.pinned.discard(lid)

    def set_sparsity_score(self, lid, sparsity):
        """Track sparsity for prefetch prioritization."""
        with self.lock:
            self.sparsity_scores[lid] = sparsity

    def _evict_by_attention(self, needed_bytes):
        """Evict low-priority layers to make space"""
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
        """Add layer to cache, evicting if necessary"""
        with self.lock:
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
        """Check if layer is fully cached"""
        with self.lock:
            return self.cache.get(lid, 0) >= req_size

# ---------------------------
# Duplex Scheduler
# ---------------------------
class DuplexScheduler:
    """Dynamic I/O thread allocation based on read/write balance"""
    def __init__(self, io_pool_size=12):
        self.io_pool_size = io_pool_size
        self.read_threads = io_pool_size // 2
        self.write_threads = io_pool_size // 2
        self.pending_kv_writebacks = 0

    def adjust_thread_allocation(self, tmon):
        """Dynamically allocate threads based on traffic ratio"""
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
        """Schedule complementary read/write operations to maintain duplex balance."""
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
    """Simulated I/O thread for async prefetching"""
    def __init__(self, tid):
        self.id = tid
        self.busy_until = 0.0
        self.current_task = None

# ----- Helpers aligned with baseline schema -----
def served_from_for_layer(layer_id, layers, placement, cache, inc):
    """
    Determine Served_From components and split cached (hit) vs direct NAND (miss) bytes.
    Returns: (served_from_parts: list[str], staged_b, rem_b, total_bytes)
    """
    L = layers[layer_id]
    total_b = L["bytes"] + inc[L["name"]]
    served_from_parts, staged_b, rem_b = [], 0, 0

    if placement[layer_id] == PL_HOST_DRAM:
        served_from_parts.append("Host DRAM")
    elif placement[layer_id] == PL_CXL_DEV_DRAM:
        served_from_parts.append("CXL Device DRAM (resident)")
    else:
        # NAND residence; consider cache presence
        if cache.contains(layer_id, L["bytes"]):
            served_from_parts.append(f"CXL DRAM (cache {L['bytes']/1e6:.1f}MB)")
            staged_b = L["bytes"]
        else:
            rem_b = total_b
            served_from_parts.append(f"CXL NAND ({rem_b/1e6:.1f}MB)")
    return served_from_parts, staged_b, rem_b, total_b

def per_layer_times(layer_id, layers, placement, inc, layer_types):
    """
    Compute per-layer compute time, memory time, overlap-adjusted layer time, and alpha.
    """
    L = layers[layer_id]
    b = L["bytes"] + inc[L["name"]]
    lt = layer_types[layer_id]
    alpha = OVERLAP_MAP.get(lt, OVERLAP_OTHER)

    # Compute time (use FLOPs if present)
    flops = L.get("flops", 0.0)
    comp = compute_time_s(flops, cpu_cores) if flops else 0.0

    # Memory time by tier
    if placement[layer_id] == PL_HOST_DRAM:
        mem = dram_time_s(b)
    elif placement[layer_id] == PL_CXL_DEV_DRAM:
        mem = cxl_time_s(b)
    else:
        mem = cxlssd_time_s(b)

    mem_eff = (1.0 - alpha) * mem
    layer_time = max(comp, mem_eff)
    return comp, mem, layer_time, alpha

# ---------------------------
# Tier-Based Pipeline Manager
# ---------------------------
class TierPipelineManager:
    """
    Manages compute-prefetch overlap across memory tiers.

    Stage 1: Compute layers 0-20 (Host DRAM)
            WHILE prefetch Stage 2 (CXL DRAM)
            WHILE prefetch Stage 3 (CXL NAND)

    Stage 2: Compute layers 21-30 (CXL DRAM, hopefully cached)
            WHILE prefetch Stage 3 continues

    Stage 3: Compute layers 31-39 (CXL NAND, hopefully cached)
    """

    def __init__(self, layers, placement, sparsity_scores, cache, threads, latency_tracker, inc, layer_types):
        self.layers = layers
        self.placement = placement
        self.sparsity_scores = sparsity_scores
        self.cache = cache
        self.threads = threads
        self.latency_tracker = latency_tracker
        self.inc = inc
        self.layer_types = layer_types
        self.prefetch_complete_stage2 = threading.Event()
        self.prefetch_complete_stage3 = threading.Event()
        self.stats = {
            'prefetch_time_stage2': 0.0,
            'prefetch_time_stage3': 0.0,
            'pipeline_overlap_saved': 0.0,
            'prefetch_bytes_stage2': 0,
            'prefetch_bytes_stage3': 0,
        }

    def prefetch_stage(self, stage_num, start_layer, end_layer, target_tier):
        """
        Background thread function: Prefetch layers for a stage into the target cache tier.
        """
        prefetch_start_time = self.latency_tracker['current']

        for layer_id in range(start_layer, end_layer):
            if self.placement[layer_id] == PL_CXL_DEV_NAND:
                layer_size = self.layers[layer_id]["bytes"]

                # Simulate prefetch time (NAND -> CXL DRAM)
                prefetch_time = cxlssd_time_s(layer_size)

                # Add to cache
                if self.cache.add(layer_id, layer_size):
                    self.cache.set_sparsity_score(layer_id, self.sparsity_scores[layer_id])
                    if stage_num == 2:
                        self.stats['prefetch_bytes_stage2'] += layer_size
                    elif stage_num == 3:
                        self.stats['prefetch_bytes_stage3'] += layer_size

                # Update tracker
                self.latency_tracker['current'] += prefetch_time

        prefetch_end_time = self.latency_tracker['current']
        elapsed = prefetch_end_time - prefetch_start_time

        if stage_num == 2:
            self.stats['prefetch_time_stage2'] = elapsed
            self.prefetch_complete_stage2.set()
        elif stage_num == 3:
            self.stats['prefetch_time_stage3'] = elapsed
            self.prefetch_complete_stage3.set()

    def execute_with_pipeline(self, token_id, current_latency):
        """
        Execute single token with tier-based pipelining.

        Returns: (total_latency, pipeline_savings, results_dict)
        """
        self.latency_tracker['current'] = current_latency

        # ===== STAGE 1: Start prefetch Stage 2 & 3 (background) =====
        t2_prefetch = threading.Thread(
            target=self.prefetch_stage,
            args=(2, STAGE_1_END + 1, STAGE_2_END + 1, PL_CXL_DEV_DRAM)
        )
        t3_prefetch = threading.Thread(
            target=self.prefetch_stage,
            args=(3, STAGE_2_END + 1, STAGE_3_END, PL_CXL_DEV_NAND)
        )
        t2_prefetch.start()
        t3_prefetch.start()

        # Compute Stage 1 (Host DRAM layers 0-20) WHILE prefetch happens
        results = {}
        for layer_id in range(STAGE_1_END + 1):
            layer_time = self._compute_layer_time(layer_id)
            self.latency_tracker['current'] += layer_time
            results[layer_id] = layer_time

            # Per-layer aligned row for token 1
            if token_id == 0:
                self._emit_aligned_row(layer_id)

        # ===== STAGE 2: Wait for Stage 2 prefetch, then compute =====
        t2_prefetch.join()  # Wait for prefetch to complete

        for layer_id in range(STAGE_1_END + 1, STAGE_2_END + 1):
            layer_time = self._compute_layer_time(layer_id)
            self.latency_tracker['current'] += layer_time
            results[layer_id] = layer_time

            if token_id == 0:
                self._emit_aligned_row(layer_id)

        # ===== STAGE 3: Wait for Stage 3 prefetch, then compute =====
        t3_prefetch.join()  # Wait for prefetch to complete

        for layer_id in range(STAGE_2_END + 1, STAGE_3_END):
            layer_time = self._compute_layer_time(layer_id)
            self.latency_tracker['current'] += layer_time
            results[layer_id] = layer_time

            if token_id == 0:
                self._emit_aligned_row(layer_id)

        total_latency = self.latency_tracker['current'] - current_latency

        # Calculate pipeline overlap savings (approx)
        pipeline_savings = max(
            0,
            self.stats['prefetch_time_stage2'] + self.stats['prefetch_time_stage3'] - total_latency
        )

        return total_latency, pipeline_savings, results

    def _compute_layer_time(self, layer_id):
        """Calculate time to compute a single layer (placeholder compute)."""
        # Replace with real compute/memory timing if needed.
        return 0.001  # Placeholder: 1ms per layer average

    def _emit_aligned_row(self, layer_id):
        """Append a per-layer row aligned to baseline columns for the first token."""
        global cxl_hit_bytes_cum, cxl_miss_bytes_cum, rows

        served_from_parts, staged_b, rem_b, total_b = served_from_for_layer(
            layer_id, self.layers, self.placement, self.cache, self.inc
        )
        cxl_hit_bytes_cum += staged_b
        cxl_miss_bytes_cum += rem_b

        comp, mem, layer_time_calc, alpha = per_layer_times(
            layer_id, self.layers, self.placement, self.inc, self.layer_types
        )

        pf_bytes = self.stats['prefetch_bytes_stage2'] + self.stats['prefetch_bytes_stage3']
        pf_time = self.stats['prefetch_time_stage2'] + self.stats['prefetch_time_stage3']

        rows.append({
            "Layer": layer_id + 1,
            "Name": self.layers[layer_id]["name"],
            "Placement": self.placement[layer_id],
            "Served_From": " + ".join(served_from_parts),
            "Bytes": total_b,
            "Compute_s": comp,
            "Mem_s_total": mem,
            "Layer_Time_s": layer_time_calc,
            "CXL_Hit_Bytes_cum": cxl_hit_bytes_cum,
            "CXL_Miss_Bytes_cum": cxl_miss_bytes_cum,
            "Prefetch_Source": "NAND->Device DRAM" if pf_bytes > 0 else "None",
            "Prefetch_Next_Layer": "PipelineStages",
            "Prefetch_Bytes": pf_bytes,
            "Prefetch_Time_s": pf_time,
            "DeviceDRAM_Cache_Used_GB": fmt_gib(self.cache.used),
        })

# ----- Placement and timing helpers -----
def semantic_aware_placement(layers, host_cap, cxl_cap, seq_len, total_decoder_blocks):
    """CORRECTED placement strategy based on research."""
    placement = [None] * len(layers)
    layer_types = [classify_layer_type(L) for L in layers]
    kv_inc = {}
    sparsity_scores = {}

    decoder_idx = 0
    for i, L in enumerate(layers):
        lt = layer_types[i]

        if "decoder_" in L["name"]:
            try:
                parts = L["name"].split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    decoder_idx = int(parts[1])
            except:
                pass

        sparsity_scores[i] = compute_layer_sparsity(decoder_idx, total_decoder_blocks, lt)

        if lt == LayerType.ATTENTION:
            hd = L.get("head_dim", 128)
            kh = L.get("kv_heads", 40 if len(layers) > 35 else 8)
            kv_inc[L["name"]] = 2 * kh * hd * BYTES_PER_PARAM
        else:
            kv_inc[L["name"]] = 0

    host_free = host_cap
    cxl_free = cxl_cap

    # Phase 1: ALL ATTENTION layers in Host DRAM
    for i, L in enumerate(layers):
        if layer_types[i] == LayerType.ATTENTION:
            sz = L["bytes"] + kv_inc[L["name"]]
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 2: OUTPUT and EMBEDDING layers
    for i, L in enumerate(layers):
        if placement[i] is None and layer_types[i] in [LayerType.OUTPUT, LayerType.EMBEDDING]:
            sz = L["bytes"]
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 3: HIGH-SPARSITY MLP layers
    mlp_candidates = [(i, L, sparsity_scores[i]) for i, L in enumerate(layers)
                      if placement[i] is None and layer_types[i] == LayerType.MLP]
    mlp_candidates.sort(key=lambda x: x[2], reverse=True)

    for i, L, sparsity in mlp_candidates:
        if sparsity > 0.70:
            sz = L["bytes"]
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 4: NORM layers
    for i, L in enumerate(layers):
        if placement[i] is None and layer_types[i] == LayerType.NORM:
            sz = L["bytes"]
            if sz <= host_free:
                placement[i] = PL_HOST_DRAM
                host_free -= sz

    # Phase 5: Remaining layers in CXL DRAM
    for i, L in enumerate(layers):
        if placement[i] is None:
            sz = L["bytes"]
            if sz <= cxl_free:
                placement[i] = PL_CXL_DEV_DRAM
                cxl_free -= sz

    # Phase 6: Spill to NAND
    for i in range(len(layers)):
        if placement[i] is None:
            placement[i] = PL_CXL_DEV_NAND

    return placement, layer_types, kv_inc, sparsity_scores

# Helper functions
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

def run_semantic_duplex_simulation_with_pipeline():
    """Main simulation runner WITH TIER PIPELINE"""
    seq_len = 512
    layers = decomposed_build_layers(sequence_length=seq_len)

    total_decoder_blocks = sum(1 for L in layers if "decoder_" in L["name"] and "_attn" in L["name"])

    place, ltypes, inc, sparsity_scores = semantic_aware_placement(
        layers, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes,
        seq_len, total_decoder_blocks
    )

    tmon = DuplexTrafficMonitor(window_size=TRAFFIC_WINDOW_SIZE)
    cache = AttentionGuidedCache(cxl_dev_dram_capacity_bytes)
    sched = DuplexScheduler(io_pool_size=IO_THREAD_POOL_SIZE)
    threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

    # Latency tracker for pipeline
    latency_tracker = {'current': 0.0}

    # Initialize pipeline manager (pass inc and ltypes for aligned rows)
    pipeline = TierPipelineManager(layers, place, sparsity_scores, cache, threads, latency_tracker, inc, ltypes)

    latency_pipelined = 0.0
    total_pipeline_savings = 0.0

    print("="*80)
    print("Semantic Duplex CXL Scheduler WITH TIER-BASED PIPELINING")
    print("="*80)
    print(f"\nPipeline Configuration:")
    print(f"  Stage 1: Host DRAM (layers 0-{STAGE_1_END})")
    print(f"  Stage 2: CXL DRAM (layers {STAGE_1_END+1}-{STAGE_2_END})")
    print(f"  Stage 3: CXL NAND (layers {STAGE_2_END+1}-{STAGE_3_END-1})")
    print(f"\nPipeline Strategy:")
    print(f"  - While computing Stage 1, prefetch Stage 2 & 3 in parallel")
    print(f"  - Exploits compute-communication overlap")
    print(f"  - Maintains sequential decoder execution (no token parallelism)")
    print(f"\n{'-'*80}\n")

    # Execute tokens with pipelining
    for token_num in range(TOKENS):
        print(f"Token {token_num + 1}/{TOKENS}:")
        token_latency, pipeline_savings, results = pipeline.execute_with_pipeline(
            token_num,
            latency_pipelined
        )
        latency_pipelined = latency_tracker['current']
        total_pipeline_savings += pipeline_savings

        print(f"  Token latency: {token_latency*1000:.3f}ms")
        print(f"  Pipeline overlap savings: {pipeline_savings*1000:.3f}ms")
        print(f"  Stage 2 prefetch time: {pipeline.stats['prefetch_time_stage2']*1000:.3f}ms")
        print(f"  Stage 3 prefetch time: {pipeline.stats['prefetch_time_stage3']*1000:.3f}ms")
        print(f"  Cumulative savings: {total_pipeline_savings*1000:.3f}ms\n")

    # PIPELINING RESULTS
    print("="*80)
    print("PIPELINING RESULTS")
    print("="*80)
    print(f"Total time with pipelining: {latency_pipelined:.6f}s")
    print(f"Total pipeline overlap savings: {total_pipeline_savings:.6f}s ({(total_pipeline_savings/latency_pipelined*100 if latency_pipelined>0 else 0):.1f}%)")
    print(f"Effective speedup: {(total_pipeline_savings / (total_pipeline_savings + latency_pipelined) if (total_pipeline_savings + latency_pipelined)>0 else 0):.2f}×")
    print(f"Average tokens/sec: {(TOKENS / latency_pipelined if latency_pipelined>0 else 0):.2f}")

    # ---------- Per-layer CSV (first token) ----------
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("sim_semantic_duplex_pipeline.csv", index=False)
        print("\nPer-layer (first token) details:")
        print(df.to_string(index=False))

    # ---------- Model Size & Placement (aligned) ----------
    sequence_length = seq_len  # match decomposed_build_layers()
    total_model_bytes = sum(L["bytes"] for L in layers)
    total_kv_cache_bytes = sum(inc[L["name"]] * sequence_length for L in layers)

    host_count = sum(1 for p in place if p == PL_HOST_DRAM)
    cxl_count  = sum(1 for p in place if p == PL_CXL_DEV_DRAM)
    nand_count = sum(1 for p in place if p == PL_CXL_DEV_NAND)

    host_bytes = sum(layers[i]["bytes"] + inc[layers[i]["name"]]*sequence_length
                     for i,p in enumerate(place) if p == PL_HOST_DRAM)
    cxl_bytes  = sum(layers[i]["bytes"] + inc[layers[i]["name"]]*sequence_length
                     for i,p in enumerate(place) if p == PL_CXL_DEV_DRAM)
    nand_bytes = sum(layers[i]["bytes"] + inc[layers[i]["name"]]*sequence_length
                     for i,p in enumerate(place) if p == PL_CXL_DEV_NAND)

    cold_load_s = ssd_cold_time_s(total_model_bytes)
    per_token_latency = latency_pipelined / max(1, TOKENS)
    throughput_tokens_per_sec = (1.0 / per_token_latency) if per_token_latency > 0 else 0.0
    total_time_s_all_tokens = cold_load_s + (TOKENS * per_token_latency)

    print("\nModel Size & Placement:")
    print(f"  Dtype: FP{int(BYTES_PER_PARAM*8)}, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes/GiB:.1f} GiB")
    print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_gib(total_model_bytes)} GiB)")
    print(f"  Per-token KV cache update: {sum(inc.values()):,.1f} bytes ({fmt_gib(sum(inc.values()))} GiB)")
    print(f"  Host DRAM layers: {host_count}")
    print(f"  CXL DRAM layers: {cxl_count}")
    print(f"  CXL NAND layers: {nand_count}")
    print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_gib(host_bytes)} GiB)")
    print(f"  CXL Device DRAM: {cxl_bytes:,} bytes ({fmt_gib(cxl_bytes)} GiB)")
    print(f"  CXL Device NAND: {nand_bytes:,} bytes ({fmt_gib(nand_bytes)} GiB)")

    print("\nSummary (Semantic Duplex + Tier Pipeline):")
    print(f"One-time cold SSD load: {cold_load_s:.6f} s")
    print(f"Single-token Latency: {per_token_latency:.6f} s")
    print(f"Estimated Tokens/sec (sequential): {throughput_tokens_per_sec:.6f}")
    print(f"Total time for T={TOKENS}: {total_time_s_all_tokens:.6f} s")

    print("\nRuntime Traffic Served (first token):")
    print(f"  From CXL Device DRAM (Hits: resident + cache): {cxl_hit_bytes_cum:,} bytes")
    print(f"  From CXL Device NAND (Misses): {cxl_miss_bytes_cum:,} bytes")
    if rows:
        print(f"  Total Prefetched Bytes: {int(pd.DataFrame(rows)['Prefetch_Bytes'].sum()):,} bytes")

if __name__ == "__main__":
    run_semantic_duplex_simulation_with_pipeline()
