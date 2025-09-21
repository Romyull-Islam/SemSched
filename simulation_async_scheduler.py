# simulation_async_scheduler.py
# An asynchronous simulation with a scheduling-based approach.
# Each layer is scheduled on a CPU core when ready, and I/O is handled by an
# async I/O pool. Prefetching is done for layers ahead in the sequence.
# This sim captures compute and I/O overlap, compute stalls due to I/O,
# and the impact of core allocation on throughput and latency.
# Note: This is a simplified model and does not capture all real-world complexities.
import heapq
from collections import deque, OrderedDict
import pandas as pd

from tiers import HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s, GiB, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
from sim_cfg import (
    cpu_freq_hz,
    cpu_cores,
    flops_per_cycle_per_core,
    parallel_efficiency,
    TOKENS,
    host_dram_capacity_bytes,
    cxl_dev_dram_capacity_bytes,
)
from model_cfg import build_layers, BYTES_PER_PARAM

IO_CHUNK_BYTES = 256 * 1024
PREFETCH_QUEUE_DEPTH = 8  # Lookahead for prefetching
IO_THREAD_POOL_SIZE = 4   # Number of I/O threads

PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"


class Layer:
    def __init__(self, idx, info):
        self.idx = idx
        self.name = info["name"]
        self.bytes_size = info["bytes"]
        self.flops = info["flops"]
        self.params = info.get("params", 0)
        self.state = "waiting"  # waiting, ready, running, completed
        self.kv_cache_bytes = info.get("kv_cache_bytes", 0)


def compute_time(flops, cores=1):
    if flops <= 0 or cores <= 0:
        return 0.0
    flops_per_sec = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_sec


def transfer_time(bytes_size, tier):
    return transfer_time_s(bytes_size, tier, IO_CHUNK_BYTES)


class AsyncIOPool:
    def __init__(self, pool_size=IO_THREAD_POOL_SIZE):
        self.pool_size = pool_size
        self.threads = [0.0] * pool_size

    def schedule_io(self, current_time, bytes_size):
        t_idx = min(range(self.pool_size), key=lambda i: self.threads[i])
        start_time = max(current_time, self.threads[t_idx])
        duration = transfer_time(bytes_size, CXL_SSD_NAND)
        finish_time = start_time + duration
        self.threads[t_idx] = finish_time
        return start_time, finish_time


class DeviceDRAMPool:
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes))
        self.used_cache = 0
        self.lru = OrderedDict()
        self.pinned = set()

    @property
    def free_bytes(self):
        return max(0, self.cap - self.used_cache)

    def cached_bytes(self, layer_id):
        return self.lru.get(layer_id, 0)

    def _evict_until(self, need_extra):
        if need_extra <= 0:
            return
        to_delete = []
        for lid, sz in self.lru.items():
            if lid in self.pinned:
                continue
            to_delete.append(lid)
            if sum(self.lru[d] for d in to_delete) >= need_extra:
                break
        for lid in to_delete:
            self.used_cache -= self.lru.pop(lid)

    def add_cache_bytes(self, layer_id, add_b):
        if add_b <= 0:
            return True
        add_b = int(add_b)
        needed = max(0, (self.used_cache + add_b) - self.cap)
        if needed > 0:
            self._evict_until(needed)
        if (self.used_cache + add_b) > self.cap:
            return False
        self.used_cache += add_b
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
        return True


class Scheduler:
    def __init__(self, layers_info, prefetch_pool_size=IO_THREAD_POOL_SIZE):
        self.layers_info = layers_info
        self.time = 0.0
        self.layers = [Layer(i, info) for i, info in enumerate(layers_info)]
        self.io_pool = AsyncIOPool(pool_size=prefetch_pool_size)
        self.event_queue = []
        self.ready_queue = deque()
        self.running = {}
        self.completed = set()
        self.dev_pool = DeviceDRAMPool(cxl_dev_dram_capacity_bytes)
        self.TOTAL_CORES = cpu_cores
        self.records = []
        self.acct = {"compute_stall_s": 0.0, "bytes_prefetched": 0, "bytes_from_nand_miss": 0}
        self.fetched_or_queued = set()

        # --- NEW: Calculate incremental KV cache size per token ---
        sequence_length = 512
        self.kv_cache_increment = {}
        total_kv_cache_increment = 0
        for L in layers_info:
            if L["kind"] == "DecoderBlock":
                head_dim = L.get("head_dim", 128)
                kv_heads = L.get("kv_heads", 40 if len(layers_info) > 35 else 8)
                self.kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
                total_kv_cache_increment += self.kv_cache_increment[L["name"]]
            else:
                self.kv_cache_increment[L["name"]] = 0

        # Placement
        host_dram_cap = host_dram_capacity_bytes
        cxl_dram_cap = cxl_dev_dram_capacity_bytes

        def can_place(sz, cap):
            return cap >= sz

        self.placement = {}
        hot_layers = ("lm_head", "final_norm")
        for layer in self.layers:
            sz = layer.bytes_size + (self.kv_cache_increment[layer.name] * sequence_length)
            if layer.name in hot_layers and can_place(sz, host_dram_cap):
                self.placement[layer.idx] = PL_HOST_DRAM
                host_dram_cap -= sz

        for layer in self.layers:
            if layer.idx in self.placement:
                continue
            sz = layer.bytes_size + (self.kv_cache_increment[layer.name] * sequence_length)
            if can_place(sz, host_dram_cap):
                self.placement[layer.idx] = PL_HOST_DRAM
                host_dram_cap -= sz
            elif can_place(sz, cxl_dram_cap):
                self.placement[layer.idx] = PL_CXL_DEV_DRAM
                self.dev_pool.add_cache_bytes(layer.idx, sz)
                self.dev_pool.pinned.add(layer.idx)
                cxl_dram_cap -= sz
            else:
                self.placement[layer.idx] = PL_CXL_DEV_NAND

    def prefetch(self, layer_idx):
        if layer_idx in self.fetched_or_queued or self.placement[layer_idx] != PL_CXL_DEV_NAND:
            return
        layer = self.layers[layer_idx]
        cached = self.dev_pool.cached_bytes(layer_idx)
        needed = max(0, (layer.bytes_size + self.kv_cache_increment[layer.name]) - cached)
        if needed <= 0:
            return
        start, finish = self.io_pool.schedule_io(self.time, needed)
        heapq.heappush(self.event_queue, (finish, "io_complete", layer_idx, needed))
        self.fetched_or_queued.add(layer_idx)

    def check_ready(self, layer_idx):
        layer = self.layers[layer_idx]
        if self.placement[layer_idx] in (PL_HOST_DRAM, PL_CXL_DEV_DRAM):
            return True
        return self.dev_pool.cached_bytes(layer_idx) >= (layer.bytes_size + self.kv_cache_increment[layer.name])

    def initialize(self):
        for i, layer in enumerate(self.layers):
            # Prefetch layers ahead
            for j in range(1, PREFETCH_QUEUE_DEPTH + 1):
                future_idx = i + j
                if future_idx < len(self.layers):
                    self.prefetch(future_idx)
            if self.check_ready(layer.idx):
                layer.state = "ready"
                self.ready_queue.append(layer.idx)
            else:
                layer.state = "waiting"
                self.prefetch(layer.idx)

    def run(self):
        self.initialize()
        tokens_completed = 0

        while tokens_completed < TOKENS:
            # Schedule layers from ready queue
            while self.ready_queue and len(self.running) < self.TOTAL_CORES:
                idx = self.ready_queue.popleft()
                layer = self.layers[idx]
                layer.state = "running"
                comp_dur = compute_time(layer.flops, cores=1)
                finish_time = self.time + comp_dur
                self.running[idx] = finish_time
                heapq.heappush(self.event_queue, (finish_time, "compute_complete", idx))

                # Include KV cache increment for per-token transfer
                sz = layer.bytes_size + self.kv_cache_increment[layer.name]
                served_from = self.placement[idx]
                if self.placement[idx] == PL_CXL_DEV_NAND:
                    if self.dev_pool.cached_bytes(idx) >= sz:
                        served_from = "CXL DRAM (prefetched hit)"
                    else:
                        served_from = "CXL NAND (stall)"
                        stall_time = transfer_time(sz, CXL_SSD_NAND)
                        self.acct["compute_stall_s"] += stall_time
                        self.acct["bytes_from_nand_miss"] += sz
                        self.time += stall_time
                        self.dev_pool.add_cache_bytes(idx, sz)
                elif self.placement[idx] == PL_CXL_DEV_DRAM:
                    served_from = "CXL DRAM (resident)"
                elif self.placement[idx] == PL_HOST_DRAM:
                    served_from = "Host DRAM"

                self.records.append(
                    {
                        "Layer": idx + 1,
                        "Name": layer.name,
                        "Placement": self.placement[idx],
                        "Served_From": served_from,
                        "Layer_Time_s": comp_dur,
                        "Bytes": sz,
                        "Flops": layer.flops,
                    }
                )

            if not self.event_queue:
                print("Deadlock detected: no events to process.")
                break

            prev_time = self.time
            event = heapq.heappop(self.event_queue)
            self.time = event[0]
            event_type = event[1]
            event_idx = event[2]

            if len(self.running) == 0 and event_type != "compute_complete":
                self.acct["compute_stall_s"] += (self.time - prev_time)

            if event_type == "io_complete":
                needed = event[3]
                if self.dev_pool.add_cache_bytes(event_idx, needed):
                    self.acct["bytes_prefetched"] += needed
                    layer = self.layers[event_idx]
                    if layer.state == "waiting" and self.check_ready(event_idx):
                        layer.state = "ready"
                        self.ready_queue.append(event_idx)
                self.fetched_or_queued.discard(event_idx)

            elif event_type == "compute_complete":
                layer = self.layers[event_idx]
                layer.state = "completed"
                self.completed.add(event_idx)
                del self.running[event_idx]

                if len(self.completed) == len(self.layers):
                    tokens_completed += 1
                    self.completed.clear()
                    self.fetched_or_queued.clear()
                    for i, layer in enumerate(self.layers):
                        layer.state = "waiting"
                        # Prefetch layers ahead for next token
                        for j in range(1, PREFETCH_QUEUE_DEPTH + 1):
                            future_idx = i + j
                            if future_idx < len(self.layers):
                                self.prefetch(future_idx)
                        if self.check_ready(layer.idx):
                            layer.state = "ready"
                            self.ready_queue.append(layer.idx)
                        else:
                            self.prefetch(layer.idx)

        # Output results
        df = pd.DataFrame(self.records)
        df = df.reindex(columns=["Layer", "Name", "Placement", "Served_From", "Layer_Time_s", "Bytes", "Flops"])
        print(df.sort_values("Layer", inplace=False).to_string(index=False))

        total_model_bytes = sum(layer.bytes_size for layer in self.layers)
        total_kv_cache_bytes = sum(layer.kv_cache_bytes for layer in self.layers)
        cold_load_s = transfer_time_s(total_model_bytes, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))
        per_token_latency = self.time / TOKENS if TOKENS > 0 else 0.0
        throughput = TOKENS / self.time if self.time > 0 else 0.0
        total_time_for_all_tokens = cold_load_s + self.time

        host_count = sum(1 for v in self.placement.values() if v == PL_HOST_DRAM)
        cxl_dram_count = sum(1 for v in self.placement.values() if v == PL_CXL_DEV_DRAM)
        cxl_nand_count = sum(1 for v in self.placement.values() if v == PL_CXL_DEV_NAND)
        host_bytes = sum(
            layer.bytes_size + (self.kv_cache_increment[layer.name] * sequence_length)
            for layer in self.layers
            if self.placement[layer.idx] == PL_HOST_DRAM
        )
        cxl_dram_bytes = sum(
            layer.bytes_size + (self.kv_cache_increment[layer.name] * sequence_length)
            for layer in self.layers
            if self.placement[layer.idx] == PL_CXL_DEV_DRAM
        )
        cxl_nand_bytes = sum(
            layer.bytes_size + (self.kv_cache_increment[layer.name] * sequence_length)
            for layer in self.layers
            if self.placement[layer.idx] == PL_CXL_DEV_NAND
        )

        def fmt_bytes(n):
            return f"{n / GiB:.3f} GiB"

        def fmt_params(n):
            if n >= 1e9:
                return f"{n / 1e9:.3f} B"
            if n >= 1e6:
                return f"{n / 1e6:.3f} M"
            if n >= 1e3:
                return f"{n / 1e3:.3f} K"
            return str(n)

        total_params = sum(layer.params for layer in self.layers if hasattr(layer, "params"))
        if total_params == 0:
            total_params = sum(l["params"] for l in self.layers_info)

        model_dtype_bits = int(BYTES_PER_PARAM * 8)

        print(f"\nSummary (Scheduling-based Async Simulation):")
        print(f"One-time cold SSD load: {cold_load_s:.6f} s")
        print(f"Single-token Latency: {per_token_latency:.6f} s")
        print(f"Estimated Tokens/sec: {throughput:.6f}")
        print(f"Total time for T={TOKENS}: {total_time_for_all_tokens:.6f} s")

        print("\nModel Size & Placement:")
        print(f"  Dtype: FP{model_dtype_bits}, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes / GiB:.1f} GiB")
        print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
        print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
        print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")
        print(f"  Host DRAM layers: {host_count}")
        print(f"  CXL DRAM layers: {cxl_dram_count}")
        print(f"  CXL NAND layers: {cxl_nand_count}")
        print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_bytes(host_bytes)})")
        print(f"  CXL Device DRAM: {cxl_dram_bytes:,} bytes ({fmt_bytes(cxl_dram_bytes)})")
        print(f"  CXL Device NAND: {cxl_nand_bytes:,} bytes ({fmt_bytes(cxl_nand_bytes)})")

        print("\nPerformance Breakdown:")
        print(f"  Total compute stall time (waiting for I/O): {self.acct['compute_stall_s']:.6f} s")
        print(f"  Total bytes successfully prefetched: {self.acct['bytes_prefetched']:,} bytes")
        print(f"  Total bytes read from NAND on a miss (stall): {self.acct['bytes_from_nand_miss']:,} bytes")


if __name__ == "__main__":
    layers_info = build_layers(sequence_length=512)
    sim = Scheduler(layers_info)
    sim.run()