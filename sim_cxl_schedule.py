import heapq
from collections import deque, defaultdict
import pandas as pd

from tiers import HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from sim_cfg import cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency, TOKENS
from model_cfg import build_layers

IO_CHUNK_BYTES = 256 * 1024

class Layer:
    def __init__(self, idx, info):
        self.idx = idx
        self.name = info["name"]
        self.bytes_size = info["bytes"]
        self.flops = info["flops"]
        self.state = 'waiting'  # waiting, ready, running, completed

def compute_time(flops, cores=cpu_cores):
    if flops <= 0 or cores <= 0:
        return 0.0
    flops_per_sec = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_sec

def transfer_time(bytes_size, tier):
    return transfer_time_s(bytes_size, tier, IO_CHUNK_BYTES)

class AsyncIOPool:
    def __init__(self, pool_size=4):
        self.pool_size = pool_size
        self.threads = [0.0]*pool_size

    def schedule_io(self, current_time, bytes_size):
        t_idx = min(range(self.pool_size), key=lambda i: self.threads[i])
        start_time = max(current_time, self.threads[t_idx])
        duration = transfer_time(bytes_size, CXL_SSD_NAND)
        finish_time = start_time + duration
        self.threads[t_idx] = finish_time
        return start_time, finish_time

class Scheduler:
    def __init__(self, layers_info, prefetch_pool_size=4):
        self.time = 0.0
        self.layers = [Layer(i, info) for i, info in enumerate(layers_info)]
        self.io_pool = AsyncIOPool(pool_size=prefetch_pool_size)
        self.event_queue = []
        self.ready_queue = deque()
        self.running = {}
        self.completed = set()
        self.prefetch_cache = defaultdict(int)
        self.TOTAL_CORES = cpu_cores

        self.records = []  # for storing output info per layer execution

        # Simple static placement:
        host_dram_cap = 4 * (1024**3)
        cxl_dram_cap = 4 * (1024**3)

        def can_place(sz, cap):
            return cap >= sz

        self.placement = {}
        hot_layers = ("lm_head", "final_norm")
        for layer in self.layers:
            if layer.name in hot_layers and can_place(layer.bytes_size, host_dram_cap):
                self.placement[layer.idx] = HOST_DRAM
                host_dram_cap -= layer.bytes_size
        for layer in self.layers:
            if layer.idx in self.placement:
                continue
            if can_place(layer.bytes_size, host_dram_cap):
                self.placement[layer.idx] = HOST_DRAM
                host_dram_cap -= layer.bytes_size
            elif can_place(layer.bytes_size, cxl_dram_cap):
                self.placement[layer.idx] = CXL_DRAM
                cxl_dram_cap -= layer.bytes_size
            else:
                self.placement[layer.idx] = CXL_SSD_NAND

    def prefetch(self, layer_idx):
        layer = self.layers[layer_idx]
        cached = self.prefetch_cache[layer_idx]
        needed = max(0, layer.bytes_size - cached)
        if needed <= 0:
            return
        start, finish = self.io_pool.schedule_io(self.time, needed)
        heapq.heappush(self.event_queue, (finish, 'io_complete', layer_idx))
        # Could track prefetch bytes/time here if needed

    def check_ready(self, layer_idx):
        tier = self.placement[layer_idx]
        if tier == HOST_DRAM or tier == CXL_DRAM:
            return True
        else:
            return self.prefetch_cache[layer_idx] >= self.layers[layer_idx].bytes_size

    def initialize(self):
        for layer in self.layers:
            if self.check_ready(layer.idx):
                layer.state = 'ready'
                self.ready_queue.append(layer.idx)
            else:
                layer.state = 'waiting'
                self.prefetch(layer.idx)

    def run(self):
        self.initialize()
        tokens_completed = 0

        while tokens_completed < TOKENS:
            while self.ready_queue and len(self.running) < self.TOTAL_CORES:
                idx = self.ready_queue.popleft()
                layer = self.layers[idx]
                layer.state = 'running'
                comp_dur = compute_time(layer.flops, cores=1)
                finish_time = self.time + comp_dur
                self.running[idx] = finish_time
                heapq.heappush(self.event_queue, (finish_time, 'compute_complete', idx))
                placement_name = self.placement[idx].name
                served_from = placement_name
                if self.placement[idx] == CXL_SSD_NAND:
                    served_from = "CXL NAND (prefetch miss)"
                elif self.placement[idx] == CXL_DRAM:
                    served_from = "CXL DRAM (resident)"
                elif self.placement[idx] == HOST_DRAM:
                    served_from = "Host DRAM"

                self.records.append({
                    "Layer": idx + 1,
                    "Name": layer.name,
                    "Placement": placement_name,
                    "Served_From": served_from,
                    "Layer_Start_s": self.time,
                    "Layer_End_s": finish_time,
                    "Layer_Time_s": comp_dur,
                    "Bytes": layer.bytes_size,
                    "Flops": layer.flops
                })

            if not self.event_queue:
                print("Deadlock: No more events to process.")
                break

            self.time, event_type, event_idx = heapq.heappop(self.event_queue)

            if event_type == 'io_complete':
                layer = self.layers[event_idx]
                self.prefetch_cache[event_idx] = layer.bytes_size
                if layer.state == 'waiting':
                    layer.state = 'ready'
                    self.ready_queue.append(event_idx)

            elif event_type == 'compute_complete':
                layer = self.layers[event_idx]
                layer.state = 'completed'
                self.completed.add(event_idx)
                del self.running[event_idx]

                if len(self.completed) == len(self.layers):
                    tokens_completed += 1
                    self.completed.clear()
                    # Reset filtering for next token
                    for l in self.layers:
                        l.state = 'waiting'
                        if self.check_ready(l.idx):
                            l.state = 'ready'
                            self.ready_queue.append(l.idx)
                        else:
                            self.prefetch(l.idx)

        # Output DataFrame
        df = pd.DataFrame(self.records)
        print(df.sort_values("Layer", inplace=False).to_string(index=False))

        total_model_bytes = sum(layer.bytes_size for layer in self.layers)
        total_time = self.time
        throughput = TOKENS / total_time if total_time > 0 else 0

        # Placement counts
        host_count = sum(1 for v in self.placement.values() if v == HOST_DRAM)
        cxl_dram_count = sum(1 for v in self.placement.values() if v == CXL_DRAM)
        cxl_nand_count = sum(1 for v in self.placement.values() if v == CXL_SSD_NAND)

        print("\nSummary (Scheduling-based Async Simulation):")
        print(f"Total simulated tokens: {TOKENS}")
        print(f"Total simulation time: {total_time:.6f} s")
        print(f"Throughput (tokens/sec): {throughput:.6f}")
        print(f"Model Size: {total_model_bytes/(1024**3):.3f} GiB")
        print(f"Placement:")
        print(f"  Host DRAM layers: {host_count}")
        print(f"  CXL DRAM layers: {cxl_dram_count}")
        print(f"  CXL NAND layers: {cxl_nand_count}")

if __name__ == "__main__":
    layers_info = build_layers()
    sim = Scheduler(layers_info)
    sim.run()
