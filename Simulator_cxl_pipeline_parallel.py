# simulator_cxl_pipeline_parallel_update.py
# ------------------------------------------------------------
# Two-stage pipeline across tokens with device-managed CXL-DRAM cache.
# - CXL Device DRAM is a *single* pool acting as a transparent cache for NAND
#   and also the transient buffer for tile double-buffering (NO separate caches).
# - LRU + write-back semantics (writes are irrelevant for read-only weights).
# - Optional host "hints" can PIN earliest Stage-2 layers in the device cache.
# - Stage split respects graph order: Stage-1 = longest prefix in Host DRAM.
# - Stage-2 is simulated across TOKENS; cache persists across tokens.
#   Token 1 is cold; later tokens benefit from DRAM hits if not evicted.
# ------------------------------------------------------------

import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, IO_CHUNK_BYTES, transfer_time_s
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

# Canonical labels
PL_HOST_DRAM        = "Host DRAM"
PL_CXL_DEVICE_DRAM  = "CXL Device DRAM (cache)"
PL_CXL_DEVICE_NAND  = "CXL Device NAND"

# ---------------------------
# Tunables
# ---------------------------
TILES_ENABLE     = True                 # enable intra-layer tiles
TILE_BYTES_REQ   = 4 * 1024 * 1024     # requested tile = 4 MiB
# Device-cache policy
DEVICE_DRAM_HINT_PIN_FIRST_K = 4        # try to PIN early Stage-2 layers in devDRAM (0 = disable)
DEVICE_DRAM_PIN_STRICT       = True     # if True, never evict pinned (simulate priority/hint)
# Optional write-back on eviction (read-only weights => 0)
EVIC_WRITEBACK = False

# ---------------------------
# Helpers
# ---------------------------
def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0:
        return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)       # from device DRAM
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)   # from NAND

def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

# ---------------------------
# Build model & placement
# ---------------------------
layers = build_layers()

# Fill Host DRAM with a greedy prefix + (optionally) tail hot layers *for capacity accounting*,
# but we will enforce Stage-1 as the LONGEST PREFIX of Host DRAM to preserve topology.
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes

# try to fit a lot of the prefix into Host DRAM (embed + early decoders)
order_prefix = [0]  # embed_tokens index
num_blocks = sum(1 for L in layers if L["kind"] == "DecoderBlock")
order_prefix.extend(range(1, 1 + num_blocks))  # decoder_0..decoder_{N-1}
for idx in order_prefix:
    if placement[idx] is not None: continue
    sz = layers[idx]["bytes"]
    if sz <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= sz

# tail layers (final_norm, lm_head) may also fit in Host DRAM capacity, but
# they must execute AFTER all decoders, so they belong to Stage-2 regardless.
tail_idx = [len(layers)-2, len(layers)-1]  # final_norm, lm_head
for idx in tail_idx:
    if placement[idx] is None:
        sz = layers[idx]["bytes"]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM  # physically in host DRAM
            host_free -= sz

# everything not placed becomes NAND-resident
for i in range(len(layers)):
    if placement[i] is None:
        placement[i] = PL_CXL_DEVICE_NAND

# Stage split by *topological order*: Stage-1 is the longest initial prefix placed in Host DRAM
split = 0
while split < len(layers) and placement[split] == PL_HOST_DRAM:
    split += 1
s1_idx = list(range(0, split))          # prefix in Host DRAM
s2_idx = list(range(split, len(layers))) # the rest (may include Host DRAM, NAND)

# ---------------------------
# Single Device-DRAM pool with LRU + optional PINs
# ---------------------------
class DeviceDRAMPool:
    """
    A single device DRAM pool:
      - Holds cached bytes per layer (LRU).
      - Shares capacity with transient tiles used during streaming.
      - Supports "PIN" (non-evictable) entries to model device hints.
    """
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes))
        self.used_cache = 0
        self.lru = OrderedDict()  # layer_id -> cached_bytes
        self.pinned = set()       # layer_ids that are pinned
        # transient bytes are not tracked per-tile—tile size selection ensures they fit

    @property
    def free_bytes(self):
        # All transient tiles must fit in remaining space (cap - used_cache)
        return max(0, self.cap - self.used_cache)

    def cached_bytes(self, layer_id):
        return self.lru.get(layer_id, 0)

    def has_full(self, layer_id, need_b):
        return self.lru.get(layer_id, 0) >= need_b

    def _evict_until(self, need_extra):
        """Evict LRU (except pinned) until 'need_extra' bytes are available; return bytes evicted."""
        evicted = 0
        if need_extra <= 0: return 0
        # iterate from oldest
        to_delete = []
        for lid, sz in self.lru.items():
            if lid in self.pinned and DEVICE_DRAM_PIN_STRICT:
                continue
            evicted += sz
            to_delete.append(lid)
            if evicted >= need_extra:
                break
        for lid in to_delete:
            sz = self.lru.pop(lid)
            self.used_cache -= sz
            # optional writeback (ignored for read-only weights)
            # if EVIC_WRITEBACK: pay cxlssd_time_s(sz) somewhere (omitted)
        return evicted

    def add_cache_bytes(self, layer_id, add_b):
        """Install 'add_b' more bytes of this layer into cache (accumulative; may evict others)."""
        if add_b <= 0 or self.cap <= 0:
            return 0
        # If adding exceeds capacity, evict LRU (respect pinned)
        need = max(0, (self.used_cache + add_b) - self.cap)
        if need > 0:
            self._evict_until(need)
            if (self.used_cache + add_b) > self.cap:
                # still no room (everything pinned), give up
                return 0
        cur = self.lru.get(layer_id, 0)
        self.lru[layer_id] = cur + add_b
        self.lru.move_to_end(layer_id, last=True)
        self.used_cache += add_b
        return add_b

    def pin_layer(self, layer_id, size_b):
        """Pin (reserve) 'size_b' bytes for 'layer_id' if possible."""
        if size_b <= 0 or self.cap <= 0:
            return 0
        # ensure full layer fits by evicting others
        cur = self.lru.get(layer_id, 0)
        add = max(0, size_b - cur)
        need = max(0, (self.used_cache + add) - self.cap)
        if need > 0:
            self._evict_until(need)
            if (self.used_cache + add) > self.cap:
                return 0
        if add > 0:
            self.add_cache_bytes(layer_id, add)
        self.pinned.add(layer_id)
        return size_b

# ---------------------------
# Tile sizing uses the same pool (no separate buffers):
# choose tile <= (free_bytes / 2) for double-buffering
# ---------------------------
def choose_tile_bytes(dev_pool: DeviceDRAMPool):
    max_by_devdram = max(IO_CHUNK_BYTES, dev_pool.free_bytes // 2)
    return max(IO_CHUNK_BYTES, min(TILE_BYTES_REQ, max_by_devdram))

def per_tile_times(sz_bytes, flops, cores, tile_bytes):
    tiles = max(1, math.ceil(sz_bytes / max(1, tile_bytes)))
    tbytes = min(tile_bytes, sz_bytes)
    f_tile = cxlssd_time_s(tbytes)      # NAND -> devDRAM
    devdram_tile = cxl_time_s(tbytes)   # serve tile from devDRAM to cores
    comp_tile = compute_time_s(flops * (tbytes / max(1, sz_bytes)), cores)
    c_tile = max(comp_tile, devdram_tile)
    return tiles, f_tile, c_tile

# ---------------------------
# Stage-1 time (prefix in Host DRAM)
# ---------------------------
def stage1_time(cores):
    total = 0.0
    rows = []
    for k in s1_idx:
        L = layers[k]; sz=L["bytes"]; fl=L["flops"]
        comp_s = compute_time_s(fl, cores)
        mem_s  = dram_time_s(sz)
        t = max(comp_s, mem_s)
        total += t
        rows.append({
            "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
            "Placement": PL_HOST_DRAM, "Stage": 1,
            "Bytes": sz, "Compute_s(on stage cores)": comp_s,
            "Mem_s_dram": mem_s, "Mem_s_cxl": 0.0, "Mem_s_nand": 0.0,
            "Stage1_Layer_Time_s": t
        })
    return total, rows

# ---------------------------
# Stage-2 across TOKENS with device cache:
# Returns (per_token_times, rows_for_token1, accounting)
# ---------------------------
def stage2_across_tokens(cores, tokens):
    dev_pool = DeviceDRAMPool(cxl_dev_dram_capacity_bytes)
    # Optional: pin earliest Stage-2 layers (host "hints") to survive LRU across tokens
    if DEVICE_DRAM_HINT_PIN_FIRST_K > 0:
        to_pin = [k for k in s2_idx if placement[k] != PL_HOST_DRAM]  # only NAND layers benefit
        to_pin = to_pin[:DEVICE_DRAM_HINT_PIN_FIRST_K]
        for k in to_pin:
            dev_pool.pin_layer(k, layers[k]["bytes"])

    per_token_times = []
    rows_token1 = []
    # accounting
    bytes_devdram_hits = 0
    bytes_nand = 0
    bytes_hostdram_in_s2 = 0

    for t in range(tokens):
        # For cross-layer overlap (F_{i+1} with C_i) we keep arrays:
        F = []
        C = []
        O = []
        # DEBUG rows for token 1
        row_builder = []

        # Current tile size depends on free DRAM (cache may have grown)
        tile_bytes = choose_tile_bytes(dev_pool) if TILES_ENABLE else None

        for kpos, k in enumerate(s2_idx):
            L = layers[k]; sz=L["bytes"]; fl=L["flops"]
            plc = placement[k]

            # Host DRAM in Stage-2 (tail layers possibly)
            if plc == PL_HOST_DRAM:
                f_i = 0.0
                comp_i = compute_time_s(fl, cores)
                dev_i  = dram_time_s(sz)     # served from host DRAM directly
                c_i = max(comp_i, dev_i)
                overhead_i = 0.0
                bytes_hostdram_in_s2 += sz

                if t == 0:
                    row_builder.append({
                        "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
                        "Placement": plc, "Stage": 2,
                        "Bytes": sz,
                        "F_i (NAND→devDRAM)_s": f_i,
                        "C_i (compute)_s": c_i,
                        "Served_From": "Host DRAM",
                        "Tiles": 0, "Tile_bytes": 0, "Tile_overhead_s": 0.0
                    })

            else:
                # NAND-backed layer with device cache
                cached = dev_pool.cached_bytes(k)
                if cached >= sz:
                    # full hit: no fetch, serve entirely from device DRAM
                    f_i = 0.0
                    comp_i = compute_time_s(fl, cores)
                    dev_i  = cxl_time_s(sz)
                    c_i = max(comp_i, dev_i)
                    overhead_i = 0.0
                    bytes_devdram_hits += sz
                    # touch in LRU
                    dev_pool.add_cache_bytes(k, 0)

                    if t == 0:
                        row_builder.append({
                            "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
                            "Placement": plc, "Stage": 2,
                            "Bytes": sz,
                            "F_i (NAND→devDRAM)_s": 0.0,
                            "C_i (compute from devDRAM)_s": c_i,
                            "Served_From": "CXL Device DRAM (cache hit)",
                            "Tiles": 0, "Tile_bytes": 0, "Tile_overhead_s": 0.0
                        })
                else:
                    # partial or cold: need to fetch the missing bytes M
                    M = sz - cached
                    bytes_nand += M
                    f_i = cxlssd_time_s(M)  # whole-missing fetch time (used for inter-layer overlap)
                    comp_i = compute_time_s(fl, cores)
                    dev_i  = cxl_time_s(sz) # compute consumes full layer from devDRAM
                    c_i = max(comp_i, dev_i)

                    # Intra-layer tiles (approximate per-layer overhead due to fill/drain)
                    if TILES_ENABLE:
                        tb = tile_bytes or IO_CHUNK_BYTES
                        tiles, f_tile, c_tile = per_tile_times(M, fl, cores, tb)  # only missing part streams
                        overhead_i = min(f_tile, c_tile) if tiles >= 2 else 0.0
                        # After finishing layer, assume we install the full layer into cache (idealized)
                        dev_pool.add_cache_bytes(k, M)  # grow towards full
                    else:
                        tiles, f_tile, c_tile, overhead_i = 1, 0.0, 0.0, 0.0
                        dev_pool.add_cache_bytes(k, M)

                    if t == 0:
                        row_builder.append({
                            "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
                            "Placement": plc, "Stage": 2,
                            "Bytes": sz,
                            "F_i (NAND→devDRAM)_s": f_i,
                            "C_i (compute from devDRAM)_s": c_i,
                            "Served_From": "NAND→devDRAM stream (partial/cold)",
                            "Tiles": tiles, "Tile_bytes": tile_bytes if TILES_ENABLE else 0,
                            "Tile_overhead_s": overhead_i
                        })

            F.append(f_i)
            C.append(c_i)
            O.append(overhead_i)

        # Cross-layer two-stage overlap:
        n = len(F)
        if n == 0:
            per_token_times.append(0.0)
        else:
            base = F[0]
            for i in range(1, n):
                base += max(F[i], C[i-1])
            base += C[-1]
            total = base + sum(O)
            per_token_times.append(total)

        if t == 0:
            rows_token1 = row_builder  # save detailed rows for token 1

    acct = {
        "bytes_devdram_hits": bytes_devdram_hits,
        "bytes_nand": bytes_nand,
        "bytes_hostdram_in_s2": bytes_hostdram_in_s2,
        "devdram_cache_used_GB": (sum(dev_pool.lru.values()) / (1024**3)) if TOKENS > 0 else 0.0
    }
    return per_token_times, rows_token1, acct

# ---------------------------
# Search best core split & compute totals
# ---------------------------
def choose_best_split(P):
    best = None
    best_rows1 = None
    best_rows_token1 = None
    best_per_token_s2 = None
    best_acct = None
    for p1 in range(1, P):
        p2 = P - p1
        S1, rows1 = stage1_time(p1)
        per_token_s2, rows_t1, acct = stage2_across_tokens(p2, TOKENS)
        # Pipeline total = fill S1 + token1 S2 + sum_{t=2..T} max(S1, S2_t)
        total = S1 + per_token_s2[0]
        for t in range(1, TOKENS):
            total += max(S1, per_token_s2[t])
        # measure steady bottleneck as max(S1, median S2 after warm)
        steady_s2 = sorted(per_token_s2[1:])[len(per_token_s2[1:])//2] if TOKENS > 1 else per_token_s2[0]
        bottleneck = max(S1, steady_s2)
        cand = (total, bottleneck, p1, p2, S1, per_token_s2, rows1, rows_t1, acct)
        if best is None or cand < best:
            best = cand
            best_rows1 = rows1
            best_rows_token1 = rows_t1
            best_per_token_s2 = per_token_s2
            best_acct = acct
    total, bottleneck, p1, p2, S1, per_token_s2, rows1, rows_t1, acct = best
    return (p1, p2, S1, per_token_s2, bottleneck, rows1, rows_t1, acct, total)

# ---------------------------
# Run
# ---------------------------
P = cpu_cores
p1, p2, S1, per_token_s2, bottleneck, rows1, rows_token1, acct, pipeline_total_time = choose_best_split(P)

# Summaries
throughput_steady = 1.0 / bottleneck if bottleneck > 0 else 0.0
S2_first = per_token_s2[0]
S2_steady = (sorted(per_token_s2[1:])[len(per_token_s2[1:])//2] if TOKENS > 1 else S2_first)

# Output tables
rows_all = []
rows_all.extend(rows1 or [])
# only show detailed Stage-2 per-layer for token 1 (cold) to keep the table readable
for r in (rows_token1 or []):
    rows_all.append(r)
df = pd.DataFrame(rows_all)
df.to_csv("sim_pipeline_devicecache_token1.csv", index=False)
print(df)

print("\nSummary (Pipeline with device-managed CXL-DRAM cache):")
print(f"Tiles enabled: {TILES_ENABLE}, tile_req={TILE_BYTES_REQ/(1024**2):.1f} MiB (actual per-layer chosen dynamically)")
print(f"Core split over P={P}: p1={p1} (Stage-1/Host DRAM prefix), p2={p2} (Stage-2)")
print(f"Stage-1 per token S1={S1:.6f} s")
print(f"Stage-2 per token S2 (token1 cold)={S2_first:.6f} s, S2 (steady median)={S2_steady:.6f} s")
print(f"Steady bottleneck per token = {bottleneck:.6f} s → throughput ≈ {throughput_steady:.6f} tok/s")
print(f"Total time for T={TOKENS}: {pipeline_total_time:.6f} s")

# Model / placement info
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_HOST_DRAM)
nand_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_CXL_DEVICE_NAND)

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  CXL Device NAND: {nand_bytes:,}  ({fmt_bytes(nand_bytes)})")
print("  CXL Device DRAM: used dynamically as cache + transient tiles (single pool)")

print("\nCapacities & Tiers:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB (single pool)")
print(f"  CXL Device NAND cap : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
print(f"  Stage-1 (Host DRAM prefix) layers: {[i+1 for i in s1_idx]}")
print(f"  Stage-2 layers: {[i+1 for i in s2_idx]}")

print("\nStage-2 memory service accounting (across all tokens):")
print(f"  Served from CXL Device DRAM (cache hits): {acct['bytes_devdram_hits']:,} bytes")
print(f"  Served from NAND (missed then installed): {acct['bytes_nand']:,} bytes")
print(f"  Served from Host DRAM (Stage-2 tail): {acct['bytes_hostdram_in_s2']:,} bytes")
print(f"  Peak devDRAM cache used (approx end of sim): {acct['devdram_cache_used_GB']:.3f} GB")
