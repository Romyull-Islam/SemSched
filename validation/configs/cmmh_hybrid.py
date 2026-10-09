"""
A CMM-H-style hybrid CXL device in gem5/SimCXL.

SimCXL ships a flat Type 3 expander; CMM-H is a DDR4 cache in front of a NAND
backend. Here the cache-hit and cache-miss paths are separate address ranges
behind a device-side crossbar, so a traffic generator picks one by address:

    gen, gen2 -> membus -+- host DRAM          (DDR5-4800)
                         +- cxl_bus -+- CXL DRAM cache  (DDR4, hit path)
                                     +- NAND backend    (miss path)

Every tier size is an argument. Run through sweep.py rather than directly.

Modes
  stream      one sequential read stream        -> calibrate against the prototype
  duplex      reads alone, then reads + writes  -> test the write-injection claim
  concurrent  target tier and host DRAM at once -> test bus independence
  trace       replay a decode trace             -> not yet wired

Calibration targets, Soltaniyeh et al. HotStorage 2025:
  cache hit   27 GB/s   505 ns
  cache miss   5 GB/s  1547 ns
"""
import argparse

import m5
from m5.objects import (
    AddrRange,
    DDR4_2400_16x4,
    DRAMInterface,
    MemCtrl,
    PyTrafficGen,
    Root,
    SimpleMemory,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
)
from m5.util import addToPath


def gb(s):
    """'48GB' or '48' -> bytes."""
    s = str(s).upper().replace("GB", "").strip()
    return int(float(s) * 1024 ** 3)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--host-dram", default="16GB")
    p.add_argument("--cxl-dram", default="48GB")
    p.add_argument("--nand", default="1024GB")
    p.add_argument("--accel-mem", default="0GB",
                   help="accelerator memory usable for weights; 0 = no GPU")
    p.add_argument("--target", default="cxl", choices=["host", "cxl", "nand"],
                   help="which tier the stream/duplex traffic reads")
    p.add_argument("--mode", default="stream",
                   choices=["stream", "duplex", "concurrent", "trace"])
    p.add_argument("--duration", type=float, default=2e6,
                   help="ns of traffic to generate per stream")
    p.add_argument("--req-size", type=int, default=64)
    return p.parse_args()


args = parse()

system = System()
system.clk_domain = SrcClockDomain(clock="2.75GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "timing"

# Address map, laid out fastest tier first; the traffic generator targets a
# tier by choosing its address range.
cur = 0
ranges = {}
for name, size in (("accel", gb(args.accel_mem)),
                   ("host",  gb(args.host_dram)),
                   ("cxl",   gb(args.cxl_dram)),
                   ("nand",  gb(args.nand))):
    if size == 0:
        ranges[name] = None
        continue
    ranges[name] = AddrRange(cur, cur + size - 1)
    cur += size
system.mem_ranges = [r for r in ranges.values() if r is not None]

system.membus = SystemXBar()


def simple_mem(rng, latency_ns, bw_GBps):
    """A tier expressed as its measured achieved bandwidth and latency.

    SimpleMemory rather than a DRAMInterface: the CMM-H numbers are published
    achieved figures, and deriving them from DDR timing parameters would mean
    guessing at the device's internals.
    """
    # null=True: otherwise gem5 backs every range with host RAM, and a
    # 16+48+1024 GB hierarchy fails in instantiate(). Only timing is modelled,
    # not data.
    m = SimpleMemory(range=rng,
                     latency=f"{latency_ns}ns",
                     bandwidth=f"{bw_GBps}GB/s",
                     null=True)
    return m


if ranges["accel"]:
    system.accel_mem = simple_mem(ranges["accel"], 300, 1792.0)   # GDDR7
    system.accel_mem.port = system.membus.mem_side_ports

# Configured rates are calibrated, not nominal: SimpleMemory admits more than
# its configured bandwidth under this generator, by a rate-dependent amount
# (~9% at 38 GB/s, ~6% at 25), so each tier has its own fitted factor. Targets
# are the CMM-H prototype's measured rates: host 38.4, device DRAM cache 27.0,
# NAND backend 5.0 GB/s.
CAL_HOST = 38.4 / 38.47 * (27.0 / 29.429)
CAL_CXL  = 27.0 / 26.33 * (27.0 / 29.429)
CAL_NAND = 5.0 / 4.96 * (27.0 / 29.429)
system.host_mem = simple_mem(ranges["host"], 200, 38.4 * CAL_HOST)     # DDR5-4800
system.host_mem.port = system.membus.mem_side_ports

# Both device tiers hang off their own device-side crossbar, behind the
# system membus.
system.cxl_bus = SystemXBar()
system.cxl_dram = simple_mem(ranges["cxl"], 505, 27.0 * CAL_CXL)      # DDR4 cache
system.cxl_dram.port = system.cxl_bus.mem_side_ports
system.nand = simple_mem(ranges["nand"], 1547, 5.0 * CAL_NAND)         # NAND backend
system.nand.port = system.cxl_bus.mem_side_ports

system.cxl_bus.cpu_side_ports = system.membus.mem_side_ports

system.gen = PyTrafficGen()
system.gen.port = system.membus.cpu_side_ports
# A second generator drives another tier at the same time. One generator
# interleaving addresses cannot test bus independence: its own issue rate
# becomes the bottleneck and the tiers take turns.
system.gen2 = PyTrafficGen()
system.gen2.port = system.membus.cpu_side_ports


def stream(rng, dur, read_frac):
    """Sequential traffic over one tier's range."""
    return system.gen.createLinear(
        dur,                       # duration, ns
        rng.start, rng.end,        # address bounds
        args.req_size,             # request size
        1000, 1000,                # min/max inter-request period, ps
        int(read_frac * 100),      # read percentage
        0,                         # data limit (0 = unbounded)
    )


def generator():
    dur = int(args.duration)
    tgt = ranges[args.target]
    if args.mode == "stream":
        # Read one tier: the calibration point.
        yield stream(tgt, dur, 1.0)
    elif args.mode == "duplex":
        # Reads alone, then reads with writes interleaved. On a full-duplex
        # link the second phase should not slow the reads down.
        yield stream(tgt, dur, 1.0)               # phase 1: pure read
        yield stream(tgt, dur, 0.5)               # phase 2: read + write
    elif args.mode == "concurrent":
        # The pipeline engine's core assumption: tiers are independent buses.
        # gen reads the CXL device while gen2 reads host DRAM; if the buses are
        # independent, each sustains its solo rate.
        yield stream(tgt, dur, 1.0)
    else:
        raise SystemExit("trace mode not wired yet")
    yield system.gen.createExit(0)


def generator2():
    dur = int(args.duration)
    if args.mode == "concurrent":
        yield system.gen2.createLinear(dur, ranges["host"].start,
                                       ranges["host"].end, args.req_size,
                                       1000, 1000, 100, 0)
    else:
        # Idle in every other mode. An exit here would end the whole
        # simulation at tick 0 and leave stats.txt empty.
        yield system.gen2.createIdle(2 * dur + 1000)
    yield system.gen2.createExit(0)


root = Root(full_system=False, system=system)

m5.instantiate()
# PyTrafficGen only accepts a generator once the C++ objects exist.
system.gen.start(generator())
system.gen2.start(generator2())
print(f"[cmmh_hybrid] mode={args.mode} host={args.host_dram} "
      f"cxl={args.cxl_dram} nand={args.nand} accel={args.accel_mem}")
exit_event = m5.simulate()
# gem5 does not dump stats on its own when a script ends this way, and an empty
# stats.txt looks the same as a run with no traffic.
m5.stats.dump()
print(f"[cmmh_hybrid] exiting @ tick {m5.curTick()}: "
      f"{exit_event.getCause()}")
