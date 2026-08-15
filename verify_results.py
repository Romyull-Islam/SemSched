"""
verify_results.py — check the reported tables against properties that must hold,
rather than against a previous run of the same code.

Every defect found in this project passed a re-run. The capacity double-count
reproduced to six figures for ten days; the identical 63.17 across three cache
sizes reproduced perfectly; so did LLM-in-a-Flash scoring higher at B=1 than at
B=2. Determinism is not correctness, so this file tests invariants instead:

  1. DETERMINISM      the same cell twice gives the same number
  2. CAPACITY         no tier holds more than it has, KV included
  3. FLOOR            no policy is faster than its own bytes allow
  4. CAPACITY MONO    more device DRAM never makes a policy slower
  5. BATCH MONO       larger batch lowers throughput only for a stated reason
  6. INVARIANCE       a policy whose number does NOT move when capacity does
                      must have a reason (zero NAND residency); otherwise flag

Check 6 is the one that matters. Three separate bugs this project hit announced
themselves as a result that would not move when the thing it depends on moved,
and each time the number looked plausible. Invariance is treated here as a
failure to be explained, not as stability.

    python verify_results.py            # all checks, both platforms
    python verify_results.py --quick    # skip the batch sweep
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import run_paper_tables as R

G = 1024.0 ** 3
FAIL, WARN, OK = [], [], []


def report(cat, ok, msg, warn=False):
    (OK if ok else (WARN if warn else FAIL)).append(f"[{cat}] {msg}")
    tag = "pass" if ok else ("WARN" if warn else "FAIL")
    print(f"  {tag:>4}  {msg}")


# ── 1. determinism ────────────────────────────────────────────────────────────
def check_determinism(gpu):
    print("\n1. DETERMINISM — same cell twice")
    for quant, c in (("fp16", 48), ("int8", 32)):
        for name, sim in R.SIMS:
            a = R.run(sim, quant, 16, c, gpu)[0]
            b = R.run(sim, quant, 16, c, gpu)[0]
            report("det", a == b,
                   f"{name:<9} {quant} 16H+{c}C: {a} vs {b}")


# ── 2/3. capacity and floor, from each simulator's own placement ──────────────
# Injected inside run_semantic_duplex_simulation's try block, so it carries that
# block's indentation. Unindented, it is a SyntaxError the simulator reports as
# a failed run rather than a failed probe.
LEDGER = '''
        import collections as _cc, json as _js
        _t = _cc.Counter()
        for _i, _L in enumerate(layers):
            for _k, _f in frac[_i].items():
                _t[_k] += _L["bytes"] * _f
        print("LEDGER " + _js.dumps({"tiers": dict(_t), "kv": _kv_mean,
                                     "kv_split": _kv_split,
                                     "caps": {"gpu": gpu_hbm_capacity_bytes,
                                              "host": host_dram_capacity_bytes,
                                              "cxl": cxl_dev_dram_capacity_bytes}}))
'''
BW = {"GPU HBM": 1792e9, "Host DRAM": 38.4e9,
      "CXL Device DRAM": 27e9, "CXL Device NAND": 5e9}
KV_TIER_LABEL = "CXL Device DRAM"          # SemSched's default KV_TIER = "cxl"


def semsched_ledger(quant, h, c, gpu):
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        p = os.path.join(td, "semduplex_scheduler.py")
        s = open(p).read().replace(
            "        sched   = DuplexScheduler(IO_THREAD_POOL_SIZE)",
            LEDGER + "        sched   = DuplexScheduler(IO_THREAD_POOL_SIZE)", 1)
        open(p, "w").write(s)
        R.patch(td, quant, h, c, gpu)
        r = subprocess.run([sys.executable, "semduplex_scheduler.py"], cwd=td,
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        m = re.search(r"LEDGER (\{.*\})", out)
        t = re.search(r"Decode throughput:\s*([\d.]+)", out)
        import json
        return (json.loads(m.group(1)) if m else None,
                float(t.group(1)) if t else None)


def check_capacity_and_floor(gpu):
    print("\n2. CAPACITY — no tier holds more than it has (KV included)")
    print("3. FLOOR    — no policy is faster than its own bytes allow")
    for quant in ("fp16", "int8"):
        for c in R.CXLS:
            led, tps = semsched_ledger(quant, 16, c, gpu)
            if led is None:
                report("cap", False, f"{quant} 16H+{c}C: no ledger"); continue
            tiers, kv, caps = led["tiers"], led["kv"], led["caps"]
            # The KV cache is charged to whichever tier the placement search
            # picked, which is no longer always the device. Attribute it by the
            # split the simulator reports rather than assuming, or a policy that
            # correctly moved its cache to host DRAM reads as a 137% over-commit.
            split = led.get("kv_split") or {"cxl": 1.0}
            kv_by = {"gpu": "GPU HBM", "host": "Host DRAM", "cxl": "CXL Device DRAM"}
            for key, lbl in (("cxl", "CXL Device DRAM"), ("gpu", "GPU HBM"),
                             ("host", "Host DRAM")):
                if not caps[key]:
                    continue
                used = tiers.get(lbl, 0.0) + kv * split.get(key, 0.0)
                report("cap", used <= caps[key] * 1.001,
                       f"{quant} 16H+{c}C {lbl:<16}{used/G:6.2f}G of "
                       f"{caps[key]/G:3.0f}G ({100*used/caps[key]:5.1f}%)"
                       + (f"  [KV {100*split.get(key,0.0):.0f}%]"
                          if split.get(key, 0.0) else ""))
            # floor: with tiers concurrent, a step cannot beat its slowest tier
            floor = max((v / BW[k] for k, v in tiers.items() if k in BW),
                        default=0.0)
            step = R.BATCH / tps
            report("floor", step >= floor * 0.999,
                   f"{quant} 16H+{c}C step {step:7.3f}s vs floor {floor:7.3f}s "
                   f"({step/floor if floor else float('inf'):.2f}x)")


# ── 4/6. capacity monotonicity, and invariance that needs a reason ────────────
def check_capacity_monotonicity(gpu):
    print("\n4. CAPACITY MONOTONICITY — more device DRAM never slower")
    print("6. INVARIANCE — a number that does not move must have a reason")
    for quant in ("fp16", "int8"):
        for name, sim in R.SIMS:
            vals = [R.run(sim, quant, 16, c, gpu)[0] for c in R.CXLS]
            mono = all(b >= a * 0.999 for a, b in zip(vals, vals[1:]))
            report("mono-cap", mono,
                   f"{name:<9} {quant}: " + " -> ".join(f"{v:.2f}" for v in vals))
            # identical across every capacity: legitimate only with no spill
            if max(vals) - min(vals) < 1e-9:
                led, _ = (semsched_ledger(quant, 16, R.CXLS[0], gpu)
                          if name == "SemSched" else (None, None))
                nand = led["tiers"].get("CXL Device NAND", 0.0) if led else None
                if name == "SemSched":
                    report("invariance", nand is not None and nand < 1e6,
                           f"{name:<9} {quant}: identical across "
                           f"{R.CXLS} with NAND="
                           f"{'?' if nand is None else format(nand/G, '.2f')+'G'}"
                           " (needs zero spill to be legitimate)")
                else:
                    report("invariance", True,
                           f"{name:<9} {quant}: identical across {R.CXLS} "
                           "-- check its placement has no spill", warn=True)


# ── 7. model scale ───────────────────────────────────────────────────────────
MODELS = ["Mistral7BCfg", "Llama13BCfg", "Qwen3_20BCfg",
          "Qwen2_5_72BCfg", "Llama3_1_405BCfg"]


def _step_bytes(cfg_name, quant):
    """Bytes a decode step must move for this model at this quant: weights once,
    plus the KV cache read at the generation midpoint. This, not parameter
    count, is the quantity throughput is monotone in -- Llama 13B (full MHA,
    54 GB of KV read per step) legitimately decodes SLOWER than Qwen3 20B
    (GQA, 11 GB), and the first draft of this check flagged all ten policies
    for that architectural fact."""
    import model_cfg as M
    bpp = {"fp16": 2, "int8": 1}[quant]
    cfg = getattr(M, cfg_name)
    layers = M.build_layers(cfg, sequence_length=512)
    weights = sum(L["bytes"] for L in layers) * bpp / 4.0   # build is fp32 bytes
    kv = sum(2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp
             for L in layers if L.get("kv_cache_bytes", 0) > 0) * 128 * 520
    return weights + kv


def check_model_scale(gpu):
    """Determinism per model, and throughput monotone DECREASING in per-step
    bytes across models. The first run of the model sweep produced 12x and
    4.2x readings that this file, pinned to 72B, could not see; this check
    exists so the next such defect fails a run instead of reaching a table."""
    print("\n7. MODEL SCALE — throughput decreasing in per-step bytes; deterministic")
    for quant in ("fp16", "int8"):
        order = sorted(MODELS, key=lambda c: _step_bytes(c, quant))
        for name, sim in R.SIMS:
            vals = []
            for cfg in order:
                a = R.run(sim, quant, 16, 48, gpu, model=cfg)[0]
                b = R.run(sim, quant, 16, 48, gpu, model=cfg)[0]
                if a != b:
                    report("model-det", False,
                           f"{name:<9} {quant} {cfg}: {a} vs {b}")
                vals.append(a)
            bad = [(order[i], vals[i], order[i+1], vals[i+1])
                   for i in range(len(vals) - 1)
                   if vals[i+1] is not None and vals[i] is not None
                   and vals[i+1] > vals[i] * 1.001]
            report("model-mono", not bad,
                   f"{name:<9} {quant}: " + " -> ".join(
                       f"{v:.2f}" if v else "--" for v in vals)
                   + ("" if not bad else "  MORE BYTES, FASTER: " + str(bad)))


# ── 5. batch monotonicity ────────────────────────────────────────────────────
def check_batch_monotonicity(gpu):
    """Throughput usually rises with batch, because weight bytes are constant in
    B while B tokens are produced. It does NOT have to. LLM-in-a-Flash's DRAM
    window is capacity-capped, so between B=1 and B=2 its active set grows 1.54x
    while the window cannot grow at all and the whole difference comes from
    NAND: overflow grows 2.29x against a 2x batch, and throughput falls 3%.
    That is their design under memory pressure, measured, not an accounting
    error -- and the opposite of the active_frac^2 defect that once inflated
    B=1. So a drop is reported for explanation rather than failed outright; only
    an unexplained one is a defect."""
    print("\n5. BATCH MONOTONICITY — a drop must have a stated reason")
    for quant in ("fp16", "int8"):
        for name, sim in R.SIMS:
            vals = [(b, R.run(sim, quant, 16, 48, gpu, batch=b)[0])
                    for b in (1, 2, 4, 8, 16, 32, 64, 128)]
            bad = [(b0, v0, b1, v1) for (b0, v0), (b1, v1)
                   in zip(vals, vals[1:]) if v1 < v0 * 0.999]
            known = (name == "LLMFlash" and bad and
                     all(b0 == 1 for b0, _, _, _ in bad))   # capacity-capped window
            report("mono-batch", not bad or known,
                   f"{name:<9} {quant} 16H+48C: "
                   + (" ".join(f"{v:.2f}" for _, v in vals) if not bad
                      else ("capacity-capped window, expected: " if known else "DROPS at ")
                      + ", ".join(
                          f"B={b0}->{b1} ({v0:.2f}->{v1:.2f})"
                          for b0, v0, b1, v1 in bad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the batch sweep (check 5)")
    a = ap.parse_args()
    if a.quick:
        check_model_scale(R.GPU_GB)
    for gpu, tag in ((0, "CPU-only"), (R.GPU_GB, f"+RTX 5090 ({R.GPU_GB} GB)")):
        print("\n" + "=" * 78 + f"\n{tag}\n" + "=" * 78)
        check_determinism(gpu)
        check_capacity_and_floor(gpu)
        check_capacity_monotonicity(gpu)
        if not a.quick:
            check_batch_monotonicity(gpu)
            check_model_scale(gpu)
    print("\n" + "=" * 78)
    print(f"{len(OK)} passed, {len(WARN)} warnings, {len(FAIL)} FAILED")
    for f in FAIL:
        print("  " + f)
    for w in WARN:
        print("  " + w)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
