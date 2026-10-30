"""
TierSched's placement search for FlexGen on a real server: choose the CPU share
and prefetch depth for a host-memory budget, before any of them is run.

Each candidate (share, depth) is scored by io_replay.py: one decode step's disk
I/O replayed under the candidate's memory room. The inputs are only start-up
measurements: tensor sizes (FG_CALIB), FlexGen's start-up footprint at each
share (memory.stat after start-up) and the decode step with all weights in host
memory. No FlexGen decode run at any candidate is used.

    python plan_search.py --calib C.json --dir D --footprints F.json \\
        --budget-gib 30 --step-s 1.02 --batch 16 --out plan.json
"""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--footprints", required=True, help='{"share": start-up bytes}')
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--step-s", type=float, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--depths", default="0,2,4,8,16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    budget = a.budget_gib * 2**30
    rows = []
    for share, fp in sorted(json.load(open(a.footprints)).items(), key=lambda x: -float(x[0])):
        room = budget - fp
        if room <= 0:
            rows.append({"cpu": float(share), "feasible": False})
            continue
        for k in map(int, a.depths.split(",")):
            out = subprocess.run(
                [os.path.join(HERE, "io_replay.sh"), "--calib", a.calib, "--dir", a.dir,
                 "--cpu", share, "--k", str(k), "--room-bytes", str(room),
                 "--step-s", str(a.step_s)],
                capture_output=True, text=True, check=True).stdout
            r = json.loads(out.strip().splitlines()[-1])
            r["tok_s"] = a.batch / r["step_s"]
            r["feasible"] = True
            rows.append(r)
            print(f"share {share:>4}  K={k:<3} room {room / 1e9:5.1f} GB  "
                  f"predicted {r['tok_s']:.2f} tok/s", flush=True)
    ok = [r for r in rows if r["feasible"]]
    best = min(ok, key=lambda r: r["step_s"])
    plain = min((r for r in ok if r["k"] == 0), key=lambda r: r["step_s"])
    json.dump({"choice": {"cpu": best["cpu"], "k": best["k"]},
               "plain_choice": {"cpu": plain["cpu"]},
               "budget_gib": a.budget_gib, "candidates": rows}, open(a.out, "w"), indent=1)
    print(f"choice: share {best['cpu']:g}% K={best['k']}  "
          f"(plain FlexGen's best by the same replay: share {plain['cpu']:g}%)")


if __name__ == "__main__":
    main()
