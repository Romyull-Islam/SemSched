"""
matrix_explore.py — sweep (prefill, batch, quant) and report where SemSched
wins vs loses against LLMFlash. Uses trace_runner.run_one().

Output: trace_workload/matrix_results.csv
        Printed table of SemSched/LLMFlash speedup per cell.
"""
import csv
import os
import sys

# Allow imports from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_runner import run_one, SIM_FILES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Representative prefill lengths (drawn from ShareGPT distribution)
PREFILLS = [50, 500, 2000, 4000]

# Fix decode at 64 tokens (typical chat reply)
DECODE = 64

# Sweep dimensions — full grid like the paper
BATCHES = [1, 16, 128]
QUANTS  = ["fp32", "fp16", "int8", "int4"]
MEMS    = ["16H+32C", "16H+64C", "32H+32C", "32H+64C"]

OUT = os.path.join(REPO, "trace_workload/matrix_full.csv")


def main():
    total = len(PREFILLS) * len(BATCHES) * len(QUANTS) * len(MEMS) * 4
    print(f"Matrix sweep: {len(PREFILLS)} prefills × {len(BATCHES)} batches × "
          f"{len(QUANTS)} quants × {len(MEMS)} mems × 4 sims = {total} runs")
    print(f"Decode fixed at {DECODE}")

    rows = []
    for mem in MEMS:
        for quant in QUANTS:
            for batch in BATCHES:
                for pf in PREFILLS:
                    for sim in SIM_FILES:
                        print(f"  mem={mem} {quant} B={batch:3d} pf={pf:5d} "
                              f"{sim:10s} ...", end="", flush=True)
                        res = run_one(sim, prefill=pf, decode=DECODE,
                                      batch=batch, quant=quant, mem=mem)
                        tps = res.get("Decode_TPS")
                        print(f" {tps}")
                        rows.append({
                            "mem": mem, "quant": quant, "batch": batch,
                            "prefill": pf, "Simulator": sim,
                            "Decode_TPS": tps,
                            "Prefill_TPS": res.get("Prefill_TPS"),
                        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT}")

    # Summary tables: SemSched/LLMFlash ratio per (mem, quant)
    for mem in MEMS:
        print("\n" + "="*72)
        print(f"SemSched / LLMFlash decode-TPS ratio  (Mem={mem}, decode={DECODE})")
        print("="*72)
        for quant in QUANTS:
            print(f"\n {quant.upper():>5s}")
            header = "  prefill | " + " | ".join(f"B={b:<4d}" for b in BATCHES)
            print(header)
            print("  " + "-"*(len(header)-2))
            for pf in PREFILLS:
                cells = []
                for batch in BATCHES:
                    llm = next((r["Decode_TPS"] for r in rows
                               if r["mem"]==mem and r["quant"]==quant
                               and r["batch"]==batch and r["prefill"]==pf
                               and r["Simulator"]=="LLMFlash"), None)
                    sem = next((r["Decode_TPS"] for r in rows
                               if r["mem"]==mem and r["quant"]==quant
                               and r["batch"]==batch and r["prefill"]==pf
                               and r["Simulator"]=="SemSched"), None)
                    if llm and sem and llm > 0:
                        cells.append(f"{sem/llm:5.2f}x")
                    else:
                        cells.append("  -   ")
                print(f"  {pf:6d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
