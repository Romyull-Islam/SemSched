"""
Build sweep_figs.json, the data behind fig_batch.pdf and fig_models.pdf, from
the output of the harness:

    python run_paper_tables.py --batch-sweep > batch.txt
    python run_paper_tables.py --models      > models.txt
    python parse_sweeps.py batch.txt models.txt
    python plot_sweeps.py

Each value is TierSched's speedup over the best baseline, the harness's ratio
column.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def platform(header):
    return "GPU" if "RTX" in header else "CPU"


def parse_batch(path):
    out, key = {}, None
    for line in open(path):
        if "batch sweep" in line:
            m = re.search(r"(FP16|INT8)\s+16H\+(\d+)C", line)
            key = f"{platform(line)}|{m.group(1)}|{m.group(2)}"
            out[key] = {}
            continue
        m = re.match(r"\s+(\d+)(?:\s+[\d.]+){5}\s+([\d.]+)x", line)
        if key and m:
            out[key][m.group(1)] = float(m.group(2))
    return out


def parse_models(path):
    out, key = {}, None
    for line in open(path):
        m = re.match(r"Model scale, (FP16|INT8) 16H\+48C, (.*), B=", line)
        if m:
            key = f"{platform(m.group(2))}|{m.group(1)}"
            out[key] = {}
            continue
        m = re.match(r"(\S.*?)\s{2,}(?:[\d.]+\s+){5}([\d.]+)x", line)
        if key and m:
            out[key][m.group(1).strip()] = float(m.group(2))
    return out


if __name__ == "__main__":
    batch, models = sys.argv[1], sys.argv[2]
    data = {"batch": parse_batch(batch), "models": parse_models(models)}
    assert len(data["batch"]) == 12 and len(data["models"]) == 4, \
        (len(data["batch"]), len(data["models"]))
    json.dump(data, open(os.path.join(REPO, "sweep_figs.json"), "w"), indent=1)
    print("wrote sweep_figs.json")
