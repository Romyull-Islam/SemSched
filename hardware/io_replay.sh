#!/bin/bash
# One replay of io_replay.py in its own memory-limited scope, pinned to a NUMA node.
#   NUMA_NODE=3 io_replay.sh --calib C.json --dir D --cpu 60 --k 8 --room-bytes 5.4e9 --step-s 1.02
PIN=()
[ -n "${NUMA_NODE:-}" ] && PIN=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
exec systemd-run --user --scope -q --unit=replay-$$ -p MemorySwapMax=0 \
    "${PIN[@]}" "${PY:-$HOME/venvs/flexgen/bin/python}" "$(dirname "$0")/io_replay.py" replay "$@"
