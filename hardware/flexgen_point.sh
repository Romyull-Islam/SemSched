#!/bin/bash
# One point of the FlexGen reserve experiment on a GPU server.
#
#   flexgen_point.sh CPU_PCT BUDGET_GB DEPTH OUT
#
# FlexGen keeps CPU_PCT% of the weights pinned in host memory and the rest on
# local disk (OFFLOAD_DIR). Pinned memory is charged to the cgroup, so the cgroup
# limit is the whole host-memory budget: pinned weights plus whatever is left
# for the page cache, which is where disk data can be staged. FlexGen starts
# under a generous limit, because start-up writes the offload files and reads
# the weight files; once it is done ("warmup - generate"), dirty pages are
# flushed and the limit drops to BUDGET_GB. A point whose pinned weights exceed
# the budget is killed and reported as infeasible. DEPTH 0 is plain FlexGen;
# DEPTH k reads k layers ahead into the free part of the budget.
# PROBE=1 records the footprint after start-up and stops; MODEL picks the model;
# GENLEN sets the generated tokens; NUMA_NODE pins threads and memory to a node.
# Cgroup memory during decoding is sampled into OUT.mem.
set -u
CPU=$1; LIMIT=$2; DEPTH=$3; OUT=$(realpath -m "$4")
mkdir -p "$(dirname "$OUT")"
OFFLOAD_DIR=${OFFLOAD_DIR:-/tmp/flexgen_offload}
FLEXGEN=${FLEXGEN:-$HOME/flexgen_exp/FlexLLMGen}
PY=${PY:-$HOME/venvs/flexgen/bin/python}
UNIT=fgpoint-$$
rm -rf "$OFFLOAD_DIR"
cd "$FLEXGEN"
PIN=()
[ -n "${NUMA_NODE:-}" ] && PIN=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
systemd-run --user --scope -q --unit="$UNIT" -p MemoryMax=${START_LIMIT:-48G} -p MemorySwapMax=0 \
    "${PIN[@]}" "$PY" -u "$HOME/tiersched_hw/flexgen_reserve.py" --prefetch-depth "$DEPTH" \
    --model "facebook/${MODEL:-opt-13b}" --path "$HOME/flexgen_exp/opt_weights" \
    --offload-dir "$OFFLOAD_DIR" --gpu-batch-size 16 --prompt-len 512 --gen-len "${GENLEN:-32}" \
    --percent 0 "$CPU" 100 0 100 0 > "$OUT" 2>&1 &
RUN=$!
until grep -q "warmup - generate" "$OUT" 2>/dev/null || ! kill -0 $RUN 2>/dev/null; do sleep 1; done
sync -f "$OFFLOAD_DIR" 2>/dev/null || sync
CG=/sys/fs/cgroup$(systemctl --user show -p ControlGroup --value "$UNIT.scope")
grep -E "^(anon|shmem|kernel|inactive_file) " "$CG/memory.stat" | tr "\n" " " >> "$OUT.ctl"
echo >> "$OUT.ctl"
if [ "${PROBE:-0}" = 1 ]; then          # footprint only: stop after start-up
    systemctl --user stop "$UNIT.scope"; wait $RUN 2>/dev/null; rm -rf "$OFFLOAD_DIR"; exit 0
fi
systemctl --user set-property --runtime "$UNIT.scope" MemoryMax="$LIMIT$([[ $LIMIT =~ ^[0-9.]+$ ]] && echo G)"
echo "budget $LIMIT applied after start-up" >> "$OUT.ctl"
# Memory during decoding, every 2 s: the cgroup total and its parts in bytes,
# and the bytes FlexGen has fetched from storage so far (/proc/PID/io).
( while kill -0 $RUN 2>/dev/null; do
    echo "$(date +%s) current $(cat "$CG/memory.current" 2>/dev/null) $(grep -E "^(anon|shmem|kernel|file|active_file|inactive_file) " "$CG/memory.stat" 2>/dev/null | tr "\n" " ") $(grep "^read_bytes" /proc/$RUN/io 2>/dev/null | tr -d " " | tr ":" " ")" >> "$OUT.mem"
    sleep 2
  done ) &
wait $RUN
echo "exit $?" >> "$OUT.ctl"
rm -rf "$OFFLOAD_DIR"
