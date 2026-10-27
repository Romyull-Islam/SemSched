#!/bin/bash
# Rerun the noisy points of the overnight FlexGen sweep.
#
#   cd ~/semsched_hw && nohup ./flexgen_retry.sh > flexgen_retry.out 2>&1 &
#
# Waits for flexgen_overnight.sh to finish, then reruns every run in
# fg_night/results.txt that was noisy. A run is noisy when another user's process
# on its GPU pair exceeded 5% compute (EXCLUDED) or the host's 1-minute load
# average went above LOAD_MAX (half of the 256 cores). The rule was fixed before
# any rerun was seen. Each rerun first waits for a quiet pair and a quiet host,
# and is tried up to 3 times; every attempt, clean or not, goes to
# fg_night/results_retry.txt.
#
#   ./flexgen_retry.sh JOBS.txt        # path relative to this directory
#
# runs the jobs listed in JOBS.txt instead, one per line in the results format
# ("tag model=... cpu=... budget=... k=..."), under the same rule, and writes to
# fg_night/results_JOBS.txt. A job with probe=1 only starts FlexGen, records the
# start-up calibration (flexgen_reserve.py FG_CALIB) in fg_night/TAG.calib.json
# and stops. The budget may carry a unit (51020M); a bare number is GB.
set -u
cd "$(dirname "$0")"
OUTDIR=fg_night
JOBS=${1:-}
RES=$OUTDIR/results_retry.txt
[ -n "$JOBS" ] && RES=$OUTDIR/results_$(basename "$JOBS" .txt).txt
LOAD_MAX=${LOAD_MAX:-128}
PAIRS=("0 1 3" "2 3 1" "4 5 7" "6 7 5")

others_on() {   # other users' processes with compute activity above 5% on GPUs $1 and $2
    local n=0
    while read -r gpu pid sm _; do
        { [ "$gpu" = "$1" ] || [ "$gpu" = "$2" ]; } || continue
        [ "$sm" = "-" ] && continue
        [ "$sm" -gt 5 ] 2>/dev/null || continue
        # stat on /proc, not ps: ps can block on a process stuck in the kernel
        [ "$(timeout 5 stat -c %U /proc/$pid 2>/dev/null)" != "$USER" ] && n=$((n + 1))
    done < <(timeout 20 nvidia-smi pmon -c 1 -s u 2>/dev/null | grep -v '^#')
    echo $n
}

load() { cut -d' ' -f1 /proc/loadavg; }

pick_quiet() {   # first quiet pair on a quiet host; wait up to 6 h
    for _ in $(seq 72); do
        if awk -v l="$(load)" -v m="$LOAD_MAX" 'BEGIN{exit !(l < m)}'; then
            for p in "${PAIRS[@]}"; do
                set -- $p
                [ "$(others_on $1 $2)" = 0 ] && { echo "$p"; return; }
            done
        fi
        sleep 300
    done
    echo ""
}

attempt() {   # MODEL CPU BUDGET K TAG; prints "clean" or "noisy"
    local model=$1 cpu=$2 budget=$3 k=$4 tag=$5 pair g p node log mon bad hot lim
    pair=$(pick_quiet); [ -z "$pair" ] && { echo "$tag no quiet pair" >> $RES; echo noisy; return; }
    set -- $pair; g=$1; p=$2; node=$3
    log=$OUTDIR/$tag.log; mon=$log.mon; : > "$mon"
    ( while true; do echo "$(date +%s) $(load) others=$(others_on $g $p)" >> "$mon"; sleep 5; done ) &
    local mpid=$!
    lim=48G; [ "$model" = opt-30b ] && lim=96G
    START_LIMIT=$lim PROBE=${probe:-0} FG_CALIB=${probe:+$PWD/$OUTDIR/$tag.calib.json} MODEL=$model NUMA_NODE=$node CUDA_VISIBLE_DEVICES=$g \
        timeout 3600 ./flexgen_point.sh "$cpu" "$budget" "$k" "$log" >/dev/null 2>&1
    kill $mpid 2>/dev/null
    bad=$(awk '{split($3,a,"="); if (a[2]>0) n++} END{print n+0}' "$mon")
    hot=$(awk -v m="$LOAD_MAX" '$2 >= m {n++} END{print n+0}' "$mon")
    echo "$tag model=$model cpu=$cpu budget=$budget k=$k gpu=$g $(grep -o 'decode throughput: [0-9.]*' "$log" | tail -1) $(tail -1 "$log.ctl") load_max=$(sort -k2 -n "$mon" | tail -1 | cut -d' ' -f2) $([ "$bad" -gt 0 ] && echo EXCLUDED) $([ "$hot" -gt 0 ] && echo HOSTLOAD)" >> $RES
    [ "$bad" = 0 ] && [ "$hot" = 0 ] && echo clean || echo noisy
}

if [ -n "$JOBS" ]; then
    LIST=$JOBS; SUFFIX=_try
else
until grep -q "^all done" $OUTDIR/results.txt 2>/dev/null; do sleep 300; done
LIST=$OUTDIR/noisy.txt; SUFFIX=_retry
awk -v m="$LOAD_MAX" '/model=/ {
    split($0, f, " "); lm = 0
    for (i in f) if (f[i] ~ /^load_max=/) { sub("load_max=", "", f[i]); lm = f[i] + 0 }
    if (lm >= m || $0 ~ /EXCLUDED/) print }' $OUTDIR/results.txt > $OUTDIR/noisy.txt
echo "$(wc -l < $OUTDIR/noisy.txt) noisy runs to redo $(date)" >> $RES
fi

while read -r line <&3; do
    tag=${line%% *}
    get() { echo "$line" | grep -o "\b$1=[^ ]*" | cut -d= -f2; }
    probe=$(get probe)
    for try in 1 2 3; do
        [ "$(attempt "$(get model)" "$(get cpu)" "$(get budget)" "$(get k)" "$tag$SUFFIX$try" < /dev/null)" = clean ] && break
    done
done 3< $LIST
echo "retry done $(date)" >> $RES
