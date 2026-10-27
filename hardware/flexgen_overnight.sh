#!/bin/bash
# Overnight FlexGen reserve sweeps on a shared DGX A100.
#
# Each run waits for a GPU pair (two GPUs on one PCIe switch) where no other
# user's process is active, pins FlexGen to that pair's NUMA node, and samples
# the pair every 5 s. Exclusion rule, fixed before any result is seen: a run is
# marked EXCLUDED if another user's process on either GPU of the pair exceeds 5%
# compute activity while it runs. Results go to fg_night/results.txt, one line per run.
set -u
cd "$HOME/semsched_hw"
OUTDIR=fg_night; mkdir -p "$OUTDIR"
export OFFLOAD_DIR=/raid2/mislamProject/flexgen_offload GENLEN=64
PAIRS=("0 1 3" "2 3 1" "4 5 7" "6 7 5")          # gpu, partner, NUMA node

others_on() {   # other users' processes with compute activity above 5% on GPUs $1 and $2
    # An idle process that only holds GPU memory does not contend for the PCIe
    # switch, so activity, not presence, is what counts.
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

pick_pair() {   # first pair with no other users; wait up to 6 h
    for _ in $(seq 72); do
        for p in "${PAIRS[@]}"; do
            set -- $p
            [ "$(others_on $1 $2)" = 0 ] && { echo "$p"; return; }
        done
        sleep 300
    done
    echo ""
}

run() {   # MODEL CPU BUDGET K TAG
    local model=$1 cpu=$2 budget=$3 k=$4 tag=$5 pair g p node log mon bad
    pair=$(pick_pair); [ -z "$pair" ] && { echo "$tag no free pair" >> $OUTDIR/results.txt; return; }
    set -- $pair; g=$1; p=$2; node=$3
    log=$OUTDIR/$tag.log; mon=$log.mon; : > "$mon"
    ( while true; do echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) others=$(others_on $g $p)" >> "$mon"; sleep 5; done ) &
    local mpid=$!
    MODEL=$model NUMA_NODE=$node CUDA_VISIBLE_DEVICES=$g timeout 3600 ./flexgen_point.sh "$cpu" "$budget" "$k" "$log" >/dev/null 2>&1
    kill $mpid 2>/dev/null
    bad=$(awk '{split($3,a,"="); if (a[2]>0) n++} END{print n+0}' "$mon")
    echo "$tag model=$model cpu=$cpu budget=$budget k=$k gpu=$g $(grep -o 'decode throughput: [0-9.]*' "$log" | tail -1) $(tail -1 "$log.ctl") load_max=$(sort -k2 -n "$mon" | tail -1 | cut -d' ' -f2) $([ "$bad" -gt 0 ] && echo EXCLUDED)" >> $OUTDIR/results.txt
}

# Phase A: OPT-13B, 30 GB budget (footprint at 63% is 29.6 GB), 4 interleaved repeats.
for rep in 1 2 3 4; do
    for cpu in 63 60 50 40 30 20; do
        for k in 0 8; do run opt-13b $cpu 30 $k a_${cpu}_k${k}_r$rep; done
    done
done

# Phase B: OPT-30B. Probe the footprint at each share; the budget is 0.5 GB above
# the footprint at 60%, which makes 60% the filled setting.
for cpu in 60 50 40 30 20; do
    PROBE=1 START_LIMIT=96G MODEL=opt-30b CUDA_VISIBLE_DEVICES=$(set -- $(pick_pair); echo $1) \
        timeout 3600 ./flexgen_point.sh $cpu 100 0 $OUTDIR/probe30b_$cpu.log >/dev/null 2>&1
done
fp=$(awk '{for(i=1;i<NF;i+=2) if($i=="anon"||$i=="kernel"||$i=="shmem") s+=$(i+1)} END{v=s/1e9+0.5; printf "%d", (v==int(v))?v:int(v)+1}' $OUTDIR/probe30b_60.log.ctl)
echo "opt-30b budget ${fp} GB" >> $OUTDIR/results.txt
for rep in 1 2 3; do
    for cpu in 60 50 40 30 20; do
        for k in 0 8; do START_LIMIT=96G run opt-30b $cpu "$fp" $k b_${cpu}_k${k}_r$rep; done
    done
done
echo "all done $(date)" >> $OUTDIR/results.txt
