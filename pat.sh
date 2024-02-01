


#!/bin/bash
FILE="./face_logfiles/pat"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 4 5 6)
LEN=${#gpu_arr[@]}
percent=(0.2 0.8)
ino_values=(0 2 3 4)

for percent in "${percent[@]}"; do
    for ino in "${ino_values[@]}"; do
        # Determine prune_iters based on the value of percent
        if [ "$percent" == "0.2" ]; then
            prune_iters=14
        elif [ "$percent" == "0.8" ]; then
            prune_iters=3
        fi

        python baseline_pat.py --max_steps=40000  --show_every=200  --ino=$ino --device_id=${gpu_arr[$((COUNTER % LEN))]}  \
        --percent=$percent --prune_iters=$prune_iters \
        >> "$FILE"/"ino${ino}_percent${percent}_prune${prune_iters}.out" &
        COUNTER=$((COUNTER + 1))
        if [ $((COUNTER % 7)) -eq 0 ]; then
            wait
        fi
    done
done



