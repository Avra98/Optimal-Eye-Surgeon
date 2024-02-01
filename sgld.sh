#!/bin/bash

FILE="./sgld"
mkdir -p $FILE

COUNTER=0
gpu_arr=(3)
LEN=${#gpu_arr[@]}

# kl_values=(1e-9)
ino_values=(0 1 2 3 4 5)  # Assuming you have 14 images
sigma=(0.1)
##vanilla DIP
for ino in "${ino_values[@]}"; do
    for sigma in "${sigma[@]}"; do
        python sgld.py --ino=$ino  --show_every=200 --noise_scale=1e-2 --lr=1e-3 --max_steps=40000\
        --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --sigma=$sigma\
        >> "$FILE"/"sgld_ino${ino}_sigma${sigma}.out" &
        COUNTER=$((COUNTER + 1))
        if [ $((COUNTER % 2)) -eq 0 ]; then
            wait
        fi
     done   
done