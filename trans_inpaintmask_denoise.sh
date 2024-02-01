#!/bin/bash
# denooising 
FILE="./trans_inpaintmask_denoise"
mkdir -p $FILE

COUNTER=0
gpu_arr=(1 2 4 6)
LEN=${#gpu_arr[@]}

# kl_values=(1e-9)  # Assuming this is not used, you can remove or uncomment this line
ino=(9 12)  # Assuming you have 14 images
#ino_trans=(1 2 3 4 5 6 7 8 9 10 11 12 13)   # Assuming you have 14 images
sigma=(0.05 0.1)
prior_sigma=(-0.5)

declare -A lookup_table=( [0]=5 [1]=10 [2]=8 [3]=7 [4]=13 [5]=12 [6]=11 [7]=1 [8]=0 [9]=2 [10]=9 [11]=4 [12]=6 [13]=3 )

for ino in "${ino[@]}"; do
    # Retrieve the corresponding ino_inpaint value from the lookup table
    ino_inpaint=${lookup_table[$ino]}
    for prior_sigma in "${prior_sigma[@]}"; do
        for sigma in "${sigma[@]}"; do
            # Run your python script with the required arguments
            python train_inpaintmask_denoise.py --max_steps=40000 --show_every=200 --prior_sigma=$prior_sigma \
            --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --sigma=$sigma --ino_inpaint=$ino_inpaint \
            >> "$FILE/ino${ino}_ino_inpaint${ino_inpaint}_prior_sigma${prior_sigma}_sigma${sigma}.out" &
            COUNTER=$((COUNTER + 1))
            # Wait for every 20th process
            if [ $((COUNTER % 4)) -eq 0 ]; then
                wait
            fi
        done    
    done       
done

