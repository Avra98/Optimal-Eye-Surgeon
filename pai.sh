FILE="./Dataset_logfiles/pai"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 4 5 6)
LEN=${#gpu_arr[@]}
ino=(0 2 3 4)
sparse=(0.9 0.95)
sigma=(0.1)
prune_type=("snip_local" "grasp_local" "synflow_local" "rand_local" "mag_global")

## iterate over ino, sparse, prune_type, save the log files with the corresponding names
for ino in "${ino[@]}"; do
    for sparse in "${sparse[@]}"; do
        for prune_type in "${prune_type[@]}"; do
            for sigma in "${sigma[@]}"; do
                python baseline_pai.py --max_steps=40000  --show_every=200  --ino=$ino --device_id=${gpu_arr[$((COUNTER % LEN))]}  \
                --sparse=$sparse --prune_type=$prune_type --sigma=$sigma\
                >> "$FILE"/"ino${ino}_sparse${sparse}_prune${prune_type}_sigma${sigma}.out" &
                COUNTER=$((COUNTER + 1))
                if [ $((COUNTER % 14)) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done    


# for variance in "${variance[@]}"; do
#     for net_choice in "${net_choice[@]}"; do
#         python fit_noise.py --max_steps=40000  --show_every=200 --mask_opt="det" --var="$variance" --kl=1e-9  --device_id=${gpu_arr[$((COUNTER % LEN))]} --net_choice="$net_choice" --prior_sigma=-1.3  --ino=$ino \
#         >> "$FILE"/"var${variance}_ino${ino}_net${net_choice}.out" &
#         COUNTER=$((COUNTER + 1))
#         if [ $((COUNTER % 6)) -eq 0 ]; then
#             wait
#         fi
#     done    
# done