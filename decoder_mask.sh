FILE="./decodermask_output"
mkdir -p $FILE

COUNTER=0
gpu_arr=(1 2 3 4)
LEN=${#gpu_arr[@]}

kl_values=(1e-9)
prior_sigma_values=(0.0 -0.5 -1.3  -2.5)
#prior_sigma_values=(-0.7 -0.8 -1.0 -1.3 -1.5)
ino_values=(1 2 3 6 7 8 9 10 11 12 13)  # Assuming you have 14 images


# First set of experiments with varying kl and prior_sigma
for kl in "${kl_values[@]}"; do
    for prior_sigma in "${prior_sigma_values[@]}"; do
        for ino in "${ino_values[@]}"; do
            python decoder_mask.py --noise_steps=80000 --mask_opt="det" --prior_sigma=$prior_sigma \
            --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl  --k=5 --ino=$ino \
            >> "$FILE"/"kl${kl}_prior_sigma${prior_sigma}_ino${ino}_k${k}_decoder.out" &
            COUNTER=$((COUNTER + 1))
            if [ $((COUNTER % 12)) -eq 0 ]; then
                wait
            fi
        done
    done
done
