#!/bin/bash
# FILE="./Set14_logfiles/eos"
# mkdir -p $FILE

COUNTER=0

optimizers=("SGD" "ADAM")
init_scales=(0.005 0.01 0.1 1.0 5.0)
IMAGES=("pepper" "baboon" "barbara" "lena")  # Assuming these are the image indices you're interested invalu
gpu_arr=(1 2 3 4 5)  # Assuming you have 4 GPUs available
LEN=${#gpu_arr[@]}

for image in "${IMAGES[@]}"; do
    python dip_mask.py --image_name=$image --device=cuda:${gpu_arr[$((COUNTER % LEN))]} &
    COUNTER=$((COUNTER + 1))
    if [ $((COUNTER % 10)) -eq 0 ]; then
        wait
    fi

    # for scale in "${init_scales[@]}"; do
    #     for optim in "${optimizers[@]}"; do
    #         python dip_eos.py --optim=$optim --init_scale=$scale --ino=$ino \
    #         --device_id=${gpu_arr[$((COUNTER % LEN))]}  --show_every=500 \
    #         >> "$FILE"/"${optim}_scale${scale}_ino${ino}.out" &
    #         COUNTER=$((COUNTER + 1))
    #         if [ $((COUNTER % 10)) -eq 0 ]; then
    #             wait
    #         fi
    #     done
    # done
done
