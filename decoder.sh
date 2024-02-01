
#!/bin/bash

FILE="./face_logfiles/decoder"
mkdir -p $FILE

COUNTER=0
gpu_arr=(4 5 6)
LEN=${#gpu_arr[@]}

# kl_values=(1e-9)
ino_values=(0 2 3 4)  # Assuming you have 14 images
k=(4)
sigma=(0.1)

##vanilla DIP
for ino in "${ino_values[@]}"; do
    for k in "${k[@]}"; do
        for sigma in "${sigma[@]}"; do
            python vanilla_decoder.py --max_steps=40000 --lr=1e-2  --show_every=200 \
            --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --k=$k --sigma=$sigma\
            >> "$FILE"/"decoder_k${k}_ino${ino}_sigma${sigma}.out" &
            COUNTER=$((COUNTER + 1))
            if [ $((COUNTER % 3)) -eq 0 ]; then
                wait
            fi
        done    
    done    
done


# FILE="./decoder_inpaint"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(1 2 4)
# LEN=${#gpu_arr[@]}

# # Assuming you have 14 images
# ino_values=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
# k_values=(4 5)
# p_values=(0.5 0.8)

# # Vanilla DIP
# for ino in "${ino_values[@]}"; do
#     for k in "${k_values[@]}"; do
#         for p in "${p_values[@]}"; do
#             python decoder_inpaint.py --max_steps=40000 --lr=1e-3  --show_every=200 \
#             --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --k=$k --p=$p \
#             >> "$FILE"/"decoder_k${k}_ino${ino}_p${p}.out" &
#             COUNTER=$((COUNTER + 1))
#             if [ $((COUNTER % 9)) -eq 0 ]; then
#                 wait
#             fi
#         done    
#     done    
# done



