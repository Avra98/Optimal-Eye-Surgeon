
# #!/bin/bash

# FILE="./face_logfiles/vanilla"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4)
# LEN=${#gpu_arr[@]}

# # kl_values=(1e-9)
# ino_values=(0 2 3 4)  # Assuming you have 14 images
# sigma=(0.1)
# ##vanilla DIP
# for ino in "${ino_values[@]}"; do
#     for sigma in "${sigma[@]}"; do
#         python vanilla_dip.py --max_steps=40000 --lr=1e-3  --show_every=200 \
#         --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --sigma=$sigma\
#         >> "$FILE"/"vanilla_ino${ino}_sigma${sigma}.out" &
#         COUNTER=$((COUNTER + 1))
#         if [ $((COUNTER % 5)) -eq 0 ]; then
#             wait
#         fi
#      done   
# done


# ##inpainting
# FILE="./Set14_logfiles/inpaint_denosiing"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4 5)
# LEN=${#gpu_arr[@]}

# # Define array of p values

# # Define array of ino values
# ino_values=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)  # Assuming you have 14 images

# # Vanilla DIP
# for ino in "${ino_values[@]}"; do
#     # Run vanilla_inpaint.py with different parameters in parallel
#     python vanilla_denoise_inpaint.py --max_steps=40000 --lr=1e-3 --show_every=200 \
#     --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino \
#     >> "$FILE"/"denoise_inpaint_ino${ino}.out" &      
#     COUNTER=$((COUNTER + 1))      
#     # Wait after every 12 tasks
#     if [ $((COUNTER % 12)) -eq 0 ]; then
#         wait
#     fi
# done



#!/bin/bash

# FILE="./face_logfiles/sgld"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4 5 6)
# LEN=${#gpu_arr[@]}

# # kl_values=(1e-9)
# ino_values=(0 2 3 4 5)  # Assuming you have 14 images
# sigma=(0.1)
# ##vanilla DIP
# for ino in "${ino_values[@]}"; do
#     for sigma in "${sigma[@]}"; do
#         python sgld.py --max_steps=40000 --lr=1e-2  --show_every=200 \
#         --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --sigma=$sigma\
#         >> "$FILE"/"sgld_ino${ino}_sigma${sigma}.out" &
#         COUNTER=$((COUNTER + 1))
#         if [ $((COUNTER % 3)) -eq 0 ]; then
#             wait
#         fi
#      done   
# done


##inpainting
FILE="./Set14_logfiles/just_salt"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 4)
LEN=${#gpu_arr[@]}

# Define array of p values

# Define array of ino values
ino_values=(0 1 2 3 4 5 6)  # Assuming you have 14 images

# Vanilla DIP
for ino in "${ino_values[@]}"; do
    # Run vanilla_inpaint.py with different parameters in parallel
    python vanilla_salt_pepper_gaussian.py --max_steps=40000 --lr=1e-4 --show_every=200 \
    --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino \
    >> "$FILE"/"vanilla_salt_ino${ino}.out" &      
    COUNTER=$((COUNTER + 1))      
    # Wait after every 12 tasks
    if [ $((COUNTER % 6)) -eq 0 ]; then
        wait
    fi
done