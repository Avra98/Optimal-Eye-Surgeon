
#!/bin/bash
FILE="./Set14_logfiles/rebtual_l1cent/"
mkdir -p $FILE

COUNTER=0
gpu_arr=(1 3)
LEN=${#gpu_arr[@]}

kl_values=(1e-2)
sparsity_values=(0.05)
ino_values=(2)  # Assuming you have a subset of images
lr_values=(1e-4 1e-3)

# for kl in "${kl_values[@]}"; do
#     for sparsity in "${sparsity_values[@]}"; do

#         for ino in "${ino_values[@]}"; do
#             python dip_mask.py --noise_steps=60000 --mask_opt="det" --sparsity=$sparsity \
#             --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl --ino=$ino \
#             >> "$FILE/sparsity${sparsity}_ino${ino}_kl${kl}_l1cent_p.out" &
#             COUNTER=$((COUNTER + 1))
#             if [ $((COUNTER % 8)) -eq 0 ]; then
#                 wait
#             fi
#         done
#     done
# done


for kl in "${kl_values[@]}"; do
    for sparsity in "${sparsity_values[@]}"; do
        for ino in "${ino_values[@]}"; do
            for lr in "${lr_values[@]}"; do
                python dip_mask.py --noise_steps=90000 --mask_opt="det" --sparsity=$sparsity \
                --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl --ino=$ino --lr=$lr \
                >> "$FILE/sparsity${sparsity}_ino${ino}_kl${kl}_lr${lr}_l1cent_p_longer.out" &
                COUNTER=$((COUNTER + 1))
                if [ $((COUNTER % 2)) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done






#!/bin/bash
# FILE="./face_mask_skip_sparsity"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4 5 6 7)
# LEN=${#gpu_arr[@]}

# #kl_values=(1e-9 1e-6 1e-3 1e-12)
# kl_values=(1e-7 1e-8)
# # Uncomment and define prior_sigma_values if needed
# # prior_sigma_values=(10.0 1.0 0.5 0.0)
# sparsity_values=(0.95 0.8 0.5 0.05 0.03)
# # sparsity_values=(0.0333)
# ino_values=(2)  # Assuming you have 14 images

# for kl in "${kl_values[@]}"; do
#     #for sparsity in "${sparsity_values[@]}"; do
#     for ino in "${ino_values[@]}"; do
#         python dip_mask.py --noise_steps=60000 --mask_opt="det" \
#         --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl --ino=$ino &
#         COUNTER=$((COUNTER + 1))
#         if [ $((COUNTER % 4)) -eq 0 ]; then
#             wait
#         fi
#     done
#     #done
# done
 


# #!/bin/bash
# FILE="./Set14_logfiles/mask_for_deblur"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4)
# LEN=${#gpu_arr[@]}

# kl_values=(1e-9)
# #prior_sigma_values=(0.0 -0.1 -0.2 -0.5 -0.7 -0.8 -1.0 -1.3 -1.5 )
# #prior_sigma_values=(-0.7 -0.8 -1.0 -1.3 -1.5)
# #ino_values=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)  # Assuming you have 14 images
# sparsity_values=(0.05 0.1 0.5)
# # sparsity_values=(0.0333)
# ino_values=(0 1 2 3 4 5 6)  # Assuming you have 14 images

# # First set of experiments with varying kl and prior_sigma
# for kl in "${kl_values[@]}"; do
#     for sparsity in "${sparsity_values[@]}"; do
#         for ino in "${ino_values[@]}"; do
#             python dip_mask_deblur.py --max_steps=40000 --mask_opt="det" --sparsity=$sparsity \
#             --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl  --ino=$ino \
#             >> "$FILE"/"kl${kl}_sparsity${sparsity}_ino${ino}.out" &
#             COUNTER=$((COUNTER + 1))
#             if [ $((COUNTER % 6)) -eq 0 ]; then
#                 wait
#             fi
#         done
#     done
# done 



