#!/bin/bash
# denooising 
FILE="./face_logfiles/transfer_pat_face_to_set14"
mkdir -p $FILE

COUNTER=0
gpu_arr=(0 1 2 3 4 5 6)
LEN=${#gpu_arr[@]}

# kl_values=(1e-9)  # Assuming this is not used, you can remove or uncomment this line
ino=(0 2 3 4)  # Assuming you have 14 images
ino_trans=(0 1 2 3 4 5 6 7 8 9 10 11 12 13) # Assuming you have 14 images
sigma=(0.1)
trans_type=("pat")
#sparsity_values=(0.069 0.05 0.03 0.02)

for ino in "${ino[@]}"; do
    for ino_trans in "${ino_trans[@]}"; do
        for trans_type in "${trans_type[@]}"; do
            for sigma in "${sigma[@]}"; do
                #for sparsity in "${sparsity_values[@]}"; do
                python transfer.py --max_steps=40000  --show_every=200 --trans_type=$trans_type \
                --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino=$ino --sigma=$sigma --ino_trans=$ino_trans \
                >> "$FILE/transtype_ino${ino}_ino_trans${ino_trans}_transtype${trans_type}_sigma${sigma}.out" &
                COUNTER=$((COUNTER + 1))
                if [ $((COUNTER % 14)) -eq 0 ]; then
                    wait
                fi
                #done    
            done    
        done    
    done   
done

# inpainting 
#!/bin/bash
# FILE="./inpaint_inpaint_transfer"
# mkdir -p ${FILE}

# COUNTER=0
# gpu_arr=(1 2 3 4 6)
# LEN=${#gpu_arr[@]}

# ino_img=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
# prior_sigma=(0.0 -0.1 -0.2 -0.5 -0.7 -0.8 -1.0 -1.3 -1.5)
# p=(0.5)

# for prior_sigma_value in "${prior_sigma[@]}"; do
#     for ino_img_value in "${ino_img[@]}"; do
#         for p_value in "${p[@]}"; do
#             python train_sparse_inpaint.py --max_steps=40000 --show_every=200 --prior_sigma=${prior_sigma_value}\
#             --device_id=${gpu_arr[$((COUNTER % LEN))]} --ino_img=${ino_img_value} --p=${p_value}  \
#             >> "${FILE}/trans_inpaint_prior_sigma${prior_sigma_value}_inpaint${ino_img_value}_p${p_value}.out" &
#             COUNTER=$((COUNTER + 1))
#             if [ $((COUNTER % 20)) -eq 0 ]; then
#                 wait
#             fi
#         done    
#     done     
# done


