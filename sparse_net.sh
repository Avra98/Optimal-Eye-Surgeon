
#!/bin/bash

# FILE="./Set14_logfiles/sparsenet_output_sparsity"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(0 1 2 3 4 5 6 7)
# LEN=${#gpu_arr[@]}

# kl_values=(1e-9)
# #prior_sigma_values=(10.0 1.0 0.5 0.0)
# #prior_sigma_values=(-0.5 -0.8 -1.0 -1.3 -1.5 0.0 0.5 1.0 10.0)
# sparsity_values=(0.8 0.5)
# sigma_values=(0.1)
# ino_values=(0 1 2 3 4 5 6 7 8 9 10 11 12 13) 


# for kl in ${kl_values[@]}; do
#     for sparsity in ${sparsity_values[@]}; do
#         for sigma in ${sigma_values[@]}; do
#             for ino in ${ino_values[@]}; do
#                 python train_sparse.py --max_steps=40000   --show_every=200 --mask_opt="det" --sparsity=$sparsity \
#                 --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl  --sigma=$sigma --ino=$ino \
#                 >> "$FILE"/"unet_sparsity${sparsity}_sigma${sigma}_ino${ino}.out" &
#                 COUNTER=$((COUNTER + 1))
#                 if [ $((COUNTER % 16)) -eq 0 ]; then
#                     wait
#                 fi
#             done    
#         done
#     done
# done

#!/bin/bash

# # Define the directory for log files
# FILE_DIR="./Set14_logfiles/gaussian_init_sparsenet_output_sparsity"
# mkdir -p "$FILE_DIR"

# # Initialize counter for GPU allocation
# COUNTER=0
# # Define an array of GPU IDs
# gpu_arr=(0 1 2 3 4 5)
# # Get the length of the GPU array
# LEN=${#gpu_arr[@]}

# # Define parameter arrays
# kl_values=(1e-9)
# sparsity_values=(0.05)
# sigma_values=(0.1)
# ino_values=(1 2 3 4 5)

# # Iterate over each combination of parameters
# for kl in "${kl_values[@]}"; do
#     for sparsity in "${sparsity_values[@]}"; do
#         for sigma in "${sigma_values[@]}"; do
#             for ino in "${ino_values[@]}"; do
#                 # Run the Python script with the current combination of parameters
#                 python train_sparse.py --max_steps=40000 --show_every=200 --mask_opt="det" --sparsity="$sparsity" \
#                 --device_id="${gpu_arr[$((COUNTER % LEN))]}" --kl="$kl" --sigma="$sigma" --ino="$ino" \
#                 >> "$FILE_DIR/unet_sparsity${sparsity}_sigma${sigma}_ino${ino}.out" &

#                 # Increment the counter
#                 COUNTER=$((COUNTER + 1))
#                 # Wait if the counter is a multiple of 16
#                 if [ $((COUNTER % 6)) -eq 0 ]; then
#                     wait
#                 fi
#             done    
#         done
#     done
# done




# #!/bin/bash
# FILE="./sparsedecoder_output"
# mkdir -p $FILE

# COUNTER=0
# gpu_arr=(1 2 3 4)
# LEN=${#gpu_arr[@]}

# kl_values=(1e-9)
# #prior_sigma_values=(-0.5 -0.7 -0.8 -1.0 -1.3)
# #prior_sigma_values=(-0.7 -0.8 -1.0 -1.3 -1.5)
# ino_values=(0 4 5)  # Assuming you have 14 images
# prior_sigma_values=(-1.3 -1.5 -2.0 -2.5)
# sigma_values=(0.1)


# for kl in ${kl_values[@]}; do
#     for prior_sigma in ${prior_sigma_values[@]}; do
#         for sigma in ${sigma_values[@]}; do
#             for ino in ${ino_values[@]}; do
#                 python train_sparse_decoder.py --max_steps=40000   --show_every=200 --mask_opt="det" --prior_sigma=$prior_sigma \
#                 --device_id=${gpu_arr[$((COUNTER % LEN))]} --kl=$kl  --sigma=$sigma --ino=$ino \
#                 >> "$FILE"/"kl${kl}_prior_sigma${prior_sigma}_sigma${sigma}_ino${ino}.out" &
#                 COUNTER=$((COUNTER + 1))
#                 if [ $((COUNTER % 12)) -eq 0 ]; then
#                     wait
#                 fi
#             done    
#         done
#     done
# done


# Define the directory for log files
FILE_DIR="./Set14_logfiles/sparsenet_deblur"
mkdir -p "$FILE_DIR"

# Initialize counter for GPU allocation
COUNTER=0
# Define an array of GPU IDs
gpu_arr=(0 1 2 3 4 5)
# Get the length of the GPU array
LEN=${#gpu_arr[@]}

# Define parameter arrays
kl_values=(1e-9)
sparsity_values=(0.05 0.1 0.5)
ino_values=(0 1 2 3 4 5 6)

# Iterate over each combination of parameters
for kl in "${kl_values[@]}"; do
    for sparsity in "${sparsity_values[@]}"; do
        for ino in "${ino_values[@]}"; do
            # Run the Python script with the current combination of parameters
            python train_sparse_deblur.py --max_steps=40000 --show_every=200  --sparsity="$sparsity" \
            --device_id="${gpu_arr[$((COUNTER % LEN))]}" --kl="$kl" --ino="$ino" \
            >> "$FILE_DIR/unet_sparsity${sparsity}_ino${ino}.out" &

            # Increment the counter
            COUNTER=$((COUNTER + 1))
            # Wait if the counter is a multiple of 16
            if [ $((COUNTER % 12)) -eq 0 ]; then
                wait
            fi
        done    
    done
done
