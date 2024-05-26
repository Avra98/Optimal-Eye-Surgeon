#!/bin/bash

# Define the source base directory
src_base_dir="/egr/research-slim/ghoshavr/DIP_quant/data/denoising/Set14/mask"

# Define the target base directory
target_base_dir="/egr/research-slim/ghoshavr/DIP_quant/data/denoising/Set14/mask"

# Array of names in the desired order
names=("Pepper" "Foreman" "Flowers" "Comic" "Lena" "Barbara" "Monarch" "Baboon" "Ppt3" "Coastguard" "Bridge" "Zebra" "Face" "Man")

# Loop through each index and copy the contents
for i in "${!names[@]}"; do
  src_dir="$src_base_dir/$i/*"
  target_dir="$target_base_dir/${names[$i]}"
  mkdir -p "$target_dir"
  cp -r $src_dir "$target_dir"
  echo "Copied contents of $i to ${names[$i]}"
done
