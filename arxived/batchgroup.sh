#!/bin/bash

## SGD with noise-level 0.35 
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=1
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=2

# Repeat with 0.45 sigma 

## Do for al images 1-20

# devs=(0 2 4 6 7)
COUNTER=0
devs=(0 2 4)
OUTDIR="./groupdip_logs"
mkdir -p $OUTDIR
LEN=${#devs[@]}

for sigma in 0.1 0.05;
do
    # for ino in {0..19};
    # for training_size in {}
    for lr in 0.1 0.01 0.05;
    do 
        # for opt in "${strings[@]}"; 
        # i_dev=$(( $ino % ${#devs[@]} ))
        # dev_id=${devs[$i_dev]}
        dev_id=${devs[$((COUNTER % LEN))]}

        for opt in "SGD" "SAM";
        # for opt in "SAM";
        do
                                            # --ino=$ino \
            python deep_hess_batch.py --device_id=$dev_id \
                                            --optim=$opt \
                                            --reg=0.1 \
                                            --training_size=50 \
                                            --lr=$lr \
                                            --sigma=$sigma \
            >> "$OUTDIR"/"groupdip_${opt}_lr${lr}_sigma${sigma}_size50_reg0.1.out" &
            
            COUNTER=$((COUNTER + 1))

            if [ $((COUNTER % LEN)) -eq 0 ]; then
                wait
            fi
            done
        done
    done
        

## Do with reg=0.1

# Define the command to be executed
# COMMAND="python deep_hess.py --device_id "

# # Define the parameter values
# PARAMETERS=("value1" "value2" "value3")

# # Define the maximum number of parallel processes
# MAX_PARALLEL=3

# # Function to run a command with a given parameter
# run_command() {
#   local parameter="$1"
#   echo "Running command with parameter: $parameter"
#   $COMMAND "$parameter"
# }

# # Loop through the parameter values
# for parameter in "${PARAMETERS[@]}"; do
#   # Check the number of running processes
#   while [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; do
#     sleep 1
#   done

#   # Run the command in the background
#   run_command "$parameter" &
# done

# # Wait for all background processes to finish
# wait
