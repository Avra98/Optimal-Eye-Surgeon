#!/bin/bash

## SGD with noise-level 0.35 
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=1
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=2

# Repeat with 0.45 sigma 

## Do for al images 1-20

# devs=(0 2 4 6 7)
COUNTER=0

# CHANGED BASED ON GPU AVAILABILITY
devs=(0 6)
OUTDIR="./result/Urban100/vanilla/train"
mkdir -p $OUTDIR
LEN=${#devs[@]}

# fixed parameters
sigma=0.1
reg=0.1

for ino in 1 2;
do
    # for decay in 0.001 0.005;
    for decay in 0.005;
    do
        # for beta in 0 0.5 0.8;
        for beta in 0.8;
        do
            # for training_size in {}
            # for lr in 0.1 0.01 0.05;
            for lr in 0.01;
            do 
                # for opt in "${strings[@]}"; 
                # i_dev=$(( $ino % ${#devs[@]} ))
                # dev_id=${devs[$i_dev]}

                for opt in SGD;
                do
                    OUTFILE="$OUTDIR/$ino/$opt(sigma=$sigma,lr=$lr,decay=$decay,beta=$beta,reg=$reg)"
                    mkdir -p $OUTFILE

                    dev_id=${devs[$((COUNTER % LEN))]}
                    echo "Running ino $ino using $opt on dev$dev_id (sigma=$sigma,lr=$lr,decay=$decay,beta=$beta,reg=$reg)"
                                                    # --ino=$ino \
                    python deep_hess.py --device_id=$dev_id \
                                                    --ino=$ino \
                                                    --optim=$opt \
                                                    --sigma=$sigma \
                                                    --max_steps=20000 \
                                                    --lr=$lr \
                                                    --beta=$beta \
                                                    --decay=$decay \
                                                    --reg=$reg \
                    > "$OUTDIR"/"${ino}/${opt}(sigma=${sigma},lr=${lr},decay=${decay},beta=${beta},reg=${reg})/log.out" &
                    
                    COUNTER=$((COUNTER + 1))

                    if [ $((COUNTER % LEN)) -eq 0 ]; then
                        wait
                    fi
                done
            done
        done
    done
done