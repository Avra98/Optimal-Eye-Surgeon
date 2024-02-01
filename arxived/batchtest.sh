#!/bin/bash

## SGD with noise-level 0.35 
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=1
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=2

# Repeat with 0.45 sigma 

## Do for al images 1-20

# devs=(0 2 4 6 7)
COUNTER=0

# CHANGED BASED ON GPU AVAILABILITY
devs=(0 2 4 7)
MODEL="SGD(batchsize=10,training=100,sigma=0.1,lr=0.08,decay=0.0,beta=0.0)"
OUTDIR="./result/Urban100/minibatch/test"
# mkdir -p $OUTDIR
LEN=${#devs[@]}

# fixed parameters
# sigma=0.1
# reg=0.1
# lr=0.08
# decay=0.0
training_size=100
steps=70000

# for ino in 0;
# do
    # for sigma in 0.1 0.05;
for decay in 0.0;
do
    for beta in 0.0;
    do
        # for training_size in {}
        for batch_size in 10 20 50;
        do 
            # for opt in "${strings[@]}"; 
            # i_dev=$(( $ino % ${#devs[@]} ))
            # dev_id=${devs[$i_dev]}

            for opt in SGD;
            do
                OUTFILE="$OUTDIR/$opt(batchsize=$batch_size,training=$training_size,sigma=${sigma},lr=${lr},decay=${decay},beta=${beta})"
                mkdir -p $OUTFILE
                dev_id=${devs[$((COUNTER % LEN))]}
                echo "Training on ${training_size} imgs using $opt on dev$dev_id (batchsize=$batch_size,sigma=$sigma,lr=$lr,decay=$decay,beta=$beta)"
                                                # --ino=$ino \
                python deep_hess_minibatch.py --device_id=$dev_id \
                                                --optim=$opt \
                                                --sigma=$sigma \
                                                --batch_size=$batch_size \
                                                --training_size=$training_size \
                                                --max_steps=$steps \
                                                --lr=$lr \
                                                --beta=$beta \
                                                --decay=$decay \
                                                --reg=$reg \
                > "$OUTFILE/log.out" &
                
                COUNTER=$((COUNTER + 1))

                if [ $((COUNTER % LEN)) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done