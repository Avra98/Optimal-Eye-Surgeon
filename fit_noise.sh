FILE="./fitnoise"
mkdir -p $FILE

COUNTER=0
gpu_arr=(1 2)
LEN=${#gpu_arr[@]}
ino=4 
variance=( 0.1 0.5 1.0 2.0 5.0 10.0 20.0)
net_choice=("sparse")


for variance in "${variance[@]}"; do
    for net_choice in "${net_choice[@]}"; do
        python fit_noise.py --max_steps=40000  --show_every=200 --mask_opt="det" --var="$variance" --kl=1e-9  --device_id=${gpu_arr[$((COUNTER % LEN))]} --net_choice="$net_choice" --prior_sigma=-1.3  --ino=$ino \
        >> "$FILE"/"var${variance}_ino${ino}_net${net_choice}.out" &
        COUNTER=$((COUNTER + 1))
        if [ $((COUNTER % 6)) -eq 0 ]; then
            wait
        fi
    done    
done

