#!/bin/bash

BLENDER="/workspace/blender/blender-4.5.9-linux-x64/blender"
SCRIPT="final_blender_pipeline.py"

# Available GPUs
GPUS=(0 2)

START=1000
END=10000
STEP=1000

for ((f=$START; f<$END; f+=$STEP)); do

    # pick random GPU
    GPU=${GPUS[$RANDOM % ${#GPUS[@]}]}

    echo "[INFO] Running fstart=$f on GPU $GPU"

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
    $BLENDER -b -P $SCRIPT -- --fstart $f > ../logs2/logs_${f}.txt 2>&1 &

    sleep 2  # small delay to avoid spikes
done

wait
echo "All jobs completed."