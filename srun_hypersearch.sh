#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set or empty. Exiting."
    exit 1
fi

echo "CUDA VISIBLE DEVICES=$CUDA_VISIBLE_DEVICES"

IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
for gpu in "${devices[@]}"; do
    echo "Starting wandb agent on GPU $gpu"
    WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=luq CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=./ wandb agent "$1" &
done

echo "Background processes started. Waiting for all to finish..."
wait
echo "All processes have finished."