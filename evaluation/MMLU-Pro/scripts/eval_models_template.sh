#!/bin/bash
set -ex

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
selected_subjects="all"
gpu_util=0.8

# ================need to modify=======================
export CUDA_VISIBLE_DEVICES=0,1,2,3
models=(
    "/your_ckpt_1"
    "/your_ckpt_2"
    "/your_ckpt_3"
)
# ================need to modify=======================

for model in "${models[@]}"; do
    if [ ! -d "$model" ]; then
        echo "Model path $model does not exist"
        exit 1
    fi
    echo "Evaluating model: $model"
    python evaluate_from_local.py \
        --selected_subjects "$selected_subjects" \
        --save_dir "$save_dir" \
        --model "$model" \
        --global_record_file "$global_record_file" \
        --gpu_util "$gpu_util"
done
