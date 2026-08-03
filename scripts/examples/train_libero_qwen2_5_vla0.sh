#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/e2e-data/embodied-research-data/large_model/huggingface"

PROJ_ROOT=$(realpath $(dirname $0)/../../)
TRAIN_DATA_ROOT=${PROJ_ROOT}/data/train-vlm/vlm-3d/

export VIDEO_SEGMENTS=2
cvlm3d sft \
  --custom_register_path ${PROJ_ROOT}/cvlm3d/swift/register.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --template qwen2_5_vla0 \
  --use_hf true \
  --train_type full \
  --torch_dtype bfloat16 \
  --dataset \
      ${PROJ_ROOT}/data/train-vla/libero-vla0/roboverse_train.jsonl#100 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --remove_unused_columns false \
  --learning_rate 2e-5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --warmup_ratio 0.05 \
  --logging_steps 2 \
  --add_version false \
  --save_strategy epoch \
  --save_only_model true \
  --ddp_find_unused_parameters false \
  --output_dir ${PROJ_ROOT}/work_dirs/debug \
  --split_dataset_ratio 0 \
  --attn_impl "flash_attention_2" \
  --model_author wujing14 \
  --model_name mi-vla-research
