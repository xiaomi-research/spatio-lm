#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJ_ROOT=$(realpath $(dirname $0)/../../)
TRAIN_DATA_ROOT=${PROJ_ROOT}/data/train-vlm/vlm-3d/

export VIDEO_SEGMENTS=12
swift sft \
  --custom_register_path ${PROJ_ROOT}/cvlm3d/swift/register.py \
  --model ${PROJ_ROOT}/data/ckpts/custom/InternVL3_5-8B-V3R-DAv3 \
  --use_hf true \
  --train_type full \
  --freeze_parameters_ratio 1 \
  --trainable_parameters_regex "^(language_model.*vision_3r.*|condition_proj)" \
  --torch_dtype bfloat16 \
  --dataset \
      ${TRAIN_DATA_ROOT}/VSI-Bench/merged_qa_scannetpp_train.jsonl#50 \
      ${TRAIN_DATA_ROOT}/VSI-590K/swift.images.jsonl#50 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --add_version false \
  --save_strategy epoch \
  --save_only_model true \
  --output_dir ${PROJ_ROOT}/work_dirs/debug \
  --split_dataset_ratio 0 \
  --attn_impl "flash_attention_2" \
  --model_author wujing14 \
  --model_name mi-vla-research
