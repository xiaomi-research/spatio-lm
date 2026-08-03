#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJ_ROOT=$(realpath $(dirname $0)/../../)
TRAIN_DATA_ROOT=${PROJ_ROOT}/data/train-vlm/vlm-3d/

export VIDEO_SEGMENTS=12
cvlm3d sft3d \
  --model ${PROJ_ROOT}/data/ckpts/custom/InternVL3_5-8B-V3R-DPT \
  --teacher3d ${PROJ_ROOT}/data/ckpts/custom/DA3-Large \
  --use_hf true \
  --train_type full \
  --freeze_parameters_ratio 1 \
  --trainable_parameters_regex "^(language_model.*(vision_3r.*|dpt_head)|condition_proj)" \
  --torch_dtype bfloat16 \
  --dataset \
      ${TRAIN_DATA_ROOT}/VSI-Bench/merged_qa_scannetpp_train.jsonl#100 \
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
