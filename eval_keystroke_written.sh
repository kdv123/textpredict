#!/bin/bash
# Causal LLM using a pretrained Hugging Face model without fine-tuning
# This version uses the GPU which speeds up inference.
# Note: If you wanted to evaluate your own local fine-tuned model, you would specify --model-dir
#
# Parameters:
#  $1 - index of the GPU to use
CUDA=${1}



MODEL_NAME=facebook/opt-350m
NAME=causal_gpu
PHRASES=spoken_dev_letter_small_keystroke.txt
model_path=/mnt/seagate/Keith/Demo/results/soda_finetune_plain
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} python3 keystroke_savings.py --strip --causal --model-dir ${model_path} --phrases ${PHRASES} --model-name ${MODEL_NAME} --beam 8  --case-simple --use-cuda --fp16

