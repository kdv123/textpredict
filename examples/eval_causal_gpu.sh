# Causal LLM using a pretrained Hugging Face model without fine-tuning
# This version uses the GPU which speeds up inference.
# Note: If you wanted to evaluate your own local fine-tuned model, you would specify --model-dir
#
# Parameters:
#  $1 - index of the GPU to use
CUDA=${1}

MODEL_NAME=distilgpt2
NAME=causal_gpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} ../lm_eval.py --phrases ${PHRASES} --model 4 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --beam-width 8 --max-completed 32000 --case-simple --mixed-case-context --use-cuda --fp16 --ppl-file ${NAME}.ppl

