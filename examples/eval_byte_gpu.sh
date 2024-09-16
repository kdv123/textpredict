# Causal LLM using a pretrained Hugging Face model without fine-tuning
#
# This version uses the GPU which speeds up inference.
# Parameters:
#  $1 - index of the GPU to use
CUDA=${1}
NAME=byte_gpu

MODEL_NAME=nllg/bygpt5-small-en

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} ../lm_eval.py --phrases ${PHRASES} --model 7 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --use-cuda --fp16 --case-simple --mixed-case-context  --ppl-file ${NAME}.ppl
