# A mixure of a causal LLM and an n-gram LM
#
# Parameters:
#  $1 - index of the GPU to use
CUDA=${1}

MODEL_NAME=distilgpt2

LM=models/lm_dec19_char_tiny_12gram.arpa
ls -l ${LM}

NAME=mix_gpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} ../lm_eval.py --phrases ${PHRASES} --model 6 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --ngram-lm ${LM} --beam-width 8 --max-completed 32000 --case-simple --mixed-case-context --ppl-file ${NAME}.ppl --ngram-mix 0.2 --use-cuda --fp16 

