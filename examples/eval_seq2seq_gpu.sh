# Using an encoder-decoder model like T5 to complete a span at the end of the context.
#
# Parameters:
#  $1 - index of the GPU to use
CUDA=${1}

MODEL_NAME=google/byt5-small
NAME=seq2seq_gpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 5 --verbose 2 --add-char "'" --model-name ${MODEL_NAME}  --ppl-file ${NAME}.ppl --use-cuda --fp16 

