# Using an encoder-decoder model like T5 to complete a span at the end of the context.

MODEL_NAME=google/byt5-small
NAME=seq2seq_cpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 5 --verbose 2 --add-char "'" --model-name ${MODEL_NAME}  --ppl-file ${NAME}.ppl

