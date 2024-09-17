# A mixure of a causal byte LLM and an n-gram LM

MODEL_NAME=nllg/bygpt5-small-en

LM=models/lm_dec19_char_tiny_12gram.arpa
ls -l ${LM}

NAME=mix_byte_cpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 9 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --ngram-lm ${LM} --case-simple --mixed-case-context --ppl-file ${NAME}.ppl --ngram-mix 0.2
