# Evalute the COMM phrase set using a simple uniform language model
NAME=uniform

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 8 --verbose 2 --add-char "'" --ppl-file ${NAME}.ppl

