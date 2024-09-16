# Causal LLM using a pretrained Hugging Face model without fine-tuning
NAME=byte_cpu

MODEL_NAME=nllg/bygpt5-small-en

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 7 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --case-simple --mixed-case-context --ppl-file ${NAME}.ppl --ppl-file ${NAME}.ppl
