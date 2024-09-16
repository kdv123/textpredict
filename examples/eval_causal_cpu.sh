# Causal LLM using a pretrained Hugging Face model without fine-tuning
# Note: If you wanted to evaluate your own local fine-tuned model, you would specify --model-dir

MODEL_NAME=distilgpt2
NAME=causal_cpu

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 4 --verbose 2 --add-char "'" --model-name ${MODEL_NAME} --beam-width 8 --max-completed 32000 --case-simple --mixed-case-context --ppl-file ${NAME}.ppl

