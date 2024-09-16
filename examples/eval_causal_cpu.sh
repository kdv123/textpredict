# Causal LLM using a pretrained Hugging Face model without fine-tuning
MODEL_NAME=distilgpt2

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 4 --verbose 2 --add-char "'" --model-name ${MODEL_NAME}
