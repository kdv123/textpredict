# Keystroke savings using a byte LLM to predict next words

# Result: m1 macbook, 4/22/25

MODEL_NAME=nllg/bygpt5-small-en
PHRASES=comm_case_dev.txt
wc -l -w ${PHRASES}

../keystroke_savings.py --phrases ${PHRASES} --model-name ${MODEL_NAME} --byte --case-simple --nbest 5 --beam 2.0 --lower --strip