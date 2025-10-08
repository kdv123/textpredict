# Keystroke savings using a byte LLM to predict next words

# RESULTS: m1 macbook, 4/22/25
# % ks_byte_cpu.sh
#  TRUNCATED: 0
#  TIME: 9452.81
#  SECS/PRED: 4.4775
#  FINAL KS: 50.1299

MODEL_NAME=nllg/bygpt5-small-en
PHRASES=comm_case_dev.txt
wc -l -w ${PHRASES}

../keystroke_savings.py --phrases ${PHRASES} --model-name ${MODEL_NAME} --byte --case-simple --nbest 5 --beam 2.0 --strip-symbols --predict-lower