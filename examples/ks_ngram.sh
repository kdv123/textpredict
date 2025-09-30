# Keystroke savings using a n-gram to predict next words

# Result: m1 macbook, 4/22/25
# TRUNCATED: 0
# TIME: 13.89
# SECS/PRED: 0.0066
# FINAL KS: 54.8311

LM=models/lm_dec19_char_tiny_12gram.arpa
PHRASES=comm_case_dev.txt
echo PHRASES ${PHRASES}
wc -l -w ${PHRASES}

../keystroke_savings.py --phrases ${PHRASES} --lm ${LM} --case-simple --nbest 5 --beam 2.0 --lower --strip-symbols