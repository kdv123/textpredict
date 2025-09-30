# Keystroke savings using a n-gram to predict next words

LM=models/lm_dec19_char_tiny_12gram.arpa
PHRASES=comm_case_dev.txt
echo PHRASES:
wc -l -w ${PHRASES}

../keystroke_savings.py --phrases ${PHRASES} --ngram --ngram-lm ${LM} --case-simple --nbest 5 --beam 2.0 --strip-symbols --predict-lower