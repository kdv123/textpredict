# Evaluate on a heavily pruned 12-gram language model trained on AAC like sentences.
LM=models/lm_dec19_char_tiny_12gram.kenlm
ls -l ${LM}

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 3 --verbose 2 --add-char "'" --model-dir ${LM}
