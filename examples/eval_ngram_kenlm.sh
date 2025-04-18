# Evaluate on a heavily pruned 12-gram language model trained on AAC like sentences.
# This version loads a KenLM binary file.
NAME=ngram_kenlm

LM=models/lm_dec19_char_tiny_12gram.kenlm
ls -l ${LM}

PHRASES=comm_dev_letter.txt
wc -l -w ${PHRASES}

../lm_eval.py --phrases ${PHRASES} --model 3 --verbose 2 --add-char "'" --ngram-lm ${LM} --ppl-file ${NAME}.ppl

