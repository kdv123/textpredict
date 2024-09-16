# Obtain some n-gram language models from:
# https://imagineville.org/software/lm/dec19_char/

OUT_DIR=models
mkdir -p ${OUT_DIR}
rm -f ${OUT_DIR}/lm_*.gz
rm -f ${OUT_DIR}/lm_*.arpa
rm -f ${OUT_DIR}/lm_*.kenlm

wget --directory-prefix=${OUT_DIR} http://data.imagineville.org/lm/dec19_char/lm_dec19_char_tiny_12gram.arpa.gz
wget --directory-prefix=${OUT_DIR} http://data.imagineville.org/lm/dec19_char/lm_dec19_char_tiny_12gram.kenlm.gz

cd ${OUT_DIR}
gunzip lm_*.gz
cd ..

ls -l ${OUT_DIR}
