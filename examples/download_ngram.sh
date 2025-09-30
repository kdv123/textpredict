# Obtain some n-gram language models from the OSF repository:
# "Language Models for Augmentative and Alternative Communication (AAC)"
# https://osf.io/ajm7t/

OUT_DIR=models
mkdir -p ${OUT_DIR}
rm -f ${OUT_DIR}/lm_*.gz ${OUT_DIR}/lm_*.arpa ${OUT_DIR}/lm_*.kenlm

# Download the text ARPA language model and a binary KenLM version
wget --output-document=${OUT_DIR}/lm_dec19_char_tiny_12gram.arpa.gz  https://osf.io/mz3db/download
wget --output-document=${OUT_DIR}/lm_dec19_char_tiny_12gram.kenlm.gz https://osf.io/zyj25/download

cd ${OUT_DIR}
gunzip lm_*.gz
cd ..

ls -l ${OUT_DIR}
