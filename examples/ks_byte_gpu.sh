# Keystroke savings using a byte LLM to predict next words
#
# Parameters:                                                                                                                                                                
#  $1 - index of the GPU to use
CUDA=${1}

# Result: primus, A100 GPU, 4/22/25
#  TRUNCATED: 0
#  TIME: 375.36
#  SECS/PRED: 0.1718
#  FINAL KS: 50.1536

# Result: cheetah, 2080Ti GPU, 4/22/25

MODEL_NAME=nllg/bygpt5-small-en
PHRASES=comm_case_dev.txt
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} ../keystroke_savings.py --phrases ${PHRASES} --model-name ${MODEL_NAME} --byte --case-simple --nbest 5 --beam 2.0 --lower --strip --use-cuda --fp16
