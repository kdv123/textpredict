# Keystroke savings using a byte LLM to predict next words
#
# Parameters:                                                                                                                                                                
#  $1 - index of the GPU to use
#  $2 - nbest list size (e.g. 5)
#  $3 - beam width (e.g. 2.0)

CUDA=${1}
NBEST=${2}
BEAM=${3}

# RESULT: primus, A100 GPU, 4/22/25
# % ks_byte_gpu.sh 0 5 2.0
#  TRUNCATED: 0
#  TIME: 375.36
#  SECS/PRED: 0.1718
#  FINAL KS: 50.1536

# RESULT: cheetah, 2080Ti GPU, 4/22/25:

# % ks_byte_gpu.sh 0 5 2.0
#  TRUNCATED: 0
#  TIME: 423.73
#  SECS/PRED: 0.2003
#  FINAL KS: 50.1299

# % ks_byte_gpu.sh 0 1 2.0
#  TRUNCATED: 0
#  TIME: 289.50
#  SECS/PRED: 0.1044
#  FINAL KS: 34.7508

# % ks_byte_gpu.sh 0 5 4.0
#  TRUNCATED: 0
#  TIME: 563.57
#  SECS/PRED: 0.2745
#  FINAL KS: 51.5710

MODEL_NAME=nllg/bygpt5-small-en
PHRASES=comm_case_dev.txt
wc -l -w ${PHRASES}

CUDA_VISIBLE_DEVICES=${CUDA} ../keystroke_savings.py --phrases ${PHRASES} --model-name ${MODEL_NAME} --byte --case-simple --nbest ${NBEST} --beam ${BEAM} --lower --strip --use-cuda --fp16

