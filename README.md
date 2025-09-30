## TextSlinger: Fast and accurate text predictions in Python
This is a Python library for making text predictions using different types of language models.
Current features:
* Predict the distribution over the next character.
* Predict the most likely next words.
* Supports: n-gram language models, subword tokenized large language models (LLMs), and byte tokenized LLMs.

N-gram language models use the KenLM library.
Our provided n-gram models and example scripts make use of a 12-gram n-gram model.
LLMs use the Hugging Face libraries.

## Setting up a Python environment
If you don't have anaconda installed in your user account you'll first need to do that.
See: https://docs.anaconda.com/anaconda/install/linux/

To create an environment:
```
conda create -n textslinger python=3.10
conda activate textslinger
```
If you want to do inference on a GPU via CUDA:
```
conda install pytorch torchvision torchaudio pytorch-cuda cuda mpi4py -c pytorch -c nvidia
```
If you don't need GPU support:
```
conda install pytorch torchvision torchaudio pytorch-cuda cuda mpi4py -c pytorch -c nvidia
```
For the byte tokenized LLM, we need to install the uniformers library. 
Installing this library degrades the transformers library which we then upgrade afterwards:
```
pip install 'git+https://github.com/potamides/uniformers.git#egg=uniformers'
pip install --upgrade transformers
```
Finally install other libraries needed by TextSlinger:
```
pip install kenlm==0.1 --global-option="--max_order=12"
pip install py-cpuinfo 
pip install rbloom bitsandbytes requests nlpaug ipywidgets psutil datasets sentencepiece protobuf evaluate scikit-learn deepspeed accelerate peft pytest wget
```

---
This material is based upon work supported by the NSF under Grant No. IIS-1909089 and IIS-2402876.
