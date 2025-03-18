# Text Prediction Toolkit 
This is a Python library for making text predictions using either an n-gram language model or a large language model (LLM).
The library using KenLM for making inferences from n-gram language models and Hugging Face for inference from LLMs.

## Setting up a Python environment
If you don't have anaconda installed in your user account you'll first need to do that.
See: https://docs.anaconda.com/anaconda/install/linux/

```
conda create -n aacllm python=3.10
conda activate aacllm
conda install pytorch torchvision torchaudio pytorch-cuda cuda mpi4py -c pytorch -c nvidia
pip install 'git+https://github.com/potamides/uniformers.git#egg=uniformers'
pip install --upgrade transformers
pip install kenlm==0.1 --global-option="--max_order=12"
pip install rbloom bitsandbytes requests nlpaug ipywidgets psutil datasets sentencepiece protobuf evaluate scikit-learn deepspeed accelerate peft
```

This material is based upon work supported by the NSF under Grant No. IIS-1909089.
