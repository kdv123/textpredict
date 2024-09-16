Python library for making text predictions using a language model.

**** Setting up a Python environment ****
These instructions are the same as those in the classify-aac which actually does training of models.
But the same environment should also work for evaluation using the models (though you may only need a subset of the packages).
This was tested on cheetah on 9/14/24.

If you don't have anaconda installed in your user account you'll first need to do that.
See: https://docs.anaconda.com/anaconda/install/linux/

% conda create -n aacllm python=3.10
% conda activate aacllm
% conda install pytorch torchvision torchaudio pytorch-cuda cuda mpi4py -c pytorch -c nvidia
% pip install 'git+https://github.com/potamides/uniformers.git#egg=uniformers'
% pip install --upgrade transformers
% pip install kenlm==0.1 --global-option="--max_order=12"
% pip install rbloom bitsandbytes requests nlpaug ipywidgets psutil datasets sentencepiece protobuf evaluate scikit-learn deepspeed accelerate
