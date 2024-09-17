Various example scripts for testing out the scripts in this repo.

Files:
    comm_case_dev.txt           COMM set of sentences written in response to hypothetical communication situations: https://www.tandfonline.com/doi/abs/10.1080/07434619912331278625
    filter_comm.sh              Filter the COMM set to restrict to lowercase sentences just with a-z, apostrophe, and space.
    download_ngram.sh           Download n-gram models for use in testing.
    eval_*.sh                   Evaluate the COMM set using a variety of different models.

**** Some preliminary prep ****
% filter_comm.sh
% download_ngram.sh

**** Evaluating using a uniform language model ****
This model mostly exists for testing or perhaps smoothing of another model.
But you can evaluate its perplexity on the COMM sentences:
% eval_uniform.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.000036 +/- 0.000014 [0.000032, 0.000641]         
95% CI = [0.000007, 0.000065]         
inference time = 0.25        
sum logprob = -5985.60         
sum symbols = 4093         
mean symbol log prob = -1.4624         
mean sentence ppl = 29.0000         
ppl = 29.0000

**** Evaluating using an n-gram language model ****
Evaluate the COMM sentences using a text ARPA format n-gram language model:
% eval_ngram.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.000096 +/- 0.000040 [0.000070, 0.001371]         
95% CI = [0.000016, 0.000175]         
inference time = 0.49        
sum logprob = -1873.96         
sum symbols = 4093         
mean symbol log prob = -0.4578         
mean sentence ppl = 2.9549         
ppl = 2.8698

We can also evaluate using a binary KenLM format n-gram language model.
The advantage is these load a lot faster for large models.
There can be some quantization when creating the KenLM format file.
Thus, the perplexity may differ a bit from the ARPA result.

% eval_ngram_kenlm.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.000086 +/- 0.000009 [0.000070, 0.000261]         
95% CI = [0.000068, 0.000103]         
inference time = 0.45        
sum logprob = -1874.05         
sum symbols = 4093         
mean symbol log prob = -0.4579         
mean sentence ppl = 2.9549         
ppl = 2.8699

**** Evaluate using a causal LLM with subword tokenization ****
You can evaluate on a CPU, but it will be slower.

On cheetah using the CPU:
% eval_causal_cpu.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.210648 +/- 0.104022 [0.053446, 0.944220]         
95% CI = [0.002604, 0.418693]         
inference time = 862.63        
sum logprob = -1708.82         
sum symbols = 4093         
mean symbol log prob = -0.4175         
mean sentence ppl = 2.7506         
ppl = 2.6152
Predict %: inference 84.574

On cheetah using a 2080 Ti GPU:
% eval_causal_gpu.sh 0
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.044924 +/- 0.008152 [0.029218, 0.188850]         
95% CI = [0.028621, 0.061227]         
inference time = 184.30        
sum logprob = -1708.38         
sum symbols = 4093         
mean symbol log prob = -0.4174         
mean sentence ppl = 2.7500         
ppl = 2.6145
Predict %: inference 21.937

**** Evaluate using a causal LLM with byte tokenization ****
The original byte level LLM was ByT5, but this was an encoder-decoder.
It was converted into ByGPT5 which uses just the decoder side: https://github.com/potamides/uniformers
Bear in mind ByGPT5 is a multilingual model compared to some of the other LLMs we've been testing.

On cheetah using the CPU:
% eval_byte_cpu.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.067271 +/- 0.021268 [0.020762, 0.104329]         
95% CI = [0.024735, 0.109806]         
inference time = 275.80        
sum logprob = -2353.58         
sum symbols = 4093         
mean symbol log prob = -0.5750         
mean sentence ppl = 4.2488         
ppl = 3.7586

On cheetah using a 2080 Ti GPU:
% eval_byte_gpu.sh 0
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.005186 +/- 0.002170 [0.004745, 0.142891]         
95% CI = [0.000846, 0.009526]         
inference time = 21.46        
sum logprob = -2352.79         
sum symbols = 4093         
mean symbol log prob = -0.5748         
mean sentence ppl = 4.2464         
ppl = 3.7569

**** Evaluate using a mixture of a causal LLM and an n-gram model ****
This combines the causal subword LLM with a mixture weight 0.8 with an n-gram with a mixture weight of 0.2.

On cheetah using the CPU:
% eval_mix_cpu.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.227256 +/- 0.107243 [0.060032, 1.040433]         
95% CI = [0.012771, 0.441741]         
inference time = 930.55        
sum logprob = -1669.51         
sum symbols = 4093         
mean symbol log prob = -0.4079         
mean sentence ppl = 2.6688         
ppl = 2.5580

On cheetah using a 2080 Ti GPU:
% eval_mix_gpu.sh 0
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.042154 +/- 0.008138 [0.026169, 0.163857]         
95% CI = [0.025878, 0.058430]         
inference time = 172.91        
sum logprob = -1669.25         
sum symbols = 4093         
mean symbol log prob = -0.4078         
mean sentence ppl = 2.6684         
ppl = 2.5576

**** Evaluate using a seq2seq encoder-decoder model ****
This uses the ByT5 byte level model that tries to complete a span between the current context and sentence end token.

On cheetah using a 2080 Ti GPU:
% eval_seq2seq_gpu.sh 0
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 1.527232 +/- 0.642333 [0.866661, 5.057585]         
95% CI = [0.242565, 2.811898]         
inference time = 6251.55        
sum logprob = -2931.41         
sum symbols = 4093         
mean symbol log prob = -0.7162         
mean sentence ppl = 5.8318         
ppl = 5.2024

**** Evaluate using a mixture of causal byte model and n-gram ****

On cheetah using the CPU:
% eval_mix_byte_cpu.sh
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.055172 +/- 0.020200 [0.019734, 0.095852]         
95% CI = [0.014772, 0.095572]         
inference time = 226.24        
sum logprob = -1974.20         
sum symbols = 4093         
mean symbol log prob = -0.4823         
mean sentence ppl = 3.2463         
ppl = 3.0362

On cheetah using a 2080 Ti GPU:
% eval_mix_byte_gpu.sh 0
...
OVERALL         
phrases = 124,         
zero-prob events = 0         
per-symbol prediction time = 0.005111 +/- 0.002392 [0.004749, 0.157505]         
95% CI = [0.000326, 0.009896]         
inference time = 21.17        
sum logprob = -1974.02         
sum symbols = 4093         
mean symbol log prob = -0.4823         
mean sentence ppl = 3.2459         
ppl = 3.0359