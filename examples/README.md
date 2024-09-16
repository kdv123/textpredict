Various example scripts for testing out the scripts in this repo.

Files:
    comm_case_dev.txt           COMM set of sentences written in response to hypothetical communication situations: https://www.tandfonline.com/doi/abs/10.1080/07434619912331278625
    filter_comm.sh              Filter the COMM set to restrict to lowercase sentences just with a-z, apostrophe, and space.
    download_ngram.sh           Download some n-gram models for use in testing.

**** Some preliminary preparation ****
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

**** Evaluate using a causal LLM ****
You can evaluate on a CPU, but it will be somewhat slower:
% eval_causal.sh
