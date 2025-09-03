import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import heapq
from language_model import LanguageModel
from exceptions import InvalidLanguageModelException
from scipy.special import logsumexp
from scipy.special import softmax
import time
from collections import defaultdict
from typing import Final
from peft import AutoPeftModelForCausalLM
import math
import re

class CausalLanguageModel(LanguageModel):
    """Character language model based on a pre-trained causal transformer language model."""

    def __init__(self,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = None,
                 fp16: bool = False,
                 max_completed: int = None,
                 lora: bool = False,
                 lora_path: str = "",
                 batch_size: int = None,
                 predict_lower: bool = True,
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            beam_width         - how many hypotheses to keep during the search, None=off (for character prediction method only)
            fp16               - convert model to fp16 to save memory/compute on CUDA
            max_completed      - stop search once we reach this many completed hypotheses, None=don't prune (for character prediction method only)
            lora_path          - load LoRA adapter from Hugging Face or local directory
            batch_size         - batch size for doing multiple inferences at same time (currently used only in predict_word)
            predict_lower      - if we internally marginalize predictions based on upper and lowercase hypotheses
        """
        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = defaultdict(list)
        # Since subword token ids are integers, use a list instead of a dictionary
        self.index_to_word = []
        self.index_to_word_lower = []
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.max_completed = max_completed
        self.lora_path = lora_path
        self.batch_size = batch_size
        self.predict_lower = predict_lower

        # Hash set versions that we'll create that let us quickly check token IDs against our entire
        # valid set, or in a subset based on a text prefix.
        self.valid_vocab_set = None
        self.vocab_set = defaultdict(set)

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model
        self.model_name = lang_model_name
        self.model_dir = lm_path if lm_path else self.model_name

        # Parameters for the search
        self.beam_width = beam_width

        # Track how much time spent in different parts of the predict function
        self.predict_total_ns = 0
        self.predict_inference_ns = 0

        # Are we a model that automatically inserts a start token that we need to get rid of
        self.drop_first_token = False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        except BaseException:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            if self.lora_path:
                self.model = AutoPeftModelForCausalLM.from_pretrained(self.lora_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            if self.fp16 and self.device == "cuda":
                self.model = self.model.half()
        except:
            raise InvalidLanguageModelException(
                f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

        self.model.eval()
        self.model.to(self.device)
        self.symbol_set_lower = []
        for ch in self.symbol_set:
            self.symbol_set_lower.append(ch.lower())
        self._build_vocab()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        # Loop over all the subword tokens in the LLM
        for i in range(self.vocab_size):
            # Create a map from the subword token integer ID to the mixed and lowercase string versions
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word += word,
            self.index_to_word_lower += word_lower,

            # Check if all the characters in the subword token are in our valid symbol set
            valid = True
            for ch in word_lower:
                if ch not in self.symbol_set_lower:
                    valid = False
                    break

            # If the subword token symbols are all valid, then add it to the list of valid token IDs
            if valid:
                self.valid_vocab += i,
                # Add this token ID to all lists for its valid text prefixes
                for j in range(len(word)):
                    key = word_lower[0:j + 1]
                    self.vocab[key] += i,
                    # Construct set for prefix of the word
                    self.vocab_set[key].add(i)

        # Hash set of the vocab indexes for quick testing if token ID is in our entire valid set
        self.valid_vocab_set = set(self.valid_vocab)

        # When done, self.vocab can be used to map to possible following subword tokens given some text, e.g.:
        # self.vocab["cyclo"] = [47495, 49484]
        # self.index_to_word[self.vocab["cyclo"][0]] = cyclop
        # self.index_to_word[self.vocab["cyclo"][1]] = cyclopedia
        self.drop_first_token = False

        if "deepseek" in self.model_name.lower():
            if self.left_context and len(self.left_context) > 0:
                # This model doesn't seem to have a string we can map, always adds 128000 size of vocab to start of tokens
                print(f"WARNING: DeepSeek doesn't support custom left context! Using blank left context.")
            self.left_context = ""
            self.drop_first_token = True
        else:
            # Get the index we use for the start or end pseudo-word
            if self.left_context == "":
                if "gpt2" in self.model_name:
                    self.left_context = "<|endoftext|>"
                elif "Llama" in self.model_name:
                    self.left_context = "<|begin_of_text|>"
                # Seems to have both sentence start and end tokens: https://docs.mistral.ai/guides/tokenization/
                elif "Mistral" in self.model_name:
                    self.left_context = "<s>"
                else:
                    self.left_context = "</s>"
            # OPT, Llama and Mistral all insert start token
            self.drop_first_token = (self.model_name.startswith("facebook/opt") or
                                     self.model_name.startswith("figmtu/opt") or
                                     "Llama" in self.model_name or
                                     "Mistral" in self.model_name)

        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        print(f"Causal: left_context = '{self.left_context}', left_context_tokens = {self.left_context_tokens}, drop_first_token = {self.drop_first_token}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        # Both OPT and Llama automatically insert a start token which we want to control ourselves
        if len(tokens) > 1 and self.drop_first_token:
            tokens = tokens[1:]

        return tokens

    def _sequence_string(self, sequence: List[int]) -> str:
        """
        Convert a sequence of subword token IDs into a string with each token in ()'s
        :param sequence: List of subword token IDs
        :return: String
        """
        return "".join([f"({self.index_to_word[x]})" for x in sequence])

    def get_all_tokens_text(self):
        """
        Return an array with the text of all subword tokens.
        The array is in order by the integer index into the vocabulary.
        This is mostly just for exploring the tokens in different LLMs.
        :return: Array of subword token text strings.
        """
        result = []
        for i in range(self.vocab_size):
            result.append(self.tokenizer.decode([i]))
        return result

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        assert self.model is not None, "language model does not exist!"
        start_ns = time.time_ns()

        # TODO: add support for prediction of mixed case

        converted_context = "".join(evidence)
        converted_context_lower = converted_context.lower()
        context = converted_context

        # Index in the hypothesis string that is the next character after our context
        target_pos = len(context)

        # For stats purposes track length of the prefix we are extending from space to match
        # prefix_len = target_pos

        # Look for the last space in the context, or -1 if no begin_text in context yet
        pos = context.rfind(" ")
        tokens = []
        tokens.extend(self.left_context_tokens)
        if pos >= 0:
            truncated_context = context[0:pos]
            tokens.extend(self._encode(truncated_context))

        # Constant indexes for use with the hypotheses tuples
        LOGP: Final[int] = 0
        SEQ: Final[int] = 1
        LEN: Final[int] = 2

        # Our starting hypothesis that we'll be extending.
        # Format is (log likelihood, token id sequence, text length).
        # Note: we only include tokens after any in left context.
        start_length = 0
        for x in tokens[len(self.left_context_tokens):]:
            start_length += len(self.index_to_word_lower[x])
        current_hypos = [(0.0, tokens, start_length)]

        # We use a priority queue to track the top hypotheses during the beam search.
        # For a beam of 8, empirical testing showed this was about the same amount
        # of time as a simpler list that used a linear search to replace when full.
        heapq.heapify(current_hypos)

        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = defaultdict(list)

        # Add new extended hypotheses to this heap
        next_hypos = []

        # Tracks count of completed hypotheses
        completed = 0

        # Used to signal to while loop to stop the search
        done = False

        # Start a beam search forward from the backed off token sequence.
        # Each iteration of this while loop extends hypotheses by all valid tokens.
        # We only keep at most self.beam_width hypotheses in the valid heap.
        # Stop extending search once we reach our max completed target.
        while len(current_hypos) > 0 and not done:
            # We'll explore hypothesis in order from most probable to least.
            # This has little impact on how long it takes since this is only sorting a small number of things.
            # But it is important with max_completed pruning since we want to bias for completing high probability things.
            current_hypos.sort(reverse=True)

            # Work on the hypotheses from the last round of extension.
            # Create the torch tensor for the inference with a row for each hypothesis.
            tokens_tensor = torch.tensor([x[SEQ] for x in current_hypos]).reshape(len(current_hypos), -1).to(self.device)

            before_inference_ns = time.time_ns()
            # Ask the LLM to predict tokens that come after our current set of hypotheses
            with torch.no_grad():
                # Compute the probabilities from the logits
                log_probs = torch.log_softmax(self.model(tokens_tensor).logits[:, -1, :], dim=1)

                # Create a column vector where each row is that hypothesis' current likelihood.
                add_tensor = torch.tensor([x[LOGP] for x in current_hypos]).reshape(-1, 1).to(self.device)

                # Add the current likelihoods with each subtoken's probability.
                new_log_probs = torch.add(log_probs, add_tensor)

                # Use the GPU to sort the tokens by probability, this allows use to do better max completed pruning in the search.
                # Move both back to the CPU and convert to numpy since this makes it a lot faster to access for some reason.
                sorted_args = torch.argsort(new_log_probs, descending=True, dim=1).detach().cpu().numpy()
                new_log_probs = new_log_probs.detach().cpu().numpy()

            self.predict_inference_ns += time.time_ns() - before_inference_ns

            # Loop over all the hypotheses from the batch
            for current_index, current in enumerate(current_hypos):
                vocab_set = set()
                extra_vocab_set = set()

                # Extending this hypothesis must match the remaining text
                remaining_context = converted_context_lower[current[LEN]:]
                if len(remaining_context) == 0:
                    # There is no remaining context thus all subword tokens that are valid under our symbol set
                    # should be considered when computing the probability of the next character.
                    #vocab = self.valid_vocab
                    vocab_set = self.valid_vocab_set
                else:
                    if remaining_context in self.vocab:
                        # We have a list of subword tokens that match the remaining text.
                        # They could be the same length as the remaining text or longer and have the remaining text as a prefix.
                        #vocab = self.vocab[remaining_context]
                        vocab_set = self.vocab_set[remaining_context]

                    # We may need to use a subword token that doesn't completely consume the remaining text.
                    # Find these by tokenizing all possible lengths of text starting from the current position.
                    for i in range(1, len(remaining_context)):
                        tokenization = self._encode(context[current[LEN]:current[LEN] + i])
                        # Ignore tokenizations involving multiple tokens since they involve an ID we would have already added.
                        if len(tokenization) == 1:
                            #extra_vocab += tokenization[0],
                            extra_vocab_set.add(tokenization[0])

                # The below for-loop takes the most time (other than the GPU inference and sort).
                #
                # Tuning notes:
                #  - With a beam of 8 and max completed of 32,000, getting around 5x speedup on written dev set.
                #  - This results in a PPL increase of 0.0025 versus old results using only beam of >= 8.
                #  - Pruning based on log probability difference and based on minimum number of hypotheses per symbol in alphabet did worse.
                #  - Code for these other pruning methods was removed.
                #  - Pruning to  top sorted tokens didn't speed it compared to max completed.
                #  - Currently 65% of time spent in GPU code block.
                # Possible ways to make it faster:
                #  - Search in parallel, Python 3.13 and threads without GIL?
                #  - Do matrix multiply with mask matrix to zero out invalid token IDs

                # Explore the token predictions in order from most to least probable.
                for token_id in sorted_args[current_index]:
                    if token_id in vocab_set or token_id in extra_vocab_set:
                        # For a hypothesis to finish it must extend beyond the existing typed context
                        subword_len = len(self.index_to_word_lower[token_id])
                        if (current[LEN] + subword_len) > len(context):
                            # Add this likelihood to the list for the character at the prediction position.
                            # Tracking the list and doing logsumpexp later was faster than doing it for each add.
                            char_to_log_probs[self.index_to_word_lower[token_id][target_pos - current[LEN]]] += new_log_probs[current_index][token_id],
                            completed += 1
                            # Exit this hypothesis if we reach global limit or limit for a single hypothesis.
                            if self.max_completed and completed >= self.max_completed:
                                done = True
                                break
                        elif not self.beam_width or len(next_hypos) < self.beam_width:
                            # If we are under the beam limit then just add it
                            heapq.heappush(next_hypos,
                                           (new_log_probs[current_index][token_id],
                                            current[SEQ] + [token_id],
                                            current[LEN] + subword_len))
                        elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                            # Or replace the worst hypotheses with the new one
                            heapq.heappushpop(next_hypos,
                                              (new_log_probs[current_index][token_id],
                                               current[SEQ] + [token_id],
                                               current[LEN] + subword_len))
                # Break out of the for loop over hypotheses and while loop if we reach our max completed goal
                if self.max_completed and completed >= self.max_completed:
                    done = True
                    break

            # Swap in the extended set as the new current working set
            current_hypos = next_hypos
            next_hypos = []

        # Parallel array to symbol_set for storing the marginals
        char_probs = []
        for ch in self.symbol_set_lower:
            target_ch = ch

            # Handle cases when symbols are never seen
            if target_ch in char_to_log_probs:
                char_probs += logsumexp(char_to_log_probs[target_ch]),
            else:
                char_probs += float("-inf"),

        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

        next_char_pred = {}
        for i, ch in enumerate(self.symbol_set_lower):
            next_char_pred[ch] = char_probs[i]

        end_ns = time.time_ns()
        self.predict_total_ns += end_ns - start_ns

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def predict_words(self,
                      left_context: str,
                      word_end_symbols: List[str] = None,
                      nbest: int = None,
                      beam_logp_best: float = None,
                      beam_search_max: int = None,
                      max_word_len: int = None,
                      max_word_hypotheses: int = None,
                      return_log_probs = False) -> List:
        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to space symbol, e.g. <sp>
        :param left_context: previous text we are conditioning on
        :param word_end_symbols: tuple of symbols that we consider to end a word, defaults to just the space character
        :param nbest: number of most likely words to return
        :param beam_logp_best: log-prob beam used during the search, hypothesis with log prob > than this distance from best hypothesis are pruned
        :param beam_search_max: maximum number of hypotheses to track during each extension of search
        :param max_word_len: maximum length of words that can be predicted
        :param max_word_hypotheses: stop search if we reach this many complete word prediction hypotheses
        :param return_log_probs: whether to return log probs of each word
        :return: Text sequences that could complete the current word prefix (if any) and (optionally) their log probs
        """
# Old method signature:
#    def predict_words(self,
#                     evidence: List[str],
#                     nbest: int,
#                     beam: int,
#                     max_new_tokens: int = 12,      # 5 was too small; 10–12 is safer
#                     max_completed: int = 32,        # distinct words
#                     batch_size: int = 32,
#                     topk_first: int = 1024,         # top-K within the allowed-first subset
#                     topk_next: int = 10,            # top-K on subsequent steps
#                     prune_margin: float = 5.0) -> List[str]:
#        """
#        Next-word prediction for subword LMs (GPT-2/OPT/...).
#
#        Behavior:
#          - A word only finishes at a boundary: a space OR terminal punctuation (., !, ?, …),
#            optionally followed by closers (") ’ ) ] }.
#          - Punctuation variants of the same word are merged into a single canonical key ('cat', 'cat!!!' -> 'cat').
#          - No dictionary/lexicon gating; coverage is left to the model.
#          - First token is restricted by the typed prefix *before* top-K (prevents dropping the right path).
#
#        Returns:
#          Top-N distinct next-word strings ranked by merged log-prob.
#        """

        # KDV NOTES:
        # evidence        -> now simple string, left_context
        # nbest           -> same
        # beam            -> not used
        # max_new_tokens  -> max steps of the search algorithm
        #                    Other models don't have this notion, they use max_word_len instead to prevent models getting stuck in a cycle.
        #                    For now, hardcoded as local variable and renamed.
        # max_completed   -> added new param to method max_word_hypotheses to handle this type of pruning
        # batch_size      -> now an instance variable
        # topk_first      -> other models don't have this notion, hardcoded as local variable for now
        # topk_next       -> changed to beam_search_max
        # prune_margin    -> changed to beam_logp_best

        # TODO: This implementation ignores word_end_symbols
        # TODO: This implementation ignores max_word_len
        # TODO: I don't think this handles mixed case predictions only lowercase

        # TODO: These two parameters shouldn't be hardcoded
        max_search_steps = 12
        topk_first = 1024

        # We want each language model class set its own default pruning values
        # We want the client keystroke_savings.py to default to these if pruning switches aren't set
        if beam_logp_best is None:
            beam_logp_best = 5.0
        if beam_search_max is None:
            beam_search_max = 10
        if max_word_len is None:
            max_word_len = 50

        # Since List is a mutable type, we can't set a default reliably in the method declaration
        # We'll set the default of a trailing space if caller didn't specify a list of right contexts
        if word_end_symbols is None:
            word_end_symbols = [" "]

        # Create a hash set to quickly determine if character is in the set of end characters
        is_word_end = set()
        for symbol in word_end_symbols:
            is_word_end.add(symbol)

        assert self.model is not None, "language model does not exist!"
        start_ns = time.time_ns()

        # ---------------- Context & prefix ----------------
        #context = "".join(evidence)

        # Split typed context at the last space to obtain current word prefix
        pos = left_context.rfind(" ")

        tokens = list(self.left_context_tokens)
        if pos >= 0:
            word_prefix = left_context[pos + 1:]
            tokens.extend(self._encode(left_context[:pos]))
        else:
            word_prefix = left_context
            tokens.extend(self._encode(""))

        base_len = len(tokens)  # used to slice newly generated suffix correctly

        # ---------------- Vocab decode cache ----------------
        if not hasattr(self, 'decoded_vocab') or len(self.decoded_vocab) != self.tokenizer.vocab_size:
            self.decoded_vocab = [
                self.tokenizer.decode([tid], skip_special_tokens=True)
                for tid in range(self.tokenizer.vocab_size)
            ]
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])

        # ---------------- Allowed first-token set ----------------
        prefix_cmp = word_prefix.lstrip().lower()
        at_bos = (pos < 0)

        if prefix_cmp == "":
            # Empty prefix: at BOS we prefer no leading-space tokens; mid-sentence we prefer leading-space tokens
            if at_bos:
                allowed_first = [
                    tid for tid, s in enumerate(self.decoded_vocab)
                    if tid not in special_ids and not s.startswith(" ")
                ]
                # fallback: allow everything if tokenizer doesn't separate this well
                if not allowed_first:
                    allowed_first = [tid for tid in range(self.tokenizer.vocab_size) if tid not in special_ids]
            else:
                allowed_first = [
                    tid for tid, s in enumerate(self.decoded_vocab)
                    if tid not in special_ids and s.startswith(" ")
                ]
        else:
            allowed_first = []
            for tid, s in enumerate(self.decoded_vocab):
                if tid in special_ids:
                    continue
                clean = s.lstrip().lower()
                if clean.startswith(prefix_cmp) or prefix_cmp.startswith(clean):
                    allowed_first.append(tid)

        if not allowed_first:
            return []

        allowed_first_tensor = torch.tensor(allowed_first, device=self.device, dtype=torch.long)

        # ---------------- Boundary & canonicalization helpers ----------------
        #TRAIL_PUNCT_RE = re.compile(r'[\.!\?…,:;]+$')
        #CLOSERS_RE     = re.compile(r'[\)\]\}\"\'”’]+$')
        #ALNUM_RE       = re.compile(r"[A-Za-z0-9]")

        #def canonicalize_word_from_suffix(raw_suffix: str) -> str:
        #    """
        #    Given the raw decoded suffix (no tokens from the fixed left context),
        #    return canonical word key:
        #      - if a space is present, take up to the first space
        #      - strip trailing punctuation and closers
        #     - lowercase
        #      - require at least one alnum
        #    """
        #    t = raw_suffix
        #    if ' ' in t:
        #        t = t.split(' ')[0]
        #    t = CLOSERS_RE.sub('', TRAIL_PUNCT_RE.sub('', t))
        #    t = t.strip().lower()
        #    return t if ALNUM_RE.search(t) else ""

        def canonicalize_word_from_suffix_new(raw_suffix: str) -> str:
            """
            Return the word prior to hitting on of the specified end of word symbols
            """

            # Scan right looking for any character in the end of word set of characters
            ch_index = 0
            found_alpha = False
            while ch_index < len(raw_suffix):
                ch = raw_suffix[ch_index]
                if not found_alpha and ch.isalpha():
                    found_alpha = True
                if ch in is_word_end:
                    break
                ch_index += 1
            # We need to have found a letter but also must have found an end symbol
            if found_alpha and ch_index < len(raw_suffix):
                return raw_suffix[:ch_index].lower()
            else:
                return ""

        # ---------------- Beam search state ----------------
        LOGP, SEQ = 0, 1
        current_hypos = [(0.0, tokens)]   # (cumulative logp, token_id_sequence)
        completed_words: dict[str, float] = {}  # word -> merged logp
        best_completed_logp = -float("inf")

        # KDV, doesn't seem to be a reason to alias this variable
        #cap_completed = max_completed

        # ---------------- Decoding loop (capped by steps) ----------------
        step = 0
        with torch.inference_mode():
            while current_hypos and step < max_search_steps:
                # Highest-first helps pruning decisions
                current_hypos.sort(reverse=True)

                add_logps = [x[LOGP] for x in current_hypos]
                seqs      = [x[SEQ]  for x in current_hypos]

                all_top_vals = []
                all_top_ids  = []

                # ---- Batched forward pass over ragged sequences (pad + attention_mask) ----

                # If no batch size, we do it all in one inference
                batch_size_to_use = len(current_hypos)
                if self.batch_size:
                    batch_size_to_use = self.batch_size

                for i in range(0, len(current_hypos), batch_size_to_use):
                    batch_seqs  = seqs[i:i + batch_size_to_use]
                    batch_logps = add_logps[i:i + batch_size_to_use]

                    pad_id = (getattr(self.tokenizer, "pad_token_id", None)
                              or getattr(self.tokenizer, "eos_token_id", None) or 0)
                    maxlen = max(len(s) for s in batch_seqs)
                    input_ids  = [s + [pad_id] * (maxlen - len(s)) for s in batch_seqs]
                    attn_mask  = [[1] * len(s) + [0] * (maxlen - len(s)) for s in batch_seqs]

                    input_ids_t  = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                    attn_mask_t  = torch.tensor(attn_mask, dtype=torch.long, device=self.device)

                    before = time.time_ns()
                    out    = self.model(input_ids=input_ids_t, attention_mask=attn_mask_t)
                    logits = out.logits[:, -1, :].float()  # stability if model is fp16
                    logp   = torch.log_softmax(logits, dim=-1)
                    add    = torch.tensor(batch_logps, dtype=logp.dtype, device=self.device).unsqueeze(1)
                    scores = logp + add  # [B, V] cumulative

                    if step == 0:
                        # Restrict BEFORE top-k to the allowed-first subset (prevents dropping correct token)
                        subset = scores.index_select(1, allowed_first_tensor)
                        K = min(subset.size(1), topk_first)
                        top_vals, top_idx = torch.topk(subset, k=K, dim=1, largest=True, sorted=True)
                        top_ids = allowed_first_tensor[top_idx]
                    else:
                        top_vals, top_ids = torch.topk(scores, k=beam_search_max, dim=1, largest=True, sorted=True)

                    self.predict_inference_ns += time.time_ns() - before

                    all_top_vals.append(top_vals)
                    all_top_ids.append(top_ids)

                if not all_top_vals:
                    break

                new_log_probs  = torch.cat(all_top_vals, dim=0)   # [H, K]
                next_token_ids = torch.cat(all_top_ids,  dim=0)   # [H, K]

                next_hypos: list[tuple[float, list[int]]] = []
                decode_cache: dict[tuple[int, ...], str] = {}

                # ---- Expand each hypothesis ----
                for row_idx, (cur_logp, cur_seq) in enumerate(current_hypos):
                    cand_ids   = next_token_ids[row_idx].tolist()
                    cand_logps = new_log_probs[row_idx].tolist()

                    for token_id, cum_logp in zip(cand_ids, cand_logps):
                        if token_id in special_ids:
                            continue

                        new_seq = cur_seq + [token_id]

                        # Decode the *raw* suffix (do not strip leading space yet)
                        suffix_tokens = new_seq[base_len:]
                        key = tuple(suffix_tokens)
                        if key in decode_cache:
                            raw_suffix = decode_cache[key]
                        else:
                            raw_suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True)
                            decode_cache[key] = raw_suffix

                        # For prefix alignment we lstrip, but for boundary detection we need the raw string
                        suffix_for_prefix = raw_suffix.lstrip()

                        # Case-insensitive typed-prefix alignment
                        if not suffix_for_prefix.lower().startswith(prefix_cmp):
                            continue

                        # Completion on boundary:
                        #   - There is a space *anywhere* in the decoded suffix (means a word boundary)
                        #   - Or suffix ends with terminal punctuation and/or closers
#                        completed_now = False
#                        if ' ' in suffix_for_prefix:
#                            # leading space was present -> boundary hit
#                            completed_now = True
#                        elif TRAIL_PUNCT_RE.search(suffix_for_prefix) or CLOSERS_RE.search(suffix_for_prefix):
#                            completed_now = True
#                        if completed_now:
#
#                            canonical = canonicalize_word_from_suffix_new(suffix_for_prefix)
#                            if not canonical:
#                                # No meaningful token after boundary
#                                continue

                        # Scan right looking for any character in the end of word set of characters
                        ch_index = 0
                        found_alpha = False
                        found_word_end = False
                        while ch_index < len(suffix_for_prefix):
                            ch = suffix_for_prefix[ch_index]
                            if not found_alpha and ch.isalpha():
                                found_alpha = True
                            if ch in is_word_end:
                                found_word_end = True
                                break
                            ch_index += 1

                        if found_alpha and found_word_end:
                            canonical = suffix_for_prefix[:ch_index].lower()

                            # Merge duplicates via stable logaddexp
                            prev = completed_words.get(canonical)
                            if prev is None:
                                completed_words[canonical] = cum_logp
                            else:
                                m = max(prev, cum_logp)
                                completed_words[canonical] = m + math.log(math.exp(prev - m) + math.exp(cum_logp - m))

                            if completed_words[canonical] > best_completed_logp:
                                best_completed_logp = completed_words[canonical]

                            # Stop when we have enough DISTINCT words
                            if max_word_hypotheses and len(completed_words) >= max_word_hypotheses:
                                current_hypos = []
                                break
                        else:
                            # Continue the word; prune if far below best completed
                            if best_completed_logp > -float("inf") and cum_logp < best_completed_logp - beam_logp_best:
                                continue

                            # Beam maintenance
                            if not self.beam_width or len(next_hypos) < self.beam_width:
                                heapq.heappush(next_hypos, (cum_logp, new_seq))
                            elif cum_logp > next_hypos[0][LOGP]:
                                heapq.heappushpop(next_hypos, (cum_logp, new_seq))

                    if not current_hypos:
                        break  # cap reached this step

                if not next_hypos:
                    break  # no further expansions

                current_hypos = next_hypos
                step += 1

        # ---------------- Ranking & return ----------------
        ranked = sorted(completed_words.items(), key=lambda kv: kv[1], reverse=True)
        results = [w for (w, _) in ranked[:nbest]]

        # KDV: other models only return the suffix and don't include the prefix
        results = [w[len(word_prefix):] for w in results]

        self.predict_total_ns += time.time_ns() - start_ns
        return results

    def dump_predict_times(self) -> None:
        """Print some stats about the prediction timing"""
        if self.predict_total_ns > 0:
            print(f"Predict %: "
                  f"inference {self.predict_inference_ns / self.predict_total_ns * 100.0:.3f}")

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())

    def get_tokenization(self, text: str) -> List[Tuple]:
        """
        Given a text string, return a list of (subword text, token ID) pairs.
        """
        result = []
        token_ids = self._encode(text)
        for i in range(len(token_ids)):
            result.append((self.index_to_word[token_ids[i]], token_ids[i]))
        return result