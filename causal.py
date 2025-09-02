import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import heapq
from language_model import LanguageModel, compute_max_hypo_len
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
                      return_log_probs: bool = False) -> List:
        """
        Given some left text context, predict the most likely next words.

        Behavior:
          • A word completes when we hit any symbol in `word_end_symbols`
            (default: space, ',', '.', '?', '!'), wherever it occurs.
          • Duplicates are merged (logaddexp) under a lowercase canonical key.
          • Returns only the suffix beyond the typed prefix (keystroke_savings.py
            adds the prefix back).
        """

        # ------------ Defaults & guards ------------
        # Recommended defaults, consistent with the other LMs
        if beam_logp_best is None:
            beam_logp_best = 5.0
        if beam_search_max is None:
            beam_search_max = 100
        if max_word_len is None:
            max_word_len = 50
        if nbest is None:
            nbest = 3

        # boundary set (can be overridden by --word-end)
        if word_end_symbols is None:
            word_end_symbols = [" ", ",", ".", "?", "!"]

        # Fast membership test for boundaries
        end_chars = set(word_end_symbols)

        assert self.model is not None, "language model does not exist!"
        start_ns = time.time_ns()

        # ------------ Context & typed prefix ------------
        pos = left_context.rfind(" ")
        tokens = list(self.left_context_tokens)
        if pos >= 0:
            word_prefix = left_context[pos + 1:]
            tokens.extend(self._encode(left_context[:pos]))
        else:
            word_prefix = left_context
            # keep BOS context only (avoid re-encoding the same string twice)
            # NOTE: for GPT-2/OPT this yields the right “no leading-space” behavior at BOS
            tokens.extend(self._encode(""))

        base_len = len(tokens)

        # Max letters we can still add before boundary (relative to the already-typed prefix)
        max_hypo_len = compute_max_hypo_len(left_context=left_context,
                                            max_word_len=max_word_len)

        # ------------ Cache & special IDs ------------
        # Token→single-piece decode cache (id -> text)
        if not hasattr(self, 'decoded_vocab') or len(self.decoded_vocab) != self.tokenizer.vocab_size:
            self.decoded_vocab = [
                self.tokenizer.decode([tid], skip_special_tokens=True)
                for tid in range(self.tokenizer.vocab_size)
            ]
        special_ids = getattr(self, "special_ids",
                              set(getattr(self.tokenizer, "all_special_ids", []) or []))
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        # IDs at or above this value are *not* decodable into normal tokens
        decodable_cutoff = int(self.tokenizer.vocab_size)


        # ------------ Allowed-first gating ------------
        prefix_cmp = word_prefix.lstrip().lower()
        at_bos = (pos < 0)

        if prefix_cmp == "":
            if at_bos:
                # Prefer tokens without a leading space at BOS
                allowed_first = [tid for tid, s in enumerate(self.decoded_vocab)
                                 if tid not in special_ids and not s.startswith(" ")]
                if not allowed_first:
                    allowed_first = [tid for tid in range(self.tokenizer.vocab_size)
                                     if tid not in special_ids]
            else:
                # Mid-sentence we prefer leading-space tokens
                allowed_first = [tid for tid, s in enumerate(self.decoded_vocab)
                                 if tid not in special_ids and s.startswith(" ")]
        else:
            allowed_first = []
            for tid, s in enumerate(self.decoded_vocab):
                if tid in special_ids:
                    continue
                clean = s.lstrip().lower()
                # allow token if it starts with the typed prefix, or is a prefix of it
                if clean.startswith(prefix_cmp) or prefix_cmp.startswith(clean):
                    allowed_first.append(tid)

        if not allowed_first:
            return []

        allowed_first_tensor = torch.tensor(allowed_first, device=self.device, dtype=torch.long)

        # ------------ Safe decode for suffix ------------
        def _decode_suffix_safe(ids: List[int]) -> str:
            """Decode a variable-length suffix robustly (no None tokens)."""
            if not ids:
                return ""
            # Keep only decodable, non-special IDs (prevents None tokens on OPT/GPT-2)
            filt = [tid for tid in ids if tid not in special_ids and tid < decodable_cutoff]
            if not filt:
                return ""
            try:
                # Fast path
                return self.tokenizer.decode(filt, skip_special_tokens=True)
            except Exception:
                # Fallback path
                toks = self.tokenizer.convert_ids_to_tokens(filt, skip_special_tokens=False)
                toks = [t for t in toks if isinstance(t, str)]
                try:
                    return self.tokenizer.convert_tokens_to_string(toks)
                except Exception:
                    return "".join(toks)

        # Utility: letters seen before hitting a boundary (and whether we hit one)
        def _letters_until_boundary(s: str) -> tuple[int, bool, bool, int]:
            """
            Returns:
              letters_count, saw_alpha, hit_boundary, boundary_index
            """
            letters = 0
            saw_alpha = False
            hit = False
            idx = 0
            while idx < len(s):
                ch = s[idx]
                if ch.isalpha():
                    saw_alpha = True
                    letters += 1
                if ch in end_chars:
                    hit = True
                    break
                idx += 1
            return letters, saw_alpha, hit, idx

        # ------------ Beam state ------------
        LOGP, SEQ = 0, 1
        current_hypos: list[tuple[float, list[int]]] = [(0.0, tokens)]
        completed_logp: dict[str, float] = {}        # canonical(lower) -> merged logp
        best_surface: dict[str, tuple[float, str]] = {}  # canonical(lower) -> (best_logp, best_surface_form)
        best_completed_logp = -float("inf")

        # ------------ Decoding loop ------------
        step = 0
        max_search_steps = 12        # original setting
        topk_first = 1024            # original setting

        with torch.inference_mode():
            while current_hypos and step < max_search_steps:
                # Highest-first helps pruning decisions; sort by logp only
                current_hypos.sort(key=lambda t: t[0], reverse=True)  
                add_logps = [x[LOGP] for x in current_hypos]
                seqs      = [x[SEQ]  for x in current_hypos]

                all_top_vals = []
                all_top_ids  = []

                # ---- Batched forward (pad + mask) ----
                batch_size_to_use = len(current_hypos) if not self.batch_size else self.batch_size
                for i in range(0, len(current_hypos), batch_size_to_use):
                    batch_seqs  = seqs[i:i + batch_size_to_use]
                    batch_logps = add_logps[i:i + batch_size_to_use]

                    pad_id = (getattr(self.tokenizer, "pad_token_id", None)
                              or getattr(self.tokenizer, "eos_token_id", None) or 0)
                    maxlen = max(len(s) for s in batch_seqs)
                    input_ids = [s + [pad_id] * (maxlen - len(s)) for s in batch_seqs]
                    attn_mask = [[1] * len(s) + [0] * (maxlen - len(s)) for s in batch_seqs]

                    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                    attn_mask_t = torch.tensor(attn_mask, dtype=torch.long, device=self.device)

                    before = time.time_ns()
                    out    = self.model(input_ids=input_ids_t, attention_mask=attn_mask_t)
                    logits = out.logits[:, -1, :].float()
                    logp   = torch.log_softmax(logits, dim=-1)
                    add    = torch.tensor(batch_logps, dtype=logp.dtype, device=self.device).unsqueeze(1)
                    scores = logp + add  # [B, V]

                    if step == 0:
                        subset = scores.index_select(1, allowed_first_tensor)
                        K = min(subset.size(1), topk_first)
                        top_vals, top_idx = torch.topk(subset, k=K, dim=1, largest=True, sorted=True)
                        top_ids = allowed_first_tensor[top_idx]
                    else:
                        K2 = min(beam_search_max, scores.size(1))
                        top_vals, top_ids = torch.topk(scores, k=K2, dim=1, largest=True, sorted=True)

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
                        # Skip specials or undecodable IDs immediately
                        if token_id in special_ids or token_id >= decodable_cutoff:
                            continue
                        if token_id == eos_id:
                            # treat EOS as a boundary; no surface char added
                            raw_suffix = _decode_suffix_safe(cur_seq[base_len:])
                            suffix_for_prefix = raw_suffix.lstrip()
                            # The typed prefix must still align
                            if not suffix_for_prefix.lower().startswith(prefix_cmp):
                                continue
                            # We've "completed" whatever was there before EOS, if it is meaningful
                            letters, saw_alpha, hit, boundary_index = _letters_until_boundary(suffix_for_prefix)
                            if saw_alpha:
                                canonical_lower = suffix_for_prefix.lower()[:boundary_index]
                                surface = suffix_for_prefix[:boundary_index]
                                prev = completed_logp.get(canonical_lower)
                                if prev is None:
                                    completed_logp[canonical_lower] = cum_logp
                                    best_surface[canonical_lower] = (cum_logp, surface)
                                else:
                                    m = max(prev, cum_logp)
                                    completed_logp[canonical_lower] = m + math.log(math.exp(prev - m) +
                                                                                   math.exp(cum_logp - m))
                                    if cum_logp > best_surface[canonical_lower][0]:
                                        best_surface[canonical_lower] = (cum_logp, surface)
                                best_completed_logp = max(best_completed_logp, completed_logp[canonical_lower])
                            # do not continue from EOS
                            continue

                        new_seq = cur_seq + [token_id]

                        # Decode just the new suffix (tokens after the fixed base)
                        suffix_tokens = new_seq[base_len:]
                        key = tuple(suffix_tokens)
                        if key in decode_cache:
                            raw_suffix = decode_cache[key]
                        else:
                            raw_suffix = _decode_suffix_safe(suffix_tokens)
                            decode_cache[key] = raw_suffix

                        # Align with typed prefix (case-insensitive, ignore leading space)
                        suffix_for_prefix = raw_suffix.lstrip()
                        if not suffix_for_prefix.lower().startswith(prefix_cmp):
                            continue
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # Cheap early guard: if we've already generated too many characters
                        # beyond the typed prefix, prune this candidate before boundary scan.
                        if max_word_len is not None:
                            generated_len = max(0, len(suffix_for_prefix) - len(prefix_cmp))
                            if generated_len > max_word_len:
                                continue
                       # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                        # Check boundary and length
                        letters, saw_alpha, hit, boundary_index = _letters_until_boundary(suffix_for_prefix)

                        if hit and saw_alpha:
                            # Completed word up to the first end char
                            surface = suffix_for_prefix[:boundary_index]
                            canonical_lower = surface.lower()

                            prev = completed_logp.get(canonical_lower)
                            if prev is None:
                                completed_logp[canonical_lower] = cum_logp
                                best_surface[canonical_lower] = (cum_logp, surface)
                            else:
                                m = max(prev, cum_logp)
                                completed_logp[canonical_lower] = m + math.log(math.exp(prev - m) +
                                                                               math.exp(cum_logp - m))
                                if cum_logp > best_surface[canonical_lower][0]:
                                    best_surface[canonical_lower] = (cum_logp, surface)

                            if completed_logp[canonical_lower] > best_completed_logp:
                                best_completed_logp = completed_logp[canonical_lower]

                            if max_word_hypotheses and len(completed_logp) >= max_word_hypotheses:
                                current_hypos = []  # stop everything
                                break
                        else:
                            # Not completed yet. Enforce beam / best-completed pruning.
                            if (best_completed_logp > -float("inf")
                                    and cum_logp < best_completed_logp - beam_logp_best):
                                continue
                            # Enforce max_word_len relative to typed prefix
                            if letters > max_hypo_len:
                                continue

                            # Beam maintenance (use beam_search_max, same as other models)
                            if len(next_hypos) < beam_search_max:
                                heapq.heappush(next_hypos, (cum_logp, new_seq))
                            elif cum_logp > next_hypos[0][LOGP]:
                                heapq.heappushpop(next_hypos, (cum_logp, new_seq))

                    if not current_hypos:
                        break  # max_word_hypotheses was hit

                if not next_hypos:
                    break

                current_hypos = next_hypos
                step += 1

        # ------------ Rank & format return ------------
        ranked = sorted(completed_logp.items(), key=lambda kv: kv[1], reverse=True)[:nbest]

        # The evaluator expects only the suffix beyond the typed prefix.
        if self.predict_lower:
            # return lowercase suffixes
            results = [key[len(word_prefix):] for (key, _) in ranked]
            if return_log_probs:
                return [(key[len(word_prefix):], lp) for (key, lp) in ranked]
            else:
                self.predict_total_ns += time.time_ns() - start_ns
                return results
        else:
            # preserve best surface casing for each canonical key
            surfaced = []
            for key, lp in ranked:
                best_surf = best_surface[key][1]
                surfaced.append((best_surf[len(word_prefix):], lp))
            if return_log_probs:
                self.predict_total_ns += time.time_ns() - start_ns
                return surfaced
            else:
                self.predict_total_ns += time.time_ns() - start_ns
                return [s for (s, _) in surfaced]
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
