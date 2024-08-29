from collections import Counter
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
import heapq
from language_model import LanguageModel
from language_model import BACKSPACE_CHAR, SPACE_CHAR
from exceptions import InvalidLanguageModelException
from scipy.special import logsumexp
from scipy.special import softmax
import time
from collections import defaultdict
import sys


class CausalLanguageModel(LanguageModel):
    """Character language model based on a pre-trained causal model, GPT-2 by default."""

    def __init__(self,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = 8,
                 token_backoff: int = -1,
                 fp16: bool = False,
                 mixed_case_context = False,
                 case_simple = False,
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            beam_width         - how many hypotheses to keep during the search
            token_backoff      - how many tokens to remove prior to search extension, -1 = previous space
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
        """
        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = defaultdict(list)
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model
        self.model_name = lang_model_name
        self.model_dir = lm_path if lm_path else self.model_name

        # Parameters for the search
        self.beam_width = beam_width

        # How much to backoff from the current tokenization
        self.token_backoff = token_backoff

        # Simple heuristic to correct case in the LM context
        self.simple_upper_words = {"i": "I",
                                    "i'll": "I'll",
                                    "i've": "I've",
                                    "i'd": "I'd",
                                    "i'm": "I'm"}

        # Track how much time spent in different parts of the predict function
        self.predict_total_ns = 0
        self.predict_start_ns = 0
        self.predict_search_ns = 0
        self.predict_end_ns = 0
        self.predict_search_inner_ns = 0
        self.predict_inference_ns = 0
        self.predict_create_sequence_ns = 0
        self.predict_create_prefixes_ns = 0
        self.predict_create_prefixes_top_ns = 0
        self.predict_create_prefixes_longer_ns = 0
        self.predict_create_prefixes_shorter_ns = 0
        self.predict_prep_vocab_ns = 0
        self.predict_create_prefixes_likelihood_ns = 0

        self.load()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        # Loop over all the subword tokens in the LLM
        for i in range(self.vocab_size):
            # Create a map from the subword token integer ID to the mixed and lowercase string versions
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower

            # Check if all the characters in the subword token are in our valid symbol set
            valid = True
            for ch in word_lower:
                # The space char is only valid once we convert spaces to the space char
                if ch == SPACE_CHAR:
                    valid = False
                    break
                if ch == ' ':
                    continue
                elif ch not in self.symbol_set_lower:
                    valid = False
                    break

            # If the subword token symbols are all valid, then add it to the list of valid token IDs
            if valid:
                self.valid_vocab += i,
                # Add this token ID to all lists for its valid text prefixes
                for j in range(len(word)):
                    key = word_lower[0:j + 1].replace(' ', SPACE_CHAR)
                    self.vocab[key] += i,

        # When done, self.vocab can be used to map to possible following subword tokens given some text, e.g.:
        # self.vocab["cyclo"] = [47495, 49484]
        # self.index_to_word[self.vocab["cyclo"][0]] = cyclop
        # self.index_to_word[self.vocab["cyclo"][1]] = cyclopedia

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            elif "Llama-3.1" in self.model_name:
                self.left_context = "<|begin_of_text|>"
            else:
                self.left_context = "</s>"

        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        print(f"Causal: left_context = '{self.left_context}', left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        # Both OPT and Llama automatically insert a start token which we want to control ourselves
        if len(tokens) > 1 and (self.model_name.startswith("facebook/opt") or "Llama-3.1" in self.model_name):
            tokens = tokens[1:]

        return tokens

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
        #start_ns = time.time_ns()

        converted_context = "".join(evidence)
        converted_context_lower = converted_context.lower()
        context = converted_context.replace(SPACE_CHAR, ' ')

        # If using the simple case feature, we need to go through the actual
        # left context and capitalize the first letter in the sentence as
        # well as any word in our list of words that should be capitalized.
        if self.case_simple and len(context) > 0:
            cased_context = ""
            words = context.split()
            for i, word in enumerate(words):
                if i == 0 and word[0] >= 'a' and word[0] <= 'z':
                    word = word[0].upper() + word[1:]
                if i > 0:
                    if word in self.simple_upper_words:
                        word = self.simple_upper_words[word]
                    cased_context += " "
                cased_context += word
            # Handle ending space in the context
            if context[-1] == ' ':
                cased_context += " "
            #print(f"DEBUG, predict, simple case from {context} to {cased_context}")
            context = cased_context

        context_lower = context.lower()

        # Index in the hypothesis string that is the next character after our context
        target_pos = len(context_lower)

        # Remove token_backoff tokens from the end of the context
        # If token_backoff is -1 or goes beyond a word boundary, then
        # search from the last space character in the context
        # If no space, then from the very beginning
        tokens = self._encode(context_lower)

        #print(f"DEBUG, predict, left_context_tokens = {self.left_context_tokens}")

        # Look for the last space in the context, or -1 if no begin_text in context yet
        pos = context_lower.rfind(" ")
        truncated_tokens = []
        truncated_tokens.extend(self.left_context_tokens)
        if pos >= 0:
            # Optionally, we condition on upper and lower case left context
            if self.mixed_case_context:
                truncated_context = context[0:pos]
            else:
                truncated_context = context_lower[0:pos]
            truncated_tokens.extend(self._encode(truncated_context))

        # If token_backoff is -1, then we backoff to the last space.
        # Otherwise, we use a fixed number looking backwards.
        if self.token_backoff == -1 or len(tokens) - self.token_backoff < len(truncated_tokens):
            tokens = truncated_tokens
        else:
            tokens = tokens[:-self.token_backoff]

        #print(f"DEBUG, predict, tokens = {tokens}")

        # Our starting hypothesis that we'll be extending
        # Format is (log likelihood, token id sequence, text length)
        start_length = 0
        for x in tokens[1:]:
            start_length += len(self.index_to_word_lower[x])
        current_hypos = [(0.0, tokens, start_length)]

        # We use a priority queue to track the top hypotheses during the beam search.
        # For a beam of 8, empirical testing showed this was about the same amount
        # of time as a simpler list that used a linear search to replace when full.
        heapq.heapify(current_hypos)

        # Create a hash mapping each valid following character to a list of log probabilities
        char_to_log_probs = defaultdict(list)

        #print(f"DEBUG, predict, search start with {current_hypos}")

        #before_search_ns = time.time_ns()
        #completed = 0

        # Add new extended hypotheses to this heap
        next_hypos = []

        # Start a beam search forward from the backed off token sequence.
        # Each iteration of this while loop extends hypotheses by all valid tokens.
        # We only keep at most self.beam_width hypotheses in the valid heap.
        while len(current_hypos) > 0:
            # Work on the hypotheses from the last round of extension
            #print(f"DEBUG, predict, loop {loop}, completed {completed}, current = {current_hypos}")

#            before_search_inner_ns = time.time_ns()
            # Keep going until we have extended all hypotheses in the current set
            #current_batch = 0
            #batch_tensors = []
            #batch_sequences = []
            #batch_likelihoods = []
            #batch_text_lengths = []

            # Empty the pending hypotheses into the parallel lists
            # and create the tensor for inference against the LLM.
            #while len(current_hypos) > 0:
#                before_create_sequence_ns = time.time_ns()
                # Get the new sequence to work on
            #    (current_likelihood, sequence, text_length) = current_hypos.pop(0)
            #    batch_tensors += torch.tensor(sequence),
            #    batch_sequences += sequence,
            #    batch_likelihoods += current_likelihood,
            #    batch_text_lengths += text_length,
            #    current_batch += 1

            # Go through the pending hypotheses and create the torch tensor for the inference
            batch_tensors = []
            for current in current_hypos:
                batch_tensors += torch.tensor(current[1]),
            tokens_tensor = torch.stack(tuple(batch_tensors)).to(self.device)

#                self.predict_create_sequence_ns += time.time_ns() - before_create_sequence_ns

#                before_inference_ns = time.time_ns()
            with torch.no_grad():
                logits = self.model(tokens_tensor).logits
                #log_probs = torch.log_softmax(logits[:, -1, :], dim=1).to("cpu")
                # This makes the likelihood sum lower down much faster
                log_probs = torch.log_softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()
#                self.predict_inference_ns += time.time_ns() - before_inference_ns

#                print(f"DEBUG, predict, log_probs = {log_probs}")

#            for batch_index in range(current_batch):
            for current_index, current in enumerate(current_hypos):
                #before_prep_vocab_ns = time.time_ns()

                # Rebuild the text from the sequence of subtoken IDs
                #sequence_text = "".join([self.index_to_word_lower[x] for x in batch_sequences[batch_index][1:]])

                # We only need to know the length of the text sequence in the hypothesis.
                #sequence_text_len = 0
                #for x in batch_sequences[batch_index][1:]:
                #    sequence_text_len += len(self.index_to_word_lower[x])
                #sequence_text_len = batch_text_lengths[batch_index]
                sequence_text_len = current[2]

                vocab = []
                extra_vocab = []

                remaining_context = converted_context_lower[sequence_text_len:]
                if len(remaining_context) == 0:
                    vocab = self.valid_vocab
                else:
                    if remaining_context in self.vocab:
                        vocab = self.vocab[remaining_context]
                    for i in range(1, len(remaining_context)):
                        tokenization = self._encode(context_lower[sequence_text_len:sequence_text_len + i])
                        if len(tokenization) == 1:
                            extra_vocab += tokenization[0],
                #hypo_likelihood = batch_likelihoods[batch_index]
                hypo_likelihood = current[0]
                #self.predict_prep_vocab_ns += time.time_ns() - before_prep_vocab_ns

                # Create a list of token indexes that are a prefix of the target text.
                # We go over all the integer IDs in the vocab and extra_vocab lists.
                #before_create_prefixes_ns = time.time_ns()
                for token_id in itertools.chain(vocab, extra_vocab):
                    #before_create_prefixes_top_ns = time.time_ns()
                    #self.predict_create_prefixes_top_ns += time.time_ns() - before_create_prefixes_top_ns

                    #before_create_prefixes_likelihood_ns = time.time_ns()
                    # Add the log prob of this token to the previous running total.
                    # For some reason the float cast makes it run about twice as fast!
                    # This line takes the majority of the time???
                    likelihood = hypo_likelihood + log_probs[current_index][token_id]
                    #self.predict_create_prefixes_likelihood_ns += time.time_ns() - before_create_prefixes_likelihood_ns

                    # For a hypothesis to finish it must extend beyond the existing typed context
                    if (sequence_text_len + len(self.index_to_word_lower[token_id])) > len(context):
                        #before_create_prefixes_longer_ns = time.time_ns()
                        # Add to this likelihood to the list for the character at the prediction position
                        char_to_log_probs[self.index_to_word_lower[token_id][target_pos - sequence_text_len]] += likelihood,
                        #completed += 1
                        #self.predict_create_prefixes_longer_ns += time.time_ns() - before_create_prefixes_longer_ns
                    else:
                        #before_create_prefixes_shorter_ns = time.time_ns()
                        # Check we are actually going to use the new hypotheses before creating it
                        if len(next_hypos) < self.beam_width or likelihood > next_hypos[0][0]:
                            # Prepare a new extended sequence to add to the heap
                            #hypo_seq = batch_sequences[batch_index].copy()
                            hypo_seq = current[1].copy()
                            hypo_seq += token_id,
                            hypo = (likelihood, hypo_seq, sequence_text_len + len(self.index_to_word_lower[token_id]))
                            if len(next_hypos) < self.beam_width:
                                # If we are under the beam limit then just add it
                                heapq.heappush(next_hypos, hypo)
                            else:
                                # Or replace the worst hypotheses with the new one
                                heapq.heappushpop(next_hypos, hypo)

                        #self.predict_create_prefixes_shorter_ns += time.time_ns() - before_create_prefixes_shorter_ns
               #self.predict_create_prefixes_ns += time.time_ns() - before_create_prefixes_ns

            # Swap in the extended set as the new current working set
            current_hypos = next_hypos
            next_hypos = []
        #self.predict_search_inner_ns += time.time_ns() - before_search_inner_ns

        #after_search_ns = time.time_ns()

        # Parallel array to symbol_set for storing the marginals
        char_probs = []
        for ch in self.symbol_set_lower:
            # Convert space to the underscore used in BciPy
            if ch == SPACE_CHAR:
                target_ch = ' '
            else:
                target_ch = ch

            # Handle cases when symbols are never seen
            if target_ch in char_to_log_probs:
                char_probs += logsumexp(char_to_log_probs[target_ch]),
            else:
                char_probs += float("-inf"),
        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

        next_char_pred = Counter()

        for i, ch in enumerate(self.symbol_set_lower):
            if ch is SPACE_CHAR:
                next_char_pred[ch] = char_probs[i]
            else:
                next_char_pred[ch.upper()] = char_probs[i]

        next_char_pred[BACKSPACE_CHAR] = 0.0

        #end_ns = time.time_ns()
        #self.predict_start_ns += before_search_ns - start_ns
        #self.predict_search_ns += after_search_ns - before_search_ns
        #self.predict_end_ns += end_ns - after_search_ns
        #self.predict_total_ns += end_ns - start_ns

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def dump_predict_times(self) -> None:
        """Print some stats about the prediction timing"""
        if self.predict_total_ns > 0:
            print(f"Predict %: start {self.predict_start_ns / self.predict_total_ns * 100.0:.3f} "
                  f"search {self.predict_search_ns / self.predict_total_ns * 100.0:.3f} "
                  f"search_inner {self.predict_search_inner_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_sequence {self.predict_create_sequence_ns / self.predict_total_ns * 100.0:.3f} "
                  f"inference {self.predict_inference_ns / self.predict_total_ns * 100.0:.3f} "
                  f"prep_vocab {self.predict_prep_vocab_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_prefixes {self.predict_create_prefixes_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_prefixes_top {self.predict_create_prefixes_top_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_prefixes_likelihood {self.predict_create_prefixes_likelihood_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_prefixes_longer {self.predict_create_prefixes_longer_ns / self.predict_total_ns * 100.0:.3f} "
                  f"create_prefixes_shorter {self.predict_create_prefixes_shorter_ns / self.predict_total_ns * 100.0:.3f} "
                  f"end {self.predict_end_ns / self.predict_total_ns * 100.0:.3f}")

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        except BaseException:
            raise InvalidLanguageModelException(f"{self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            if self.fp16 and self.device == "cuda":
                self.model = self.model.half()
        except:
            raise InvalidLanguageModelException(f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

        self.model.eval()

        self.model.to(self.device)

        self.symbol_set_lower = []
        for ch in self.symbol_set:
            if ch is SPACE_CHAR:
                self.symbol_set_lower.append(SPACE_CHAR)
            elif ch is BACKSPACE_CHAR:
                continue
            else:
                self.symbol_set_lower.append(ch.lower())

        self._build_vocab()

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbol with probability
        """
        next_char_pred = self.predict(evidence)

        return next_char_pred

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())



