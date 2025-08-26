import torch
from typing import List, Tuple, Final
from numpy import ndarray
from numpy import sum
from transformers import AutoModelForCausalLM, AutoTokenizer

from language_model import LanguageModel, compute_max_hypo_len
from exceptions import InvalidLanguageModelException
from scipy.special import logsumexp
from scipy.special import softmax
import heapq
import numpy as np

# This updates transformers 4.20.0 to be able to use the ByGPT5 tokenizer and model
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer, ByGPT5Config

class CausalByteLanguageModel(LanguageModel):
    """Character byte-level language model based on a pre-trained causal model, e.g. ByGPT5"""

    def __init__(self,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 fp16: bool = False,
                 batch_size: int = None,
                 predict_lower: bool = True,
                ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type      - SYMBOL only
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            fp16               - convert model to fp16 to save memory/compute on CUDA
            batch_size         - batch size for doing multiple inferences at same time (currently used only in predict_word)
            predict_lower      - if we internally marginalize predictions based on upper and lowercase hypotheses
        """
        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.vocab = {}
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.result_to_vocab_indexes = []
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.symbol_index_to_vocab_index = []
        self.batch_size = batch_size
        self.predict_lower = predict_lower

        # Taken from: https://github.com/potamides/uniformers/blob/main/examples/inference/lm_perplexity.py
        # We need to add this to be able to use ByGPT5 with AutoModel
        CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
        MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
        TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))

        # And this too, if we want to test the raw ByT5 decoder
        MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model
        self.model_name = lang_model_name
        self.model_dir = lm_path if lm_path else self.model_name

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
            self.symbol_set_lower.append(ch.lower())

        self._build_vocab()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        # List of empty lists
        self.result_to_vocab_indexes = [ [] for _ in range(len(self.symbol_set_lower)) ]

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower

            # Create a mapping between the vocab index and the index in the result set
            try:
                self.result_to_vocab_indexes[self.symbol_set_lower.index(word_lower)].append(i)
            except ValueError:
                pass

        # Make an array that can convert the index of a symbol into the token ID
        for ch in self.symbol_set:
            self.symbol_index_to_vocab_index.append(self._encode(ch)[0])

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            self.left_context = "</s>"

        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        print(f"CausalByte: left_context = '{self.left_context}', left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > 1 and self.model_name.startswith("facebook/opt"):
            # Some models always add </s> at start which we don't want since we may have our own left context
            tokens = tokens[1:]
        elif len(tokens) > 1 and self.model_name.startswith("google/byt5"):
            # Some models always add </s> at end
            tokens = tokens[:-1]
        return tokens

    def _get_symbol_log_probs(self, log_probs: ndarray) -> List[float]:
        """
        Create a simple list with the log probs of all the characters we need to return
        :param log_probs: List of log probs from all the tokens in the LM
        :return: List of log probs of just the character in our vocab, marginalized over upper and lowercase
        """
        result_log_probs = []
        for i in range(len(self.symbol_set_lower)):
            # List of 1 or more indexes in the LLM vocab we need to sum
            indexes = self.result_to_vocab_indexes[i]
            if len(indexes) == 1:
                result_log_probs.append(float(log_probs[indexes[0]]))
            elif len(indexes) > 1:
                # Create a list of the log probs for this character
                char_log_probs = []
                for index in indexes:
                    char_log_probs.append(log_probs[index])
                result_log_probs.append(logsumexp(char_log_probs))
            else:
                # This should only happen if the language model doesn't have all our characters
                result_log_probs.append(float("-inf"))
        return result_log_probs

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

        # We want each language model class set its own default pruning values
        # We want the client keystroke_savings.py to default to these if pruning switches aren't set
        if beam_logp_best is None:
            beam_logp_best = 5.0
        if beam_search_max is None:
            beam_search_max = 100
        if max_word_len is None:
            max_word_len = 50

        # Since List is a mutable type, we can't set a default reliably in the method declaration
        # We'll set the default of a trailing space if caller didn't specify a list of right contexts
        if word_end_symbols is None:
            word_end_symbols = [" "]

        # Create a symbol set that also includes any end of word symbols that aren't in our normal symbol set
        # If any of the end symbols occur in the normal symbol set, we include at end of list
        search_symbols = []
        for symbol in self.symbol_set:
            if symbol not in word_end_symbols:
                # We'll add both an upper and lowercase version of this symbol
                # since we want the search to consider any casing
                if symbol.lower() != symbol.upper():
                    search_symbols.append(symbol.lower())
                    search_symbols.append(symbol.upper())
                else:
                    search_symbols.append(symbol)
        index_first_end_symbol = len(search_symbols)
        for end_symbol in word_end_symbols:
            search_symbols.append(end_symbol)

        # Create a parallel list of the token IDs for our search symbols
        search_symbol_ids = [self._encode(symbol) for symbol in search_symbols]

        tokens = []
        tokens.extend(self.left_context_tokens)
        # Don't extend if the context is empty, this avoids some models like byt5 from adding extra </s> at start
        if len(left_context) > 0:
            tokens.extend(self._encode(left_context))

        # We store hypotheses in a list with a tuple (log_prob, current word characters, subword token sequence)
        current_hypos = [(0.0, "", tokens)]

        LOGP: Final[int] = 0
        STR: Final[int] = 1
        TOKENS: Final[int] = 2

        pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]

        # Finished hypotheses map a word string (without ending symbols) to its log prob
        # We use a dictionary since we may want to merge hypotheses that are the same word
        finished_hypos = {}
        best_finished_log_prob = float("-inf")

        # Compute the maximum length of hypotheses based on existing prefix of word (if any)
        max_hypo_len = compute_max_hypo_len(left_context=left_context, max_word_len=max_word_len)

        # Flag that breaks out of all loops
        done = False

        while len(current_hypos) > 0 and not done:
            # We'll store extended hypotheses in a min heap to make it easy to maintain only a fixed number of the best
            next_hypos = []

            # Go through all the current hypotheses loading into one or more mini-batches
            hypo_index = 0
            while hypo_index < len(current_hypos) and not done:
                size = 0
                max_length = 0
                batch_tokens = []

                # Remember the starting index for this minibatch
                batch_start_index = hypo_index

                # Add to the next mini-batch until we either run out of hypotheses or we hit mini-batch size
                # Keep track of the maximum length hypothesis as we go through them
                while (not self.batch_size or size < self.batch_size) and hypo_index < len(current_hypos):
                    tokens = current_hypos[hypo_index][TOKENS]
                    max_length = max(len(tokens), max_length)
                    batch_tokens.append(tokens)
                    size += 1
                    hypo_index += 1

                # Pad out every token sequence to make length and make the tensor for inference
                batch_tensors = []
                for tokens in batch_tokens:
                    while len(tokens) < max_length:
                        tokens.append(pad_id)
                    batch_tensors.append(torch.tensor(tokens))
                tokens_tensor = torch.stack(tuple(batch_tensors)).to(self.device)

                with torch.no_grad():
                    logits = self.model(tokens_tensor).logits  # shape (batch_size, max_length, 384)
                    # We care about the logits in the last position of second dimension
                    # We want to sum to one in the last dimension over the first dimensions (different hypotheses)
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()

                # For each hypothesis in the mini-batch, create a new next hypothesis for every character
                # in the symbol set that isn't one of our end of word symbols.
                # For end of word symbols, add the hypothesis to the finished_hypos dictionary.
                batch_index = 0
                while batch_index < len(log_probs) and not done:
                    for search_index, token_index in enumerate(search_symbol_ids):
                        # Grab the corresponding hypothesis from the original set
                        hypo = current_hypos[batch_start_index + batch_index]

                        # Pull the log prob for the corresponding position in the inference and add to existing accumulated log prob
                        # Use float cast to avoid 16-bit floating point when on GPU with --fp16 switch
                        new_log_prob = float(log_probs[batch_index, token_index]) + hypo[LOGP]

                        # We avoid adding finished or intermediate hypotheses if they are outside log prob beam
                        # This is a bit faster than only doing it for intermediate hypotheses
                        if (best_finished_log_prob - new_log_prob) < beam_logp_best:
                            # See if we have finished by generating any of the valid right symbols
                            # These were organized to be at the end of the list of search_symbols
                            if search_index >= index_first_end_symbol and len(hypo[STR]) <= max_hypo_len:
                                # NOTE: we don't add the ending symbol to the finished hypothesis
                                # Optionally we marginalize all cases to predict just lowercase words
                                if self.predict_lower:
                                    word = hypo[STR].lower()
                                else:
                                    word = hypo[STR]

                                if word in finished_hypos:
                                    # If already had this word finish with a different end symbol we sum the probabilities
                                    finished_hypos[word] = np.logaddexp(finished_hypos[word], new_log_prob)
                                else:
                                    # Haven't seen this word, we will just always add it to the dictionary
                                    # It would be expensive to maintain a fixed dictionary size of the best finished hypotheses
                                    finished_hypos[word] = new_log_prob

                                    # Stop the search if we hit a hard cap on distinct completed word hypotheses
                                    if max_word_hypotheses and len(finished_hypos) >= max_word_hypotheses:
                                        done = True
                                        break
                                # Update the current best log prob of any finishing hypothesis
                                best_finished_log_prob = max(best_finished_log_prob, new_log_prob)

                            # This hypothesis didn't finish
                            # Keep if it is still within beam width of our best hypothesis thus far
                            elif len(hypo[STR]) < max_hypo_len:
                                # This hypothesis is within the beam of the best to date
                                # Extend by the text and token ID
                                new_hypo = (new_log_prob, hypo[STR] + search_symbols[search_index], hypo[TOKENS] + token_index)
                                if len(next_hypos) < beam_search_max:
                                    # Add if we haven't reached our beam width limit so add
                                    heapq.heappush(next_hypos, new_hypo)
                                else:
                                    # Or replace the worst hypotheses with the new one
                                    heapq.heappushpop(next_hypos, new_hypo)
                    batch_index += 1

            current_hypos = next_hypos

        # Convert our dictionary of finished hypotheses to a sorted list
        sorted_best = sorted(finished_hypos.items(), key=lambda item: item[1], reverse=True)[:nbest]

        # Optional inclusion of log prob in result
        return [(hypo[0], hypo[1]) if return_log_probs else hypo[0] for hypo in sorted_best]


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

        context = "".join(evidence)

        # TODO: add support for prediction of mixed case

        tokens = []
        tokens.extend(self.left_context_tokens)
        # Don't extend if the context is empty, this avoids some models like byt5 from adding extra </s> at start
        if len(context) > 0:
            tokens.extend(self._encode(context))

        tensor = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).logits # Shape is (1, 1, 384)
            log_probs = torch.log_softmax(logits[-1, -1, :], dim=0).detach().cpu().numpy()

        # Create a simple list with the probabilities of all the characters we need to return
        char_probs = self._get_symbol_log_probs(log_probs=log_probs)

        # Normalize to a distribution that sums to 1
        char_probs = softmax(char_probs)

        # Now construct the return dictionary that maps the character to its probability
        next_char_pred = {}
        for i, ch in enumerate(self.symbol_set_lower):
            next_char_pred[ch] = char_probs[i]

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())
