import torch
from typing import List, Tuple, Final
from numpy import ndarray
from numpy import sum
from transformers import AutoModelForCausalLM, AutoTokenizer
from language_model import LanguageModel
from exceptions import InvalidLanguageModelException
from scipy.special import logsumexp
from scipy.special import softmax
import heapq

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
                 case_simple: bool = False,
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
            case_simple        - simple fixing of left context case
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
        self.case_simple = case_simple
        self.symbol_index_to_vocab_index = []

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

        self.load()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        # List of empty lists
        self.result_to_vocab_indexes = [ [] for _ in range(len(self.symbol_set_lower)) ]

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            #print(f"DEBUG vocab {i} '{word}'")
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
                      return_log_probs=False) -> List:

        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to <sp>
        :param left_context: previous text we are condition on
        :param word_end_symbols: tuple of symbols that we consider to end a word, defaults to just the space character
        :param nbest: number of most likely words to return
        :param beam_logp_best: log-prob beam used during the search, hypothesis with log prob > than this distance from best hypothesis are pruned
        :param beam_search_max: maximum number of hypotheses to track during each extension of search
        :param return_log_probs: whether to return log probabilities of each word
        :return: List of tuples with words and (optionally) their log probabilities
        """

        # We want each language model class set its own default pruning values
        # We want the client keystroke_savings.py to default to these if pruning switches aren't set
        if beam_logp_best is None:
            beam_logp_best = 5.0
        if beam_search_max is None:
            beam_search_max = 100

        # Since List is a mutable type, we can't set a default reliably in the method declaration
        # We'll set the default of a trailing space if caller didn't specify a list of right contexts
        if word_end_symbols is None:
            word_end_symbols = [" "]

        # TODO: temporary fix before redoing algorithm to handle word_end_symbols
        right_context = " "

        # Figure out the prefix of the current word (if any)
        word_start_index = -1
        for i in range(len(left_context)):
            ch = left_context[i]
            if ch == " ":
                word_start_index = i
        word_prefix = left_context[word_start_index+1:]

        tokens = []
        tokens.extend(self.left_context_tokens)
        # Don't extend if the context is empty, this avoids some models like byt5 from adding extra </s> at start
        if len(left_context) > 0:
            tokens.extend(self._encode(left_context))

        # We can now search forward from the starting state
        # A hypothesis needs to generate the right_context on the right side to finish
        # Hypotheses are stored as a tuple (log prob, text, tokens)
        # It was faster (CPU anyway) to have the token sequence in the hypotheses to avoid re-tokenizing repeatedly from the string

        # Constant indexes for use with the hypotheses tuples
        # log prob is first since we want to use a heap for the finished hypotheses
        LOGP: Final[int] = 0
        STR: Final[int] = 1
        TOKENS: Final[int] = 2

        hypo = (0.0, "", tokens)
        current_hypos = [hypo]
        finished_hypos = []
        best_finished_log_prob = float("-inf")
        pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]

        # TODO: Change to support flexible batch size in a single section of code
        # This version does the search in one giant minibatch
        while len(current_hypos) > 0:
            next_hypos = []
            batch_tensors = []

            # Get length of longest sequence in our batch
            max_length = 0
            for i in range(len(current_hypos)):
                max_length = max(max_length, len(current_hypos[i][2]))

            for i in range(len(current_hypos)):
                tokens = current_hypos[i][TOKENS]
                # Pad out this sentence
                while len(tokens) < max_length:
                    tokens.append(pad_id)
                batch_tensors.append(torch.tensor(tokens))
            tokens_tensor = torch.stack(tuple(batch_tensors)).to(self.device)

            with torch.no_grad():
                logits = self.model(tokens_tensor).logits   # shape (batch_size, max_length, 384)
                # We care about the logits in the last position of second dimension
                # We want to sum to one in the last dimension over the first dimensions (different hypotheses)
                log_probs = torch.log_softmax(logits[:,-1,:], dim=1).detach().cpu().numpy()

            # For through all the current hypotheses and extend them by the symbol set
            # Loop backwords since the most probable is at the end of the heap
            # This should improve our ability to prune
            for i in reversed(range(len(current_hypos))):
                # Limit to just the symbols we care about
                log_probs_current = self._get_symbol_log_probs(log_probs=log_probs[i])

                # Extend the ith current hypothesis by all possible symbols
                for j in range(len(log_probs_current)):
                    new_hypo = (current_hypos[i][LOGP] + log_probs_current[j],
                                current_hypos[i][STR] + self.symbol_set[j],
                                current_hypos[i][TOKENS].copy() + [self.symbol_index_to_vocab_index[j]])
                    # See if we have finished by generating the right context
                    if new_hypo[STR].endswith(right_context):
                        if not nbest or len(finished_hypos) < nbest:
                            # Add if we haven't reached our n-best limit so add
                            heapq.heappush(finished_hypos, (new_hypo[LOGP], new_hypo[STR]))
                        elif new_hypo[LOGP] > finished_hypos[0][LOGP]:
                            # Or replace the worst hypotheses with the new one
                            heapq.heappushpop(finished_hypos, (new_hypo[LOGP], new_hypo[STR]))
                        # See if we need to update the current best log prob of any hypothesis
                        if new_hypo[LOGP] > best_finished_log_prob:
                            best_finished_log_prob = new_hypo[LOGP]
                    # Keep if it is still within beam width of our best hypothesis thus far
                    # But if the n-best list is fully populated then we must beat the current worse on the heap
                    elif (best_finished_log_prob - new_hypo[LOGP]) < beam_logp_best and \
                            (not nbest or len(finished_hypos) < nbest or new_hypo[LOGP] > finished_hypos[0][LOGP]):
                        next_hypos.append(new_hypo)
            current_hypos = next_hypos

        # We want them sorted with the most probable at the front (unlike the heap)
        finished_hypos.sort(key=lambda x: x[0], reverse=True)

        # Remove the right context from the results and add any prefix to the front
        result = []
        for hypo in finished_hypos:
            # Optional return of log probabilities
            word = word_prefix + hypo[1].removesuffix(right_context)
            # TODO: For now we only make lower case predictions
            word = word.lower()
            # Reverse the order so it is word followed optionally by log prob
            if return_log_probs:
                result.append((word, hypo[1]))
            else:
                result.append(word)
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

        context = context = "".join(evidence)

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
