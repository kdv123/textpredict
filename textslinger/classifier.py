from collections import Counter
import torch
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textslinger.language_model import LanguageModel
from textslinger.exceptions import InvalidLanguageModelException, WordPredictionsNotSupportedException
from scipy.special import softmax


class ClassifierLanguageModel(LanguageModel):
    """Character language model based on a transformer with a classification head"""

    def __init__(self,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = 8,
                 batch_size: int = 8,
                 fp16: bool = False,
                 mixed_case_context = False,
                 case_simple = False,
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
            beam_width         - how many hypotheses to keep during the search
            batch_size         - how many sequences to pass in at a time during inference
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
        """
        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = {}
        self.longest_token = 0
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

        # parameters for search
        self.beam_width = beam_width
        self.batch_size = batch_size

        self.simple_upper_words = {"i": "I",
                                    "i'll": "I'll",
                                    "i've": "I've",
                                    "i'd": "I'd",
                                    "i'm": "I'm"}
        self.load()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower
            valid = True
            for ch in word_lower:
                if ch not in self.symbol_set_lower:
                    valid = False
                    break
            if valid:
                self.valid_vocab += i,
                length = len(word)
                if length > self.longest_token:
                    self.longest_token = length
                for j in range(length):
                    key = word_lower[0:j + 1]
                    if key not in self.vocab:
                        self.vocab[key] = []
                    self.vocab[key] += i,

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            else:
                self.left_context = "</s>"
        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        # print(f"left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        if text == "":
            return []

        tokens = self.tokenizer.encode(text)
        if len(tokens) > 1 and (self.model_name.startswith("facebook/opt") or self.model_name.startswith("figmtu/opt")):
            tokens = tokens[1:]

        return tokens

    def predict_character(self, evidence: List[str]) -> List[Tuple]:
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
            #print(f"Simple casing of left context, from '{context}' to '{cased_context}'")
            context = cased_context

        else:
            context = context.lower()

        tokens = []
        tokens.extend(self.left_context_tokens)

        # Optionally, we condition on upper and lower case left context
        if not self.mixed_case_context:
            context = context.lower()
        tokens.extend(self._encode(context))

        tensor = torch.tensor([tokens]).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor).logits
            char_probs = torch.softmax(logits, dim=1).to("cpu").numpy()[0]

        keys = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' "]
        next_char_pred = Counter(dict(zip(keys, char_probs)))

        for low in self.symbol_set_lower:
            if low.isalpha():
                next_char_pred[low.upper()] += next_char_pred[low]
                del next_char_pred[low]

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))

    def predict_word(self, 
                     left_context: List[str], 
                     right_context: List[str] = [" "],
                     nbest: int = 3,
                     ) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods over the next sequence of symbols
        Args:
            left_context - The text that precedes the desired prediction.
            right_context - The text that will follow the desired prediction. For simple word
                predictions, this should be a single space.
            nbest - The number of top predictions to return

        Response:
            A list of tuples, (predicted text, log probability)
        """
        raise WordPredictionsNotSupportedException("Word predictions are not supported for this model.")

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
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
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

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())
