"""Defines the language model base class."""
from abc import ABC, abstractmethod
from typing import List, Tuple

def compute_max_hypo_len(left_context: str,
                         max_word_len: int,
                        ) -> int:
    """Compute the maximum length of hypotheses based on existing prefix of word (if any)
    :param left_context: Previous typed text including any prefix of the current word
    :param max_word_len: Maximum length of word we want to predict
    """
    prefix_len = 0
    if len(left_context) > 0:
        pos = len(left_context) - 1
        while left_context[pos] != " " and pos >= 0:
            pos -= 1
            prefix_len += 1
    return max(0, max_word_len - prefix_len)

class LanguageModel(ABC):
    """Parent class for language model classes"""

    symbol_set: List[str] = None

    def __init__(self,
                 symbol_set: List[str]):
        self.symbol_set = symbol_set

    @classmethod
    def name(cls) -> str:
        """Model name used for configuration"""
        suffix = 'LanguageModel'
        if cls.__name__.endswith(suffix):
            return cls.__name__[0:-len(suffix)].upper()
        return cls.__name__.upper()

    @abstractmethod
    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods over the entire symbol set.
        Args:
            evidence - ['A', 'B']

        Response:
            probability - dependent on response type, a list of words or symbols with probability
        """
        ...

    @abstractmethod
    def predict_words(self,
                      left_context: str,
                      word_end_symbols: List[str] = None,
                      nbest: int = None,
                      beam_logp_best: float = None,
                      beam_search_max: int = None,
                      max_word_len: int = None,
                      return_log_probs=False) -> List:
        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to <sp>
        :param left_context: previous text we are conditioning on, note this includes the prefix of the current word
        :param word_end_symbols: tuple of symbols that we consider to end a word, defaults to just the space character
        :param nbest: number of most likely words to return
        :param beam_logp_best: log-prob beam used during the search, hypothesis with log prob > than this distance from best hypothesis are pruned
        :param beam_search_max: maximum number of hypotheses to track during each extension of search
        :param max_word_len: maximum length of words that can be predicted
        :param return_log_probs: whether to return log probabilities of each word
        :return: List of tuples with words and (optionally) their log probabilities
        """
        ...
