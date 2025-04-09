"""Defines the language model base class."""
from abc import ABC, abstractmethod
from typing import List, Tuple
from string import ascii_uppercase, ascii_lowercase

# Returns all uppercase English letters
def alphabet_upper():
    """Alphabet Upper

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    return list(ascii_uppercase)

# Returns all lowercase English letters
def alphabet_lower():
    """Alphabet Lower
    
    Function used to standardize the symbols we use as alphabet
    
    Returns
    -------
        array of letters.
    """
    return list(ascii_lowercase)

# Returns all English letters in mixed case
def alphabet_mixed():
    """Alphabet Mixed
    
    Function used to standardize the symbols we use as alphabet
    
    Returns
    -------
        array of letters.
    """
    return alphabet_lower() + alphabet_upper()

DEFAULT_SYMBOL_SET = alphabet_upper() + [' ']

class LanguageModel(ABC):
    """Parent class for Language Models."""

    symbol_set: List[str] = None

    def __init__(self,
                 symbol_set: List[str] = DEFAULT_SYMBOL_SET):
        self.symbol_set = symbol_set

    @classmethod
    def name(cls) -> str:
        """Model name used for configuration"""
        suffix = 'LanguageModel'
        if cls.__name__.endswith(suffix):
            return cls.__name__[0:-len(suffix)].upper()
        return cls.__name__.upper()

    @abstractmethod
    def predict_character(self, evidence: List[str]) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods over the entire symbol set.
        Args:
            evidence - ['A', 'B']

        Response:
            probability - dependent on response type, a list of words or symbols with probability
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model from the provided assets/path"""
        ...