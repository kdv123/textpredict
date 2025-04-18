"""Defines the language model base class."""
from abc import ABC, abstractmethod
from typing import List, Tuple
from string import ascii_uppercase

# Eventually these should go away, but for now leaving to test existing code
SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'


# Eventually replace
def alphabet():
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    return list(ascii_uppercase) + [BACKSPACE_CHAR, SPACE_CHAR]


class LanguageModel(ABC):
    """Parent class for Language Models."""

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
    def update(self) -> None:
        """Update the model state"""
        ...

    @abstractmethod
    def load(self) -> None:
        """Restore model state from the provided checkpoint"""
        ...

    def reset(self) -> None:
        """Reset language model state"""
        ...

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """Update state by predicting and updating"""
