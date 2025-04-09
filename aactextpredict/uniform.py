"""Uniform language model"""
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from aactextpredict.language_model import LanguageModel, DEFAULT_SYMBOL_SET
from aactextpredict.exceptions import WordPredictionsNotSupportedException

class UniformLanguageModel(LanguageModel):
    """Language model in which probabilities for symbols are uniformly
    distributed.

    Parameters
    ----------
        response_type - SYMBOL only
        symbol_set - optional specify the symbol set, otherwise uses DEFAULT_SYMBOL_SET
    """

    def __init__(self,
                 symbol_set: Optional[List[str]] = DEFAULT_SYMBOL_SET):
        super().__init__(symbol_set=symbol_set)

    def predict_character(self, evidence: Union[str, List[str]]) -> List[Tuple]:
        """
        Using the provided data, compute probabilities over the entire symbol.
        set.

        Parameters
        ----------
            evidence  - list of previously typed symbols

        Returns
        -------
            list of (symbol, probability) tuples
        """
        probs = equally_probable(self.symbol_set)
        return list(zip(self.symbol_set, probs))

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
        """Restore model state from the provided checkpoint"""

def equally_probable(alphabet: List[str],
                     specified: Optional[Dict[str, float]] = None) -> List[float]:
    """Returns a list of probabilities which correspond to the provided
    alphabet. Unless overridden by the specified values, all items will
    have the same probability. All probabilities sum to 1.0.

    Parameters:
    ----------
        alphabet - list of symbols; a probability will be generated for each.
        specified - dict of symbol => probability values for which we want to
            override the default probability.
    Returns:
    --------
        list of probabilities (floats)
    """
    n_letters = len(alphabet)
    if not specified:
        return np.full(n_letters, 1 / n_letters)

    # copy specified dict ignoring non-alphabet items
    overrides = {k: specified[k] for k in alphabet if k in specified}
    assert sum(overrides.values()) < 1

    prob = (1 - sum(overrides.values())) / (n_letters - len(overrides))
    # override specified values
    return [overrides[sym] if sym in overrides else prob for sym in alphabet]
