from typing import List, Tuple
import json
from language_model import LanguageModel
from language_model import BACKSPACE_CHAR, SPACE_CHAR
from exceptions import InvalidLanguageModelException


class UnigramLanguageModel(LanguageModel):
    """Character language model based on trained unigram weights"""

    def __init__(self,
                 symbol_set: List[str],
                 lm_path: str):

        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.lm_path = lm_path

        try:
            with open(self.lm_path) as json_file:
                self.unigram_lm = json.load(json_file)
        except BaseException:
            raise InvalidLanguageModelException("Unable to load Unigram model from file")

        self.unigram_lm[SPACE_CHAR] = self.unigram_lm.pop("SPACE_CHAR")
        self.unigram_lm[BACKSPACE_CHAR] = self.unigram_lm.pop("BACKSPACE_CHAR")

        if not set(self.unigram_lm.keys()) == set(self.symbol_set):
            raise InvalidLanguageModelException("Invalid unigram model symbol set!")

        self.unigram_lm = sorted(self.unigram_lm.items(), key=lambda item: item[1], reverse=True)

        self.load()

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        return self.unigram_lm

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """