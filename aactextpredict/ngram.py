from collections import Counter
from typing import Optional, List, Tuple
from aactextpredict.language_model import LanguageModel
from aactextpredict.exceptions import InvalidLanguageModelException, WordPredictionsNotSupportedException
import kenlm
import numpy as np


class NGramLanguageModel(LanguageModel):
    """Character n-gram language model using the KenLM library for querying"""

    space_symbol = None

    def __init__(self,
                 symbol_set: List[str],
                 lm_path: str,
                 skip_symbol_norm: Optional[bool] = False,
                 start_symbol: str = "<s>",
                 end_symbol: str = "</s>",
                 space_symbol: str = "<sp>"):

        super().__init__(symbol_set=symbol_set)
        print(f"Creating n-gram language model, lm_path = {lm_path}")
        self.model = None
        self.lm_path = lm_path
        self.skip_symbol_norm = skip_symbol_norm
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.space_symbol = space_symbol
        self.load()

    def predict_character(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        # Do not modify the original parameter, could affect mixture model
        context = evidence.copy()

        if len(context) > 11:
            context = context[-11:]

        for i, ch in enumerate(context):
            if ch == ' ':
                context[i] = self.space_symbol

        self.model.BeginSentenceWrite(self.state)

        # Update the state one token at a time based on evidence, alternate states
        for i, token in enumerate(context):
            if i % 2 == 0:
                self.model.BaseScore(self.state, token.lower(), self.state2)
            else:
                self.model.BaseScore(self.state2, token.lower(), self.state)

        next_char_pred = None

        # Generate the probability distribution based on the final state
        if len(context) % 2 == 0:
            next_char_pred = self.prob_dist(self.state)
        else:
            next_char_pred = self.prob_dist(self.state2)

        return next_char_pred

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
            Load the language model, initialize state variables
        Args:
            path: language model file path
        """

        try:
            self.model = kenlm.LanguageModel(self.lm_path)
        except BaseException:
            raise InvalidLanguageModelException(
                f"A valid model path must be provided for the KenLMLanguageModel.\nPath {self.lm_path} is not valid.")

        self.state = kenlm.State()
        self.state2 = kenlm.State()

    def prob_dist(self, state: kenlm.State) -> List[Tuple]:
        """
            Take in a state and generate the probability distribution of next character
        Args:
            state - the kenlm state updated with the evidence
        Response:
            A list of symbols with probability
        """
        next_char_pred = Counter()

        temp_state = kenlm.State()

        for char in self.symbol_set:
            # Replace the space character with whatever space token the model uses
            if char == ' ':
                score = self.model.BaseScore(state, self.space_symbol, temp_state)
            else:
                score = self.model.BaseScore(state, char.lower(), temp_state)

            # BaseScore returns log probs, convert by putting 10 to its power
            next_char_pred[char] = pow(10, score)

        # We can optionally disable normalization over our symbol set
        # This is useful if we want to compare against SRILM with a LM with a larger vocab
        if not self.skip_symbol_norm:
            sum = np.sum(list(next_char_pred.values()))
            for char in self.symbol_set:
                next_char_pred[char] /= sum

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))
