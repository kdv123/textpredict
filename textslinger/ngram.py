from collections import Counter
from typing import Optional, List, Tuple
from textslinger.language_model import LanguageModel
from textslinger.exceptions import InvalidLanguageModelException
import kenlm
import numpy as np


class NGramLanguageModel(LanguageModel):
    """Character n-gram language model using the KenLM library for querying"""

    def __init__(self,
                 symbol_set: List[str],
                 lm_path: str,
                 skip_symbol_norm: Optional[bool] = False):

        super().__init__(symbol_set=symbol_set)
        print(f"Creating n-gram language model, lm_path = {lm_path}")
        self.model = None
        self.lm_path = lm_path
        self.skip_symbol_norm = skip_symbol_norm
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

        # Do not modify the original parameter, could affect mixture model
        context = evidence.copy()

        if len(context) > 11:
            context = context[-11:]

        evidence_str = ''.join(context).lower()

        for i, ch in enumerate(context):
            if ch == ' ':
                context[i] = "<sp>"

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
                f"A valid model path must be provided for the KenLMLanguageModel.\nPath{self.lm_path} is not valid.")

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
            # Replace the space character with KenLM's <sp> token
            if char == ' ':
                score = self.model.BaseScore(state, '<sp>', temp_state)
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
