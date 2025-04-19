from collections import Counter
from typing import Optional, List, Tuple, Final
from language_model import LanguageModel
from language_model import BACKSPACE_CHAR, SPACE_CHAR
from exceptions import InvalidLanguageModelException
import kenlm
import numpy as np
import heapq

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

    def predict_words(self,
                      left_context: str,
                      right_context: str = " ",
                      nbest: int = None,
                      beam: float = 3.0,
                      return_log_probs = False) -> List:
        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to <sp>
        :param left_context: previous text we are condition on
        :param right_context: characters that must appear right of our predicted next word
        :param nbest: number of most likely words to return
        :param beam: log-prob beam used during the search
        :param return_log_probs: whether to return log probabilities of each word
        :return: List of tuples with words and their log probabilities
        """
        state1 = kenlm.State()
        state2 = kenlm.State()
        # This conditions the LM on the sentence start token
        self.model.BeginSentenceWrite(state1)

        # Update the state one character at a time for the left context
        # Also note the last space so we can figure out the prefix of the current word (if any)
        # TODO: If we knew the order of the KenLM model, we might be able to truncate left_context
        word_start_index = -1
        for i in range(len(left_context)):
            ch = left_context[i]
            if ch == " ":
                word_start_index = i
                ch = "<sp>"
            if i % 2 == 0:
                log_prob = self.model.BaseScore(state1, ch, state2)
            else:
                log_prob = self.model.BaseScore(state2, ch, state1)
        word_prefix = left_context[word_start_index+1:]
        if len(left_context) % 2 == 0:
            start_state = state1
        else:
            start_state = state2

        # Constant indexes for use with the hypotheses tuples
        # log prob is first since we want to use a heap for the finished hypotheses
        LOGP: Final[int] = 0
        STR: Final[int] = 1
        STATE: Final[int] = 2

        # We can now search forward from the starting state
        # A hypothesis needs to generate the right_context on the right side to finish
        # Hypotheses are stored as a tuple (log prob, text, starting KenLM state)
        hypo = (0.0, "", start_state)
        current_hypos = [hypo]
        finished_hypos = []
        best_finished_log_prob = float("-inf")

        while len(current_hypos) > 0:
            next_hypos = []
            for hypo in current_hypos:
                # Extend this hypothesis by all possible symbols
                for ch in self.symbol_set:
                    # TODO: Can we reuse the KenLM state objects?
                    out_state = kenlm.State()
                    if ch == " ":
                        use_ch = "<sp>"
                    else:
                        use_ch = ch
                    log_prob = self.model.BaseScore(hypo[STATE], use_ch, out_state)
                    new_hypo = (hypo[LOGP] + log_prob, hypo[STR] + ch, out_state)
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
                    elif (best_finished_log_prob - new_hypo[LOGP]) < beam and \
                            (not nbest or len(finished_hypos) < nbest or new_hypo[LOGP] > finished_hypos[0][LOGP]):
                        next_hypos.append(new_hypo)

            current_hypos = next_hypos

        # Reverse the order to get the most probable first
        finished_hypos.sort(key=lambda x: x[LOGP], reverse=True)

        # Remove the right context from the results and add any prefix to the front
        result = []
        for hypo in finished_hypos:
            # Optional return of log probabilities
            if return_log_probs:
                result.append((word_prefix + hypo[STR].removesuffix(right_context), hypo[LOGP]))
            else:
                result.append(word_prefix + hypo[STR].removesuffix(right_context))
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

        # Do not modify the original parameter, could affect mixture model
        context = evidence.copy()

        if len(context) > 11:
            context = context[-11:]

        evidence_str = ''.join(context).lower()

        for i, ch in enumerate(context):
            if ch == SPACE_CHAR:
                context[i] = "<sp>"

        self.model.BeginSentenceWrite(self.state)

        # Update the state one token at a time based on evidence, alternate states
        for i, token in enumerate(context):
            if i % 2 == 0:
                score = self.model.BaseScore(self.state, token.lower(), self.state2)
            else:
                self.model.BaseScore(self.state2, token.lower(), self.state)

        next_char_pred = None

        # Generate the probability distribution based on the final state
        if len(context) % 2 == 0:
            next_char_pred = self.prob_dist(self.state)
        else:
            next_char_pred = self.prob_dist(self.state2)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

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

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text and outputs probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probabilities
        """
        next_char_pred = self.predict(evidence)

        return next_char_pred

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
            # Backspace probability under the LM is 0
            if char == BACKSPACE_CHAR:
                next

            # Replace the space character with KenLM's <sp> token
            if char == SPACE_CHAR:
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
