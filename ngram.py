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
                 skip_symbol_norm: Optional[bool] = False,
                 space_symbol: str = "<sp>",
                 sentence_end: str = "</s>"):
        """
        Construct an instance of NGramLanguageModel.
        :param symbol_set: symbols we want to make predictions over
        :param lm_path: location of the KenLM format n-gram language model
        :param skip_symbol_norm: don't normalize character prediction distribution to just symbol_set
        :param space_symbol: pseudo-word for spaces between words
        :param sentence_end: pseudo-word for end of sentence event
        """
        super().__init__(symbol_set=symbol_set)
        print(f"Creating n-gram language model, lm_path = {lm_path}")
        self.model = None
        self.lm_path = lm_path
        self.skip_symbol_norm = skip_symbol_norm
        self.load()
        self.space_symbol = space_symbol
        self.sentence_end = sentence_end

        # Create a parallel version of symbol_set that does any conversion required to bring plain characters into the n-gram's vocab
        self.symbol_set_converted = []
        for symbol in symbol_set:
            if symbol == " ":
                self.symbol_set_converted.append(self.space_symbol)
            else:
                self.symbol_set_converted.append(symbol)

    def predict_words(self,
                      left_context: str,
                      right_context: str = " ",
                      nbest: int = None,
                      beam: float = 3.0,
                      return_log_probs = False) -> List:
        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to space symbol, e.g. <sp>
        :param left_context: previous text we are condition on
        :param right_context: characters that must appear right of our predicted next word
        :param nbest: number of most likely words to return
        :param beam: log-prob beam used during the search
        :param return_log_probs: whether to return log probabilities of each word
        :return: List of tuples with words and their log probabilities
        """
        state1 = kenlm.State()
        state2 = kenlm.State()
        # Condition the LM on the sentence start token since we may not have enough left context to move <s> out of the markov window
        self.model.BeginSentenceWrite(state1)

        # Update the state one character at a time for the left context
        # Also note the last space so we can figure out the prefix of the current word (if any)
        # Shorten to one character less than the order of the n-gram model
        truncated_left_context = left_context[-self.model.order+1:]

        # Advance the language model state base on the left context
        for i in range(len(truncated_left_context)):
            ch = truncated_left_context[i]
            if ch == " ":
                ch = self.space_symbol
            if i % 2 == 0:
                self.model.BaseScore(state1, ch, state2)
            else:
                self.model.BaseScore(state2, ch, state1)
        if len(truncated_left_context) % 2 == 0:
            start_state = state1
        else:
            start_state = state2

        # Find the rightmost space
        word_start_index = -1
        for i in range(len(left_context)):
            if left_context[i] == " ":
                word_start_index = i
        word_prefix = left_context[word_start_index+1:]

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

        # Reuseable KenLM state, we use this to compute the probability of next character before we decide if we are keeping it
        temp_out_state = kenlm.State()

        while len(current_hypos) > 0:
            next_hypos = []
            # Loop backwords since the most probable is at the end of the heap
            # This should improve our ability to prune
            for i in reversed(range(len(current_hypos))):
                hypo = current_hypos[i]
                # Extend this hypothesis by all possible symbols
                for j in range(len(self.symbol_set)):
                    # Compute the new log prob and string for our candidate new hypothesis
                    # NOTE: We convert normal characters to any special version in the LM, e.g. " " -> "<sp>"
                    new_log_prob = hypo[LOGP] + self.model.BaseScore(hypo[STATE], self.symbol_set_converted[j], temp_out_state)
                    new_str = hypo[STR] + self.symbol_set[j]

                    # See if we have finished by generating the right context
                    if new_str.endswith(right_context):
                        if not nbest or len(finished_hypos) < nbest:
                            # Add if we haven't reached our n-best limit so add
                            heapq.heappush(finished_hypos, (new_log_prob, new_str))
                        elif new_log_prob > finished_hypos[0][LOGP]:
                            # Or replace the worst hypotheses with the new one
                            heapq.heappushpop(finished_hypos, (new_log_prob, new_str))
                        # See if we need to update the current best log prob of any hypothesis
                        if new_log_prob > best_finished_log_prob:
                            best_finished_log_prob = new_log_prob
                    # Keep if it is still within beam width of our best hypothesis thus far
                    # But if the n-best list is fully populated then we must beat the current worst hypothesis on the heap (first element)
                    elif (best_finished_log_prob - new_log_prob) < beam and \
                            (not nbest or len(finished_hypos) < nbest or new_log_prob > finished_hypos[0][LOGP]):
                        # Now that we know we are keeping it, we'll make a copy of the KenLM state object
                        next_hypos.append((new_log_prob, new_str, temp_out_state.__copy__()))
            # Swap in the next hypotheses for the current so the outer loop keeps going
            current_hypos = next_hypos

        # Reverse the order to get the most probable first
        finished_hypos.sort(key=lambda x: x[LOGP], reverse=True)

        # Remove the right context from the results and add any prefix to the front
        result = []
        for hypo in finished_hypos:
            # Optional return of log probs in our result list
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
                context[i] = self.space_symbol

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
            if char != BACKSPACE_CHAR:
                # Replace the space character to whatever symbol is used in the n-gram model, e.g. "<sp>"
                if char == SPACE_CHAR:
                    score = self.model.BaseScore(state, self.space_symbol, temp_state)
                else:
                    score = self.model.BaseScore(state, char.lower(), temp_state)

                # BaseScore returns log probs, convert by putting 10 to its power
                next_char_pred[char] = pow(10, score)

        # We can optionally disable normalization over our symbol set
        # This is useful if we want to compare against SRILM with a LM with a larger vocab
        if not self.skip_symbol_norm:
            sum_probs = np.sum(list(next_char_pred.values()))
            for char in self.symbol_set:
                next_char_pred[char] /= sum_probs

        return list(sorted(next_char_pred.items(), key=lambda item: item[1], reverse=True))
