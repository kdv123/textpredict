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
                      word_end_symbols: List[str] = None,
                      nbest: int = None,
                      beam_logp_best: float = None,
                      beam_search_max: int = None,
                      return_log_probs = False) -> List:
        """
        Given some left text context, predict the most likely next words.
        Left and right context use normal space character for any spaces, we convert internally to space symbol, e.g. <sp>
        :param left_context: previous text we are condition on
        :param word_end_symbols: tuple of symbols that we consider to end a word, defaults to just the space character
        :param nbest: number of most likely words to return
        :param beam_logp_best: log-prob beam used during the search, hypothesis with log prob > than this distance from best hypothesis are pruned
        :param beam_search_max: maximum number of hypotheses to track during each extension of search
        :param return_log_probs: whether to return log probabilities of each word
        :return: List of tuples with words and their log probabilities
        """

        # We want each language model class set its own default pruning values
        # We want the client keystroke_savings.py to default to these if pruning switches aren't set
        if beam_logp_best is None:
            beam_logp_best = 5.0
        if beam_search_max is None:
            beam_search_max = 100

        # Since List is a mutable type, we can't set a default reliably in the method declaration
        # We'll set the default of a trailing space if caller didn't specify a list of right contexts
        if word_end_symbols is None:
            word_end_symbols = [" "]

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

        # Reuseable KenLM state, we use this to compute the probability of next character before we decide if we are keeping it
        temp_out_state = kenlm.State()

        # Create a symbol set that also includes any end of word symbols that aren't in our normal symbol set
        # If any of the end symbols occur in the normal symbol set, we include at end of list
        search_symbols = []
        for symbol in self.symbol_set:
            if symbol not in word_end_symbols:
                search_symbols.append(symbol)
        index_first_end_symbol = len(search_symbols)
        for end_symbol in word_end_symbols:
            search_symbols.append(end_symbol)

        # Parallel version of the set of all search symbols that handles any conversion to pseudo-words needed by LM like <sp>
        search_symbols_converted = []
        for symbol in search_symbols:
            if symbol == " ":
                search_symbols_converted.append(self.space_symbol)
            else:
                search_symbols_converted.append(symbol)

        # We store hypotheses in a list with a tuple (log_prob, current word characters, KenLM state)
        current_hypos = [(0.0, "", start_state)]

        LOGP: Final[int] = 0
        STR: Final[int] = 1
        STATE: Final[int] = 2

        # Finished hypotheses map a word string (without ending symbols) to its log prob
        # We use a dictionary since we may want to merge hypotheses that are the same word
        finished_hypos = {}
        best_finished_log_prob = float("-inf")

        # Note: Python's heap pops minimum value, so we are going to explore worst first.
        # Might be better to explore best first, but this is in conflict with the need to easily replace the worst hypothesis.
        while len(current_hypos) > 0:
            # We'll store extended hypotheses in a min heap to make it easy to main only a fixed number of the best
            next_hypos = []

            for hypo in current_hypos:
                # Extend this hypothesis by all possible symbols
                # This could include symbols that were specified to end words but aren't valid character inside of words
                for i in range(len(search_symbols)):
                    # Compute the new log prob and string for our candidate new hypothesis
                    # NOTE: We convert normal characters to any special version in the LM, e.g. " " -> "<sp>"
                    new_log_prob = hypo[LOGP] + self.model.BaseScore(hypo[STATE], search_symbols_converted[i], temp_out_state)

                    # We avoid adding finished or intermediate hypotheses if they are outside log prob beam
                    # This is a bit faster than only doing it for intermediate hypotheses
                    if (best_finished_log_prob - new_log_prob) < beam_logp_best:
                        # See if we have finished by generating any of the valid right symbols
                        # These were organized to be at the end of the list of search_symbols
                        if i >= index_first_end_symbol:
                            # NOTE: we don't add the ending symbol to the finished hypothesis
                            if hypo[STR] in finished_hypos:
                                # If already had this word finish with a different end symbol we sum the probabilities
                                finished_hypos[hypo[STR]] = np.logaddexp(finished_hypos[hypo[STR]], new_log_prob)
                            else:
                                # Haven't seen this word, we will just always add it to the dictionary
                                # It would be expensive to maintain a fixed dictionary size of the best finished hypotheses
                                finished_hypos[hypo[STR]] = new_log_prob
                            # Update the current best log prob of any finishing hypothesis
                            best_finished_log_prob = max(best_finished_log_prob, new_log_prob)
                        # This hypothesis didn't finish
                        # Keep if it is still within beam width of our best hypothesis thus far
                        else:
                            # This hypothesis is within the beam of the best to date
                            # We'll make a copy of the KenLM state object and extend the string
                            if len(next_hypos) < beam_search_max:
                                # Add if we haven't reached our beam width limit so add
                                heapq.heappush(next_hypos, (new_log_prob, hypo[STR] + search_symbols[i], temp_out_state.__copy__()))
                            else:
                                # Or replace the worst hypotheses with the new one
                                heapq.heappushpop(next_hypos, (new_log_prob, hypo[STR] + search_symbols[i], temp_out_state.__copy__()))
            # This slows it down for no improvement in KS
#            next_hypos.sort(key=lambda x: x[LOGP], reverse=True)
            current_hypos = next_hypos

        # Convert our dictionary of finished hypotheses to a sorted list
        sorted_best = sorted(finished_hypos.items(), key=lambda item: item[1], reverse=True)[:nbest]

        # Optional inclusion of log prob in result
        return [(word_prefix + hypo[0], hypo[1]) if return_log_probs else word_prefix + hypo[0] for hypo in sorted_best]

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

        # Limit context based on the context length of the n-gram model
        # This saves doing extra work that won't matter
        if len(context) >= self.model.order:
            context = context[-(self.model.order-1):]

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
