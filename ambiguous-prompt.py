from collections import Counter
import torch
import heapq
from typing import Final, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from language_model import LanguageModel
from language_model import BACKSPACE_CHAR, SPACE_CHAR
from exceptions import InvalidLanguageModelException
from scipy.special import softmax
import time

class AmbiguousLanguageModel(LanguageModel):
    """Word language model based on disambiguating FlexType character groups"""

    def __init__(self,
                 symbol_set: List[str],
                 lang_model_name: str,
                 lm_path: str = None,
                 lm_device: str = "cpu",
                 lm_left_context: str = "",
                 beam_width: int = 8,
                 fp16: bool = True,
                 mixed_case_context = True,
                 case_simple = True,
                 max_completed = 100
                 ):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type      - SYMBOL only
            symbol_set         - list of symbol strings
            lang_model_name    - name of the Hugging Face casual language model to load
            lm_path            - load fine-tuned model from specified directory
            lm_device          - device to use for making predictions (cpu, mps, or cuda)
            lm_left_context    - text to condition start of sentence on
            beam_width         - how many hypotheses to keep during the search
            fp16               - convert model to fp16 to save memory/compute on CUDA
            mixed_case_context - use mixed case for language model left context
            case_simple        - simple fixing of left context case
            max_completed      - stop search once we reach this many completed hypotheses, default 1000
        """
        super().__init__(symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.valid_vocab = []
        self.vocab = {}
        self.longest_token = 0
        self.index_to_word = {}
        self.index_to_word_lower = {}
        self.symbol_set_lower = None
        self.device = lm_device
        self.left_context = lm_left_context
        self.fp16 = fp16
        self.mixed_case_context = mixed_case_context
        self.case_simple = case_simple
        self.max_completed = max_completed

        # We optionally load the model from a local directory, but if this is not
        # specified, we load a Hugging Face model
        self.model_name = lang_model_name
        self.model_dir = lm_path if lm_path else self.model_name

        # parameters for search
        self.beam_width = beam_width

        self.simple_upper_words = {"i": "I",
                                    "i'll": "I'll",
                                    "i've": "I've",
                                    "i'd": "I'd",
                                    "i'm": "I'm"}
        self.load()

    def _build_vocab(self) -> None:
        """
        Build a vocabulary table mapping token index to word strings
        """

        for i in range(self.vocab_size):
            word = self.tokenizer.decode([i])
            word_lower = word.lower()
            self.index_to_word[i] = word
            self.index_to_word_lower[i] = word_lower
            valid = True
            for ch in word_lower:
                # The space char is only valid once we convert spaces to the space char
                if ch == SPACE_CHAR:
                    valid = False
                    break
                if ch == ' ':
                    continue
                elif ch not in self.symbol_set_lower:
                    valid = False
                    break
            if valid:
                self.valid_vocab += i,
                length = len(word)
                if length > self.longest_token:
                    self.longest_token = length
                for j in range(length):
                    key = word_lower[0:j + 1].replace(' ', SPACE_CHAR)
                    if key not in self.vocab:
                        self.vocab[key] = []
                    self.vocab[key] += i,

        # Get the index we use for the start or end pseudo-word
        if self.left_context == "":
            if "gpt2" in self.model_name:
                self.left_context = "<|endoftext|>"
            else:
                self.left_context = "</s>"
        # Get token id(s) for the left context we condition all sentences on
        self.left_context_tokens = self._encode(self.left_context)
        # print(f"left_context_tokens = {self.left_context_tokens}")

    def _encode(self, text: str) -> List[int]:
        if text == "":
            return []

        tokens = self.tokenizer.encode(text)
        if len(tokens) > 1 and (self.model_name.startswith("facebook/opt") or self.model_name.startswith("figmtu/opt")):
            tokens = tokens[1:]

        return tokens

    def predict(self, evidence: List[str], input: List[int]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of strings or characters, previous typed text
            input    - a list of integer group numbers input by the user
        Response:
            A list of words with log likelihood
        """

        assert self.model is not None, "language model does not exist!"

        converted_context = "".join(evidence)
        converted_context_lower = converted_context.lower()

        context = converted_context.replace(SPACE_CHAR, ' ')

        if self.case_simple and len(context) > 0:
            cased_context = ""
            words = context.split()
            for i, word in enumerate(words):
                if i == 0 and word[0] >= 'a' and word[0] <= 'z':
                    word = word[0].upper() + word[1:]
                if i > 0:
                    if word in self.simple_upper_words:
                        word = self.simple_upper_words[word]
                    cased_context += " "
                cased_context += word
            # Handle ending space in the context
            if context[-1] == ' ':
                cased_context += " "
            #print(f"Simple casing of left context, from '{context}' to '{cased_context}'")
            context = cased_context

        else:
            context = context.lower()

        context_lower = context.lower()

        tokens = []
        tokens.extend(self.left_context_tokens)

        # Optionally, we condition on upper and lower case left context
        if not self.mixed_case_context:
            context = context.lower()
        tokens.extend(self._encode(context))

        prompt = ["[GRP]"]
        prompt += [str(x) for x in input]
        prompt += ["[PRED]"]
        tokens.extend(self._encode("".join(prompt)))

        print(f"Final tokens: {tokens}")

        # Constant indexes for use with the hypotheses tuples
        LOGP: Final[int] = 0
        SEQ: Final[int] = 1
        LEN: Final[int] = 2

        # Our starting hypothesis that we'll be extending.
        # Format is (log likelihood, token id sequence).
#       start_length = 0
#       for x in tokens[len(self.left_context_tokens):]:
#           start_length += len(self.index_to_word_lower[x])
        current_hypos = [(0.0, tokens)]

        # We use a priority queue to track the top hypotheses during the beam search.
        # For a beam of 8, empirical testing showed this was about the same amount
        # of time as a simpler list that used a linear search to replace when
        # full.
        heapq.heapify(current_hypos)

        # Add new extended hypotheses to this heap
        next_hypos = []

        # Tracks completed hypotheses
        predictions = {}

        # Used to signal to while loop to stop the search
        done = False

        # Start a beam search forward from the backed off token sequence.
        # Each iteration of this while loop extends hypotheses by all valid tokens.
        # We only keep at most self.beam_width hypotheses in the valid heap.
        # Stop extending search once we reach our max completed target.
        while len(current_hypos) > 0 and not done:
            # We'll explore hypothesis in order from most probable to least.
            # This has little impact on how long it takes since this is only sorting a small number of things.
            # But it is important with max_completed pruning since we want to
            # bias for completing high probability things.
            current_hypos.sort(reverse=True)

            print(f"Current hypotheses: {current_hypos}")

            # Work on the hypotheses from the last round of extension.
            # Create the torch tensor for the inference with a row for each
            # hypothesis.
            tokens_tensor = torch.tensor([x[SEQ] for x in current_hypos]).reshape(
                len(current_hypos), -1).to(self.device)

#            before_inference_ns = time.time_ns()
            # Ask the LLM to predict tokens that come after our current set of
            # hypotheses
            with torch.no_grad():
                # Compute the probabilities from the logits
                log_probs = torch.log_softmax(self.model(
                    tokens_tensor).logits[:, -1, :], dim=1)

                # Create a big 2D tensor where each row is that hypothesis' current likelihood.
                # First create a list of just the hypotheses' likelihoods.
                # Then reshape to be a column vector.
                # Then duplicate the column based on the number of subword
                # tokens in the LLM.
                add_tensor = torch.tensor([x[LOGP] for x in current_hypos]).reshape(
                    (log_probs.size()[0], 1)).repeat(1, log_probs.size()[1]).to(self.device)

                # Add the current likelihoods with each subtoken's probability.
                # Move it back to the CPU and convert to numpy since this makes
                # it a lot faster to access for some reason.
                new_log_probs = torch.add(
                    log_probs, add_tensor).detach().cpu().numpy()
            #self.predict_inference_ns += time.time_ns() - before_inference_ns

            for current_index, current in enumerate(current_hypos):
            
                for token_id in range(self.vocab_size):
                
                    # For a hypothesis to finish it must generate the end token
                    if token_id in self.end_token:
                        pred = ""
                        for x in current[SEQ][len(tokens):]:
                            pred += self.index_to_word_lower[x]
                        
                        # Add this likelihood to the list for finished predictions.
                        if len(pred):
                            predictions[pred] = new_log_probs[current_index][token_id],
                            print(f"Completed prediction: (\"{pred}\", {current[LOGP]})")
                    elif not self.beam_width or len(next_hypos) < self.beam_width:
                        # If we are under the beam limit then just add it
                        heapq.heappush(next_hypos,
                                       (new_log_probs[current_index][token_id],
                                        current[SEQ] + [token_id]))
                    elif new_log_probs[current_index][token_id] > next_hypos[0][LOGP]:
                        # Or replace the worst hypotheses with the new one
                        heapq.heappushpop(next_hypos,
                                          (new_log_probs[current_index][token_id],
                                           current[SEQ] + [token_id]))

                    # Break out of the for loop over hypotheses and while loop if
                    # we reach our max completed goal
                    if self.max_completed and len(predictions) >= self.max_completed:
                        done = True
                        break

            # Swap in the extended set as the new current working set
            current_hypos = next_hypos
            next_hypos = []
            
        return list(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        """
        if self.model_dir:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
            except BaseException:
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
                    # Tokenizer vocab should be modified at TRAIN time, and should not need to be modified here if
                    # loaded properly
                    #special_tokens = {'additional_special_tokens': ['[AMB]', '[GRP]', '[PRED]', '[END]']}
                    #num_added = self.tokenizer.add_special_tokens(special_tokens)
                except BaseException:
                    raise InvalidLanguageModelException(f"{self.model_dir} does not contain a valid tokenizer and {self.model_name} is not a valid model identifier on HuggingFace.")
        self.vocab_size = len(self.tokenizer)
        self.end_token = self._encode('[END]')
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            if self.fp16 and self.device == "cuda":
                self.model = self.model.half()
            #model.resize_token_embeddings(len(tokenizer))
        except:
            raise InvalidLanguageModelException(f"{self.model_dir} is not a valid local folder or model identifier on HuggingFace.")

        self.model.eval()

        self.model.to(self.device)

        self.symbol_set_lower = []
        for ch in self.symbol_set:
            if ch is SPACE_CHAR:
                self.symbol_set_lower.append(SPACE_CHAR)
            elif ch is BACKSPACE_CHAR:
                continue
            else:
                self.symbol_set_lower.append(ch.lower())

        self._build_vocab()

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbol with probability
        """
        word_pred = self.predict(evidence)

        return word_pred

    def get_num_parameters(self) -> int:
        """
            Find out how many parameters the loaded model has
        Args:
        Response:
            Integer number of parameters in the transformer model
        """
        return sum(p.numel() for p in self.model.parameters())
