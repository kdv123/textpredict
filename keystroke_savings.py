#!/usr/bin/env python
# Computes keystroke savings of typing a set of phrases assume optimal use of 1 or more word predictions.
# Supports the following types of language models:
#  1) n-gram character, via KenLM library
#  2) ByGPT5 byte tokenized LLM, via Hugging Face plus uniformers library

from ngram import NGramLanguageModel
from causal_byte import CausalByteLanguageModel
from timeit import default_timer as timer
import argparse
from datetime import datetime
from socket import gethostname
import re
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases", type=str, required=True, help="Input text file with phrases")
    parser.add_argument("--phrase-limit", type=int, help="Max phrases to evaluate")
    parser.add_argument("--lm", type=str, help="Filename of n-gram model to load")
    parser.add_argument("--lower", action="store_true", help="Lowercase the phrases")
    parser.add_argument("--strip", action="store_true", help="Strip symbols except apostrophe")
    parser.add_argument("--nbest", type=int, help="N-best list size", default=3)
    parser.add_argument("--beam", type=float, help="Beam for search, log-prob", default=3.0)
    parser.add_argument("--symbols", type=str, default="abcdefghijklmnopqrstuvwxyz' ", help="Valid symbols")
    parser.add_argument("--model-name", help="Model name of LLM")
    parser.add_argument("--model-dir", help="Local directory to load fine-tuned LLM")
    parser.add_argument("--byte", action="store_true", help="LLM uses byte tokenization")
    parser.add_argument("--fp16", action="store_true", help="Convert LLM to fp16 (CUDA only)")
    parser.add_argument("--case-simple", action="store_true", default=False, help="Simple automatic casing of left context")
    parser.add_argument("--use-mps", action="store_true", help="Use MPS Apple Silicon GPU during inference")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA GPU during inference")
    parser.add_argument("--batch-size", type=int, help="Limit batch size during LLM inference")

    args = parser.parse_args()

    if not args.lm and not args.model_name:
        print(f"ERROR: Must specify either --lm  or --model_name")
        sys.exit(1)

    # Handy stuff to print out in our log files
    print(f"START: {datetime.now()}")
    print(f"ARGS: {args}")
    print(f"HOSTNAME: {gethostname()}")

    # Read in the input file with sentences
    phrase_file = open(args.phrases, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()
    # We may want to limit to only the first so many phrases
    if args.phrase_limit:
        while len(phrases) > args.phrase_limit:
            phrases.pop()
    print(f"Number phrases loaded: {len(phrases)}")

    start = timer()
    symbols = list(args.symbols)
    print(f"Symbols, size {len(symbols)}: {symbols}")

    lm = None
    device = "cpu"
    if args.use_mps:
        device = "mps"
    elif args.use_cuda:
        device = "cuda"

    if args.lm:
        print(f"Loading n-gram LM: {lm}")
        lm = NGramLanguageModel(symbols, args.lm, False)
    elif args.byte:
        if args.model_dir:
            print(f"Loading byte LLM: {args.model_name} from {args.model_dir}")
        else:
            print(f"Loading byte LLM: {args.model_name}")
        lm = CausalByteLanguageModel(symbol_set=symbols,
                                     lang_model_name=args.model_name,
                                     lm_device=device,
                                     lm_path=args.model_dir,
                                     lm_left_context="",
                                     fp16=args.fp16,
                                     mixed_case_context=False,
                                     case_simple=args.case_simple,
                                     normal_space=True,
                                     batch_size=args.batch_size)

    print(f"Model load time = {timer() - start:.2f}")

    total_chars = 0
    total_keystrokes = 0

    # Iterate over all the phrases
    for i, phrase in enumerate(phrases):
        phrase = phrase.strip()
        if args.lower:
            phrase = phrase.lower()
        if args.strip:
            phrase = re.sub(r'[^a-zA-Z \']', '', phrase)
        total_chars += len(phrase)


        print(f"*** Phrase {i}: {phrase}, len: {len(phrase)}")
        # Iterate over all character positions in the phrase
        j = 0
        phrase_keystrokes = 0
        while j < len(phrase):
            left_context = phrase[0:j]
            # Figure out the target word
            # If the next letter is space, then our target is the current word
            if j> 0 and phrase[j] == " ":
                k = j - 1
            else:
                k = j
            # Back up until we hit space or start of string
            while k > 0 and phrase[k] != " ":
                k -= 1
            target_word = ""
            # Go forward until we hit a space or end of string
            if phrase[k] == " ":
                k += 1
            while k < len(phrase) and phrase[k] != " ":
                target_word += phrase[k]
                k += 1
            words = lm.predict_words(left_context, nbest=args.nbest, beam=args.beam)
            print_words = ""
            for word in words:
                if word == target_word:
                    print_words += f" {word.upper()}"
                else:
                    print_words += f" {word}"
            print(f" predictions:{print_words}, left '{left_context}', target '{target_word}', keys {phrase_keystrokes}")

            # See if we can get our target word via a prediction slot
            if target_word in words:
                print(f" SELECTED: {target_word}")
                # Advance to space or end of phrase
                while j < len(phrase) and phrase[j] != " ":
                    j += 1
            else:
                print(f" TYPED: '{phrase[j]}'")

            j += 1

            total_keystrokes += 1
            phrase_keystrokes += 1
        ks = (len(phrase) - phrase_keystrokes) / len(phrase) * 100.0
        print(f"KS: {ks:.2f} {phrase_keystrokes} {len(phrase)}")

    print()
    final_ks = (total_chars - total_keystrokes) / total_chars * 100.0
    print(f"TIME: {timer() - start:.2f}")
    print(f"FINAL KS: {final_ks:.4f}")
