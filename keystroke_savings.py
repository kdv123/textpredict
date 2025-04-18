#!/usr/bin/env python
# Computes keystroke savings of typing a set of phrases assume optimal use of 1 or more word predictions.
#
from ngram import NGramLanguageModel
from timeit import default_timer as timer
import argparse
from datetime import datetime
from language_model import alphabet
from socket import gethostname
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases", type=str, required=True, help="Input text file with phrases")
    parser.add_argument("--phrase-limit", type=int, help="Max phrases to evaluate")
    parser.add_argument("--lm", type=str, required=True, help="Filename of n-gram model to load")
    parser.add_argument("--lower", action="store_true", help="lowercase phrases")
    parser.add_argument("--strip", action="store_true", help="strip symbols except apostrophe")
    parser.add_argument("--nbest", type=int, help="N-best list size", default=3)
    parser.add_argument("--beam", type=float, help="log-prob beam for search", default=5.0)
    args = parser.parse_args()

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
    # TODO: tool currently assumes lowercase plus apostrophe and space
    symbol_set = list("abcdefghijklmnopqrstuvwxyz' ")
    print(f"Symbol set: {symbol_set}")

    lm = NGramLanguageModel(symbol_set, args.lm, False)
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

        print(f"*** Phrase {i}: {phrase}")
        # Iterate over all character positions in the phrase
        j = 0
        phrase_keystrokes = 0
        while j < len(phrase):
            left_context = phrase[0:j]
            # Figure out the target word
            # Back up until we hit space or start of string
            k = j
            while k > 0 and phrase[k] != " ":
                k -= 1
            target_word = ""
            # Go forward until we hit a space or end of string
            if phrase[k] == " ":
                k += 1
            while k < len(phrase) and phrase[k] != " ":
                target_word += phrase[k]
                k += 1

            print(f" left: '{left_context}', target: '{target_word}', keystrokes: {phrase_keystrokes}, len: {len(phrase)}")
            words = lm.predict_words(left_context, nbest=args.nbest)
            print(f" words: {words}")

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
