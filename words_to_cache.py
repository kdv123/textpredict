# Utility for computing a cache that maps the prefix of any word in a word list
# to the subword sequences that would need to be queried to compute the
# character distribution over the next character.

import argparse
import os.path
import sys
import time
from transformers import AutoTokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--words", dest="words", type=str, required=True,
                        help="Word list filename")
    parser.add_argument("--model-name", dest="model_name", required=True,
                        help="Language model name")
    parser.add_argument("--lower", dest="lower", action="store_true",
                        help="Lowercase words and subword tokens")
    parser.add_argument("--symbols", dest="symbols", default="abcdefghijklmnopqrstuvwxyz' ",
                        help="valid symbols")

    args = parser.parse_args()
    start_time_ns = time.time_ns()

    # First get the words we are going to compute over
    if not os.path.isfile(args.words):
        print(f"ERROR: can't find filename: '{args.words}'")
        sys.exit(1)

    # Create a set of the valid characters we plan to use during predictions
    # that make use of the cache we are generating.
    valid_symbols = set(args.symbols)
    print(f"{len(valid_symbols)} valid symbols: {args.symbols}")

    seen_words = set()
    words = []
    dropped_words = 0
    with open(args.words) as file:
        for word in file:
            word = word.strip()
            if args.lower:
                word = word.lower()
            # Don't include words more than once
            if word not in seen_words:
                # Drop words containing a character not in our symbol set
                if 0 not in [ch in valid_symbols for ch in word]:
                    words.append(word)
                else:
                    dropped_words += 1
    print(f"Dropped {dropped_words} words")

    if len(words) == 0:
        print(f"ERROR: no words found in filename: '{args.words}'")
        sys.exit(1)

    print(f"Computing cache for {len(words)} words...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    vocab_size = tokenizer.vocab_size
    print(f"Language model vocab size: {vocab_size}")

    # Loop over all the subword tokens in the LLM creating list from
    # integer index to the subword token's text.
    index_to_word = []
    for i in range(vocab_size):
        word = tokenizer.decode([i])
        if args.lower:
            word = word.lower()
        index_to_word += word,

    # Create a dictionary where the keys are a prefix of a word in the vocabulary
    # and the value is a list of all subword token sequences that are one subword
    # token short of generating the prefix. This includes the entire word itself
    # since we need to predict the following space.
    # For example:
    # WORD: catching
    # PREFIX: catchi

    prefix_to_sequence = {}
    prefix_count = 0
    for word in words:
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            prefix_count += 1
    print(f"Prefix count: {prefix_count}")

    end_time_ns = time.time_ns()
    print(f"Total time: {(end_time_ns - start_time_ns) / 1e9 :.3f}")