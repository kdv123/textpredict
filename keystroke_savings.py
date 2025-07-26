#!/usr/bin/env python
# Computes keystroke savings of typing a set of phrases assume optimal use of 1 or more word predictions.
# Supports the following types of language models:
#  1) n-gram character, via KenLM library
#  2) ByGPT5 byte tokenized LLM, via Hugging Face plus uniformers library
from eval_helper import load_language_model
from ngram import NGramLanguageModel
from causal_byte import CausalByteLanguageModel
from timeit import default_timer as timer
import argparse
from datetime import datetime
from socket import gethostname
from sys import exit
from sys import stdout
import os
import eval_helper
import fcntl

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    eval_helper.add_args(parser)
    parser.add_argument("--nbest", type=int, help="Number of word predictions made by simulated interface", default=3)
    parser.add_argument("--beam", type=float, help="For pruning search, log prob difference versus best completed hypothesis")
    parser.add_argument("--beam-max", type=int, help="For pruning search, max number of hypotheses to track per extension of search")
    parser.add_argument("--symbols", type=str, default="abcdefghijklmnopqrstuvwxyz' ", help="Valid symbols in predicted words")
    parser.add_argument("--word-end", type=str, help="Additional symbols that can end a word", action="append", dest="word_end_symbols")
    parser.add_argument("--case-simple", action="store_true", default=False, help="Simple automatic casing of left context")
    parser.add_argument("--trailing-space", action="store_true", help="Assume user has to write a trailing space (VelociTap compatability)")
    parser.add_argument("--literal-slot", action="store_true", help="Use one slot for literal letters typed (except at start of word)")
    args = parser.parse_args()

    # Check for a variety of invalid command line switch combinations
    if sum([args.ngram, args.causal, args.byte]) != 1:
        print(f"ERROR: Exactly one of --ngram, --causal, --byte must be specified!")
        exit(1)
    if (args.causal or args.byte) and not args.model_name:
        print(f"ERROR: Transformer model must be specified with --model-name!")
        exit(1)
    eval_helper.check_args_for_errors(args)

    eval_helper.print_startup_info(args)
    eval_helper.set_cpu_cores(args)
    device = eval_helper.get_device(args)
    phrases = eval_helper.load_phrases(args)

    eval_helper.prep_left_context(args)
    print(f"Prediction left context: '{args.left_context}'")
    stdout.flush()

    start = timer()
    symbol_set = list(args.symbols)
    print(f"Symbols, size {len(symbol_set)}: {symbol_set}")
    print(f"Word end symbols: {args.word_end_symbols}")

    lm = eval_helper.load_language_model(args=args,
                                         symbol_set=symbol_set,
                                         device=device,
                                         normal_space=True)
    total_chars = 0
    total_keystrokes = 0
    total_truncated = 0

    # Iterate over all the phrases
    total_predictions = 0
    prediction_start = timer()
    for i, phrase in enumerate(phrases):
        phrase_start = timer()

        phrase_len = len(phrase)
        if args.trailing_space:
            phrase_len += 1

        total_chars += phrase_len

        print(f"*** Phrase {i}: {phrase}")
        j = 0
        phrase_keystrokes = 0
        phrase_predictions = 0
        # Iterate over all character positions in the phrase
        while j < len(phrase):
            current_left_context = phrase[0:j]

            # Figure out the target word
            # If the next letter is space, then our target is the current word
            if j > 0 and phrase[j] == " ":
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

            # Adjust to one less prediction if using literal slot and not at the start of a word
            use_literal = args.literal_slot and len(current_left_context) > 0 and current_left_context[-1] != " "
            nbest = args.nbest
            if use_literal:
                nbest -= 1
            words = lm.predict_words(left_context=current_left_context,
                                     nbest=nbest,
                                     beam_logp_best=args.beam,
                                     beam_search_max=args.beam_max,
                                     word_end_symbols=args.word_end_symbols)
            # Add the literal text type as the final slot
            if use_literal:
                words.append(current_left_context)

            total_predictions += 1
            phrase_predictions += 1
            print_words = ""
            for word in words:
                if word == target_word:
                    print_words += f" {word.upper()}"
                else:
                    print_words += f" {word}"
            print(f" predictions:{print_words}, left '{current_left_context}', target '{target_word}', keys {phrase_keystrokes}")

            # See if we can get our target word via a prediction slot
            if target_word in words:
                print(f" SELECTED: {target_word}")
                # Advance to space or end of phrase
                while j < len(phrase) and phrase[j] != " ":
                    j += 1
            else:
                print(f" TYPED: '{phrase[j]}'")
            stdout.flush()
            j += 1

            total_keystrokes += 1
            phrase_keystrokes += 1
        ks = (phrase_len - phrase_keystrokes) / phrase_len * 100.0
        print(f"KS: {ks:.2f} keys {phrase_keystrokes} len {phrase_len} secs/pred {(timer() - phrase_start) / phrase_predictions:.2f}")
        stdout.flush()

    print()
    final_ks = (total_chars - total_keystrokes) / total_chars * 100.0
    total_time = timer() - start
    secs_per_pred = (timer() - prediction_start) / total_predictions

    print(f"TRUNCATED: {total_truncated}")
    print(f"CHARS, KEYSTROKES, PHRASES: {total_chars} {total_keystrokes} {len(phrases)}")
    print(f"TIME: {total_time:.2f}")
    print(f"SECS/PRED: {secs_per_pred:.4f}")
    print(f"FINAL KS: {final_ks:.4f}")

    # Optional output of a tab-delimited file for easy tracking of results over multiple experiments
    if args.out_stats:
        if not os.path.exists(args.out_stats):
            # New file, write a header line
            file = open(args.out_stats, "w")
            # We may run this script in parallel so try and prevent writing to the stats file at the same time
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(f"final_ks"
                       f"\tphrases"
                       f"\ttotal_words"
                       f"\ttotal_chars"
                       f"\ttotal_keystrokes"
                       f"\ttotal_time"
                       f"\tsecs_per_pred"                       
                       f"\tdate_time"
                       f"\thostname"
                       )
            # Write any of the optional column names the client intends to log
            if args.out_extra_cols:
                for extra in args.out_extra_cols:
                    extra_col_name = extra.split(",")[0]
                    file.write(f"\t{extra_col_name}")
            file.write("\n")
        else:
            file = open(args.out_stats, "a")
            fcntl.flock(file, fcntl.LOCK_EX)

        file.write(f"{final_ks:.6f}"
                   f"\t{len(phrases)}"
                   f"\t{eval_helper.count_words(phrases)}"
                   f"\t{total_chars}"
                   f"\t{total_keystrokes}"
                   f"\t{timer() - start:.2f}"
                   f"\t{secs_per_pred:.6f}"                   
                   f"\t{datetime.now()}"
                   f"\t{gethostname()}"
                   )
        # Write any of the optional column values
        if args.out_extra_cols:
            for extra in args.out_extra_cols:
                extra_col_val = extra.split(",")[1]
                file.write(f"\t{extra_col_val}")
        file.write("\n")
        fcntl.flock(file, fcntl.LOCK_UN)
        file.close()
