#!/usr/bin/env python
# Computes keystroke savings of typing a set of phrases assume optimal use of 1 or more word predictions.
# Supports the following types of language models:
#  1) n-gram character, via KenLM library
#  2) ByGPT5 byte tokenized LLM, via Hugging Face plus uniformers library

from timeit import default_timer as timer
from argparse import ArgumentParser
from datetime import datetime
from socket import gethostname
from sys import exit, stderr, stdout
from fcntl import flock, LOCK_UN, LOCK_EX
from os import path
import eval_helper

if __name__ == "__main__":

    parser = ArgumentParser()
    eval_helper.add_args(parser)
    parser.add_argument("--nbest", type=int, help="Number of word predictions made by simulated interface", default=3)
    parser.add_argument("--beam", type=float, help="For pruning word prediction search, log prob difference versus best completed hypothesis")
    parser.add_argument("--beam-max", type=int, help="For pruning word prediction search, max number of hypotheses to track per extension of search")
    parser.add_argument("--word-end", type=str, help="Additional symbols that can end a word", action="append", dest="word_end_symbols")
    parser.add_argument("--trailing-space", action="store_true", help="Assume user has to write a trailing space (VelociTap compatability)")
    parser.add_argument("--literal-slot", action="store_true", help="Use one slot for literal letters typed (except at start of word)")
    args = parser.parse_args()

    # Check for a variety of invalid command line switch combinations
    if sum([args.ngram, args.causal, args.byte]) != 1:
        print(f"ERROR: Exactly one of --ngram, --causal, --byte must be specified!", file = stderr)
        exit(1)
    if (args.causal or args.byte) and not args.model_name:
        print(f"ERROR: Transformer model must be specified with --model-name!", file = stderr)
        exit(1)
    eval_helper.check_args_for_errors(args)
    eval_helper.check_args_for_warnings(args)

    eval_helper.print_startup_info(args)
    eval_helper.set_cpu_cores(args)
    device = eval_helper.get_device(args)
    phrases, unstripped_context = eval_helper.load_phrases(args)

    eval_helper.prep_left_context(args)
    print(f"Prediction left context: '{args.left_context}'")
    stdout.flush()

    symbol_set = list(args.symbols)
    print(f"Symbols, size {len(symbol_set)}: {symbol_set}")
    print(f"Word end symbols: {args.word_end_symbols}")
    eval_helper.sanity_check_symbols(symbol_set = symbol_set,
                                     phrases = phrases,
                                     predict_lower= args.predict_lower)

    start = timer()
    lm = eval_helper.load_language_model(args=args,
                                         symbol_set=symbol_set,
                                         device=device)

    total_chars = 0
    total_keystrokes = 0
    total_truncated = 0

    previous_context = ""
    last_phrase_final_context = ""

    # Iterate over all the phrases
    total_predictions = 0
    prediction_start = timer()
    for phrase_index, phrase in enumerate(phrases):
        phrase_start = timer()

        phrase_len = len(phrase)
        if args.trailing_space:
            phrase_len += 1

        total_chars += phrase_len

        # In the case of simple automatic casing, we want to start with lowercase text
        # and then try and recover case in the left context as it is generated
        if args.case_simple:
            phrase = phrase.lower()

        print(f"*** Phrase {phrase_index + 1}: {phrase}")
        token_index = 0
        phrase_keystrokes = 0
        phrase_predictions = 0

        # We may want to maintain left context that makes use of previous phrases
        if args.previous_max_len:
            previous_context = eval_helper.update_context(context=previous_context + last_phrase_final_context,
                                                          max_len=args.previous_max_len,
                                                          previous_add=args.previous_add)
        else:
            # Reset the context on every phrase
            previous_context = ""

        word_prefix = ""
        context_to_use = ""

        # Iterate over all character positions in the phrase
        while token_index < len(phrase):
            # Figure out the target word
            # If the next letter is space, then our target is the current word
            if token_index > 0 and phrase[token_index] == " ":
                k = token_index - 1
            else:
                k = token_index
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

            # We may have mixed case phrases and want to match a lowercase predicted word
            if args.predict_lower:
                target_word = target_word.lower()

            # See if we should use a literal slot for this prediction (not used at start of words)
            use_literal = args.literal_slot and token_index > 0 and phrase[token_index] != " "

            # Use either the actual prefix of this sentence, or the context before symbols stripping
            if args.unstripped_context:
                context = unstripped_context[phrase_index][token_index]
            else:
                context = phrase[0:token_index]

            extra = ""
            context_to_use = context
            # Optionally automatically try and fix case in lowercase phrases
            if args.case_simple:
                context_to_use = eval_helper.case_simple(context)
                extra = f", case simple '{context_to_use}'"
            if args.previous_max_len:
                extra += f", previous_context '{previous_context}'"
            print(f"prefix '{word_prefix}', context '{context}'{extra}")

            words = lm.predict_words(left_context=previous_context + context_to_use,
                                     nbest=args.nbest,
                                     beam_logp_best=args.beam,
                                     beam_search_max=args.beam_max,
                                     word_end_symbols=args.word_end_symbols)

            # predict_words only returns the text that completes the current left_context
            # We need to add back in the prefix of the word thus far
            if args.predict_lower:
                words = [word_prefix.lower() + word for word in words]
            else:
                words = [word_prefix + word for word in words]

            # Add the literal text type as the final slot
            # But not if it already appears in the n-best results
            # If the n-best result is at maximum size, remove the least probable to make space for literal slot
            if use_literal:
                space_pos = context.rfind(" ")
                if space_pos != -1:
                    literal = context[space_pos + 1:]
                else:
                    literal = context
                if literal not in words:
                    if len(words) == args.nbest:
                        words[-1] = literal

            total_predictions += 1
            phrase_predictions += 1
            print_words = ""
            for k, word in enumerate(words):
                if word == target_word:
                    print_words += f"{k}:{word.upper()}, "
                else:
                    print_words += f"{k}:{word}, "
            print(f" predictions {print_words}, target '{target_word}', keys {phrase_keystrokes}")

            # See if we can get our target word via a prediction slot
            if target_word in words:
                print(f" SELECTED: {target_word}")
                # Advance to space or end of phrase
                while token_index < len(phrase) and phrase[token_index] != " ":
                    context += phrase[token_index]
                    token_index += 1
                word_prefix = ""
            else:
                if phrase[token_index] == " ":
                    word_prefix = ""
                else:
                    word_prefix += phrase[token_index]

                print(f" TYPED: '{phrase[token_index]}'")

            stdout.flush()
            token_index += 1

            total_keystrokes += 1
            phrase_keystrokes += 1

        # Update the context based on either the entire phrase or its unstripped content
        if args.unstripped_context:
            last_phrase_final_context = unstripped_context[phrase_index][-1]
        else:
            last_phrase_final_context = phrase

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
        if not path.exists(args.out_stats):
            # New file, write a header line
            file = open(args.out_stats, "w")
            # We may run this script in parallel so try and prevent writing to the stats file at the same time
            flock(file, LOCK_EX)
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
            flock(file, LOCK_EX)

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
        flock(file, LOCK_UN)
        file.close()
