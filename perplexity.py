#!/usr/bin/env python
# Calculates per-character perplexity on a set of sentences using a language model.
# Supports the following types of models:
#   1) N-gram language model via the KenLM library, ARPA or KenLM format file
#   2) Causal LLM using subword tokenization, Hugging Face model support by the automodel
#   3) Causal LLM using byte tokenization, Hugging Face ByGPT5 model: https://github.com/potamides/uniformers
#   4) Mixture model of n-gram and subword LLM, using linear interpolation
#   5) Mixture model of n-gram and byte LLM, using linear interpolation
#   6) Causal LLM with a classification layer, requires downstream task training using AutoModelForSequenceClassification
#
# This is a cleaned up version of the lm_eval.py script

from mixture import MixtureLanguageModel
from classifier import ClassifierLanguageModel
from math import log10
from timeit import default_timer as timer
from argparse import ArgumentParser
import json
import numpy as np
from sys import exit, stderr, stdout
from scipy.stats import bootstrap
from datetime import datetime
from os import path
from fcntl import flock, LOCK_UN, LOCK_EX
import eval_helper

if __name__ == "__main__":

    parser = ArgumentParser()
    eval_helper.add_args(parser)
    parser.add_argument("--verbose", type=int, default=0, help="0: Only output model averages\n1: Output results from each phrase\n2: Output results from each character")
    parser.add_argument("--classifier", action="store_true", help="Use classifier model")
    parser.add_argument("--mix", action="store_true", help="Use mixture of n-gram and subword LLM")
    parser.add_argument("--mix-byte", action="store_true", help="Use mixture of n-gram and byte LLM")
    parser.add_argument("--ngram-mix", type=float, default=0.5, help="Weight for n-gram in mixture model")
    parser.add_argument("--time-outliers", action="store_true", help="Print time outliers at end")
    parser.add_argument("--beam-width", type=int, help="Search beam width for causal LM, recommended value = 8")
    parser.add_argument("--max-completed", type=int, help="Stop causal LM search after this many completed hypotheses, recommended value = 32000")
    parser.add_argument("--ppl-file", help="Output sentence and ppl to a file")
    parser.add_argument("--symbol-file", help="Output symbol log probs to a file")
    parser.add_argument("--json-file", help="Output overall model data to JSON file with specified file name.")
    parser.add_argument("--srilm-file", help="Output SRILM format debug 2 log file")
    parser.add_argument("--bootstrap-samples", type=int, default=9999, help="Number of samples to use for bootstrap estimates")
    parser.add_argument("--bootstrap-method", default="BCa", help="Method to use for bootstrap, BCa | basic | percentile")
    parser.add_argument("--skip-oov-symbols", action="store_true", help="Skip symbols that aren't in our symbol set")
    args = parser.parse_args()

    # Check for various invalid combinations of command line options
    if sum([args.ngram, args.causal, args.byte, args.classifier, args.mix, args.mix_byte]) != 1:
        print(f"ERROR: Exactly one of --ngram --causal, --mix, --mix-byte, --byte, --classifier must be specified!", file = stderr)
        exit(1)
    if (args.causal or args.byte or args.classifier or args.mix or args.mix_byte) and not args.model_name:
        print(f"ERROR: Transformer model must be specified with --model-name!", file = stderr)
        exit(1)
    if (args.mix or args.mix_byte) and not args.ngram_lm:
        print(f"ERROR: Mixture model requires n-gram model to be specified with --ngram-lm!", file = stderr)
        exit(1)
    eval_helper.check_args_for_errors(args)
    eval_helper.check_args_for_warnings(args)

    eval_helper.print_startup_info(args)
    eval_helper.set_cpu_cores(args)
    phrases, unstripped_context = eval_helper.load_phrases(args)
    device = eval_helper.get_device(args)

    eval_helper.prep_left_context(args)
    print(f"Prediction left context: '{args.left_context}'")
    stdout.flush()

    symbol_set = list(args.symbols)
    print(f"Symbols, size {len(symbol_set)}: {symbol_set}")
    eval_helper.sanity_check_symbols(symbol_set = symbol_set,
                                     phrases = phrases,
                                     predict_lower = args.predict_lower)

    rng = np.random.default_rng(234893458942534)

    lm = None

    ppl_file = None
    if args.ppl_file:
        ppl_file = open(args.ppl_file, "w")

    # Optional output of a log file in the same format as SRILM at debug level 2.
    # This allows us to compute a mixture weight based on the multiple log files using compute-best-mix script.
    srilm_file = None
    if args.srilm_file:
        srilm_file = open(args.srilm_file, "w")

    start = timer()

    # Some language models are only supported by the perplexity evaluation script
    if args.mix:
        print(f"Loading n-gram + subword mixture: {args.model_name}, model directory {args.model_dir}")
        lm = MixtureLanguageModel(symbol_set=symbol_set,
                                  lm_types=["CAUSAL", "NGRAM"],
                                  lm_weights=[1.0 - args.ngram_mix, args.ngram_mix],
                                  lm_params=[{"lang_model_name": args.model_name,
                                             "lm_device": device,
                                             "lm_path": args.model_dir,
                                             "lm_left_context": args.left_context,
                                             "beam_width": args.beam_width,
                                             "fp16": args.fp16,
                                             "max_completed": args.max_completed,
                                            },
                                            {"lm_path": args.ngram_lm}])
        print(f"Model load time = {timer() - start:.2f}")
    elif args.mix_byte:
        print(f"Loading n-gram + byte mixture: {args.model_name}, model directory {args.model_dir}")
        lm = MixtureLanguageModel(symbol_set=symbol_set,
                                  lm_types=["CAUSALBYTE", "NGRAM"],
                                  lm_weights=[1.0 - args.ngram_mix, args.ngram_mix],
                                  lm_params=[{"lang_model_name": args.model_name,
                                             "lm_device": device,
                                             "lm_path": args.model_dir,
                                             "lm_left_context": args.left_context,
                                             "fp16": args.fp16,
                                            },
                                            {"lm_path": args.ngram_lm}])
        print(f"Model load time = {timer() - start:.2f}")
    elif args.classifier:
        print(f"Loading classifier model: {args.model_name}, model directory {args.model_dir}")
        lm = ClassifierLanguageModel(symbol_set=symbol_set,
                                     lang_model_name=args.model_name,
                                     lm_path=args.model_dir,
                                     lm_device=device,
                                     lm_left_context=args.left_context,
                                     beam_width=args.beam_width,
                                     fp16=args.fp16,
                                     mixed_case_context=args.mixed_case_context)
        print(f"Model load time = {timer() - start:.2f}")
    else:
        lm = eval_helper.load_language_model(args=args,
                                             symbol_set=symbol_set,
                                             device=device)

    phrase_count = 0
    sum_per_symbol_logprob = 0.0
    zero_prob = 0
    overall_predict_time_arr = np.array([])
    overall_predict_details_arr = np.array([])
    skipped_symbols = 0

    start = timer()

    sum_log_prob = 0.0
    sum_symbols = 0
    all_symbol_log_probs = []
    all_sentence_ppls = []
    context = ""

    previous_context = ""
    last_phrase_final_context = ""

    # Iterate over phrases
    for phrase_index, phrase in enumerate(phrases):
        symbols = 0
        accum = 0.0
        sent_ppl = 0.0

        # In the case of simple automatic casing, we want to start with lowercase text
        # and then try and recover case in the left context as it is generated
        if args.case_simple:
            phrase = phrase.lower()

        # Phrase-level output
        if args.verbose >= 1:
            print(f"*** Phrase {phrase_index + 1}: {phrase}")

        # Split into a list of characters and convert spaces to pseudo-word
        tokens = [char for char in phrase]

        # Optionally skip symbols not in our set
        if args.skip_oov_symbols:
            tokens_stripped = [token for token in tokens if token in symbol_set]
            skipped_symbols += len(tokens) - len(tokens_stripped)
            tokens = tokens_stripped

        symbols = len(tokens)

        # SRILM starts with the sentence being evaluated
        if srilm_file:
            for symbol_index, symbol in enumerate(tokens):
                if symbol_index > 0:
                    srilm_file.write(" ")
                srilm_file.write(symbol)
            srilm_file.write("\n")

        # Initial previous token is the start symbol, initial context empty
        prev_token = "<s>"
        prev_token_display = prev_token

        predict_time_arr = np.array([])
        predict_details_arr = np.array([])

        # We may want to maintain left context that makes use of previous phrases
        if args.previous_max_len:
            previous_context = eval_helper.update_context(context=previous_context + last_phrase_final_context,
                                                          max_len=args.previous_max_len,
                                                          previous_add=args.previous_add)
        else:
            # Reset the context on every phrase
            previous_context = ""

        context_to_use = ""

        # Iterate over characters in phrase
        for token_index, token in enumerate(tokens):
            # Even if the LM supports mixed case we may want to predict only lowercase
            if args.predict_lower:
                token_to_predict = token.lower()
            else:
                token_to_predict = token

            # Use the space pseudo-word for display of the space character
            token_display = token_to_predict.replace(" ", args.space_symbol)

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
            if args.verbose >= 3:
                print(f"context '{context}'{extra}")

            # We want to precisely measure the inference time, don't insert any code in this block
            context_list = list(previous_context + context_to_use)
            start_predict = timer()
            next_char_pred = lm.predict(context_list)
            predict_time = timer() - start_predict

            predict_time_arr = np.append(predict_time_arr, predict_time)
            predict_details_arr = np.append(predict_details_arr,
                                            f"sentence = {phrase}, index = {token_index}, p( {token_display} | {prev_token_display} )")

            # Find the probability for the correct character
            prob_correct_char = next_char_pred[[c[0] for c in next_char_pred].index(token_to_predict)][1]
            if prob_correct_char == 0:
                log_prob_correct_char = 0.0
                zero_prob += 1
                accum = 1
                if args.verbose >= 2:
                    print(f"p( {token_display} | {prev_token_display} ...) = 0")
                    print(f"prediction time = {predict_time:.6f}")
                break
            else:
                log_prob_correct_char = log10(prob_correct_char)
                # Character-level output
                if args.verbose >= 2:
                    print(f"p( {token_display} | {prev_token_display} ...) = {prob_correct_char:.6f} [ {log_prob_correct_char:.6f} ]")
                    print(f"prediction time = {predict_time:.6f}")

            # SRILM line for a character looks like: "	p( w | <s> ) 	= [2gram] 0.095760 [ -1.018816 ]"
            if srilm_file:
                extra = ""
                if token_index > 0:
                    extra = " ..."
                # The 1gram bit is only relevant for the n-gram, we'll just hard code to 1gram for everything
                srilm_file.write(f"\tp( {token_display} | {prev_token_display}{extra}) \t= [1gram] {prob_correct_char:.6f} [ {log_prob_correct_char:.6f} ]\n")

            accum += log_prob_correct_char
            prev_token_display = token_display

            #context += token
            all_symbol_log_probs.append(log_prob_correct_char)
            stdout.flush()

        # Update the context based on either the entire phrase or its unstripped content
        if args.unstripped_context:
            last_phrase_final_context = unstripped_context[phrase_index][-1]
        else:
            last_phrase_final_context = phrase

        # Compute summary stats on prediction times for this phrase
        per_symbol_time = np.average(predict_time_arr)
        phrase_std = np.std(predict_time_arr)
        phrase_max = np.max(predict_time_arr)
        phrase_min = np.min(predict_time_arr)

        # Add this phrase's prediction times to overall array
        overall_predict_time_arr = np.append(overall_predict_time_arr, predict_time_arr, axis=None)
        overall_predict_details_arr = np.append(overall_predict_details_arr, predict_details_arr, axis=None)

        if accum == 1:
            if args.verbose >= 1:
                print("Zero-prob event encountered, terminating phrase")
                print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, "
                      f"{phrase_max:.6f}]\n")
        else:
            per_symbol_logprob = accum / symbols
            sent_ppl = pow(10, -1 * per_symbol_logprob)

            all_sentence_ppls.append(sent_ppl)

            # Phrase-level output
            if args.verbose >= 1:
                print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {sent_ppl:.4f}")
                print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, {phrase_max:.6f}]\n")

            sum_per_symbol_logprob += per_symbol_logprob
            phrase_count += 1

            # Optional output to a file with a sentence and its ppl and log prob
            if ppl_file:
                ppl_file.write(f"{sent_ppl:.4f}\t{accum:.4f}\t{phrase}\n")
                ppl_file.flush()

            # To calculate the overall file perplexity, we need the sum of log probs of all sentences.
            # This is how SRILM does it and makes it less sensitive to particular outlier sentences.
            sum_log_prob += accum
            sum_symbols += symbols

        # SRILM state for the sentence
        if srilm_file:
            srilm_file.write(f"0 sentences, {symbols} words, 0 OOVs\n")
            srilm_file.write(f"0 zeroprobs, logprob= {accum:.4f} ppl= {sent_ppl:.3f} ppl1= {sent_ppl:.3f}\n")
            srilm_file.write("\n")
            srilm_file.flush()

        stdout.flush()
    inference_time = timer() - start

    if ppl_file:
        ppl_file.close()

    overall_per_symbol_time = np.average(overall_predict_time_arr)
    overall_std_time = np.std(overall_predict_time_arr)
    overall_min_time = np.min(overall_predict_time_arr)
    overall_max_time = np.max(overall_predict_time_arr)

    ci_floor = overall_per_symbol_time - (2 * overall_std_time)
    ci_ceiling = overall_per_symbol_time + (2 * overall_std_time)

    ppl = float("+inf")
    if sum_symbols > 0:
        ppl = pow(10, -1 * sum_log_prob / sum_symbols)

    # SRILM final overall stats lines
    if srilm_file:
        srilm_file.write(f"file {args.phrases}: 0 sentences, {sum_symbols} words, 0 OOVs\n")
        srilm_file.write(f"0 zeroprobs, logprob= {sum_log_prob:.4f} ppl= {ppl:.3f} ppl1= {ppl:.3f}\n")
        srilm_file.close()

    avg_sentence_ppl = np.average(all_sentence_ppls)

    # Model-level output
    print(f"OVERALL \
        \nphrases = {phrase_count} \
        \nzero-prob events = {zero_prob} \
        \nper-symbol prediction time = {overall_per_symbol_time:.6f} +/- {overall_std_time:.6f} [{overall_min_time:.6f}, {overall_max_time:.6f}] \
        \n95% CI = [{ci_floor:.6f}, {ci_ceiling:.6f}] \
        \ninference time = {inference_time:.2f}\
        \nsum logprob = {sum_log_prob:.2f} \
        \nsum symbols = {sum_symbols} \
        \nskipped symbols = {skipped_symbols} \
        \nmean symbol log prob = {np.average(all_symbol_log_probs):.4f} \
        \nmean sentence ppl = {avg_sentence_ppl:.4f} \
        \nppl = {ppl:.4f}")
    stdout.flush()

    if args.json_file:
        output_dict = {}
        output_dict["phrases"] = phrase_count
        output_dict["zero_prob_events"] = zero_prob
        output_dict["per_symbol_predict_time"] = f"{overall_per_symbol_time:.6f} +/- {overall_std_time:.6f} [{overall_min_time:.6f}, {overall_max_time:.6f}]"
        output_dict["confidence_interval"] = f"[{ci_floor:.6f}, {ci_ceiling:.6f}]"
        output_dict["inference_time"] = round(inference_time, 2)
        output_dict["sum_log_prob"] = round(sum_log_prob, 2)
        output_dict["sum_symbols"] = sum_symbols
        output_dict["skipped_symbols"] = skipped_symbols
        output_dict["mean_symbol_log_prob"] = round(np.average(all_symbol_log_probs), 4)
        output_dict["mean_sentence_ppl"] = round(avg_sentence_ppl, 4)
        output_dict["ppl"] = round(ppl, 4)

        with open(args.json_file, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
            f.write("\n")

    # Optional fill that contains the log prob of each prediction
    # Could be useful for recomputing confidence intervals or such
    if args.symbol_file:
        symbol_file = open(args.symbol_file, "w")
        for log_prob in all_symbol_log_probs:
            symbol_file.write(str(log_prob) + "\n")
        symbol_file.close()

    if args.out_stats:
        # Single line file output, useful for running experiments
        print(f"Outputting stats to {args.out_stats}, running bootstrap on {len(all_symbol_log_probs)} samples.")
        stdout.flush()
        time_bootstrap = timer()
        bootstrap_log_prob = bootstrap(data=(all_symbol_log_probs,),
                                        statistic=np.mean,
                                        confidence_level=0.95,
                                        n_resamples=args.bootstrap_samples,
                                        method=args.bootstrap_method,
                                        random_state=rng)
        print(f"Bootstrap on log probs completed in {(timer() - time_bootstrap):.2f} seconds.")
        stdout.flush()

        ppl_high = pow(10, -1 * bootstrap_log_prob.confidence_interval.low)
        ppl_low = pow(10, -1 * bootstrap_log_prob.confidence_interval.high)
        error_bar = (ppl_high - ppl_low) / 2.0

        print(f"Outputting stats to {args.out_stats}, running bootstrap on {len(all_sentence_ppls)} samples.")
        stdout.flush()
        time_bootstrap = timer()
        bootstrap_sentence_ppl = bootstrap(data=(all_sentence_ppls,),
                                            statistic=np.mean,
                                            confidence_level=0.95,
                                            n_resamples=args.bootstrap_samples,
                                            method=args.bootstrap_method,
                                            random_state=rng)
        print(f"Bootstrap on sentence ppls completed in {(timer() - time_bootstrap):.2f} seconds.")
        sentence_ppl_high = bootstrap_sentence_ppl.confidence_interval.high
        sentence_ppl_low = bootstrap_sentence_ppl.confidence_interval.low
        sentence_ppl_error_bar = (sentence_ppl_high - sentence_ppl_low) / 2.0

        params = -1
        if args.causal or args.byte or args.classifier:
            params = lm.get_num_parameters()

        exists = path.isfile(args.out_stats)
        with open(args.out_stats, 'a') as file:
            # We may run this script in parallel so try and prevent writing to the stats file at the same time
            flock(file, LOCK_EX)
            if not exists:
                # Header if the stats file doesn't already exist
                file.write(f"ppl"
                           f"\tsum_log_prob"
                           f"\tsum_symbols"
                           f"\tboot_ppl_pm"
                           f"\tboot_ppl_low"
                           f"\tboot_ppl_high"
                           f"\tphrases"
                           f"\ttime"
                           f"\tparams"
                           f"\tdate_time"
                           f"\tper_symbol_time"
                           f"\tsd_per_symbol_time"
                           f"\tsentence_ppl"
                           f"\tboot_sentence_ppl")
                # Write any of the optional column names the client intends to log
                if args.out_extra_cols:
                    for extra in args.out_extra_cols:
                        extra_col_name = extra.split(",")[0]
                        file.write(f"\t{extra_col_name}")
                file.write("\n")

            file.write(f"{ppl:.6f}"
                       f"\t{sum_log_prob:.6f}"
                       f"\t{sum_symbols}"
                       f"\t{error_bar:.6f}"
                       f"\t{ppl_low:.6f}"
                       f"\t{ppl_high:.6f}"
                       f"\t{phrase_count}"
                       f"\t{inference_time:.6f}"
                       f"\t{params}"
                       f"\t{datetime.now()}"
                       f"\t{overall_per_symbol_time:.6e}"
                       f"\t{overall_std_time:.6e}"
                       f"\t{avg_sentence_ppl:.6f}"
                       f"\t{sentence_ppl_error_bar:.6f}")
            # Write any of the optional column values
            if args.out_extra_cols:
                for extra in args.out_extra_cols:
                    extra_col_val = extra.split(",")[1]
                    file.write(f"\t{extra_col_val}")
            file.write("\n")
            flock(file, LOCK_UN)

    # Prediction timing stats for the causal LLM
    if args.causal or args.mix:
        lm.dump_predict_times()

    # Optionally print the predictions that took an abnormal amount of time
    if args.time_outliers:
        for (i, time) in enumerate(overall_predict_time_arr):
            if time < ci_floor:
                print(f"LOW OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")
            if time > ci_ceiling:
                print(f"HIGH OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")

