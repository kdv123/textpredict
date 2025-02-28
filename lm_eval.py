#!/usr/bin/env python
# Calculates per-character perplexity on a set of sentences using a language model.
# Supports the following types of models:
#   1) Uniform language model, same probability for every symbol in the vocabulary
#   2) N-gram language model via the KenLM library, ARPA or KenLM format file
#   3) Causal LLM using subword tokenization, Hugging Face model support by the automodel
#   4) Causal LLM using byte tokenization, Hugging Face ByGPT5 model: https://github.com/potamides/uniformers
#   5) Mixture model using linear interpolation, mixture of the above types
#   6) Causal LLM with a classification layer, requires downstream task training using AutoModelForSequenceClassification

from ngram import NGramLanguageModel
from mixture import MixtureLanguageModel
from causal import CausalLanguageModel
from seq2seq import Seq2SeqLanguageModel
from causal_byte import CausalByteLanguageModel
from uniform import UniformLanguageModel
from classifier import ClassifierLanguageModel
from math import log10
from timeit import default_timer as timer
import argparse
import string
import json
import numpy as np
from sys import exit
from scipy.stats import bootstrap
from datetime import datetime
from os import path
from language_model import DEFAULT_SYMBOL_SET
from socket import gethostname
from torch import set_num_threads
from psutil import cpu_count
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=0,
                        help="0: Only output model averages\n1: Output results from each phrase\n2: Output results from each character")
    parser.add_argument("--model", type=int, required=True,
                        help=("3: KenLM n-gram\n4: Causal Hugging Face\n5: Seq2Seq\n6: Mixture (Causal/Ngram\n7: Causal Byte\n8: Uniform\n9: Mix (Casual Byte/Ngram)\n10: Classifier"))
    parser.add_argument("--phrases", type=str, required=True,
                        help="Phrase set filename")
    parser.add_argument("--model-name",
                        help="Model name of causal model")
    parser.add_argument("--model-dir",
                        help="Local directory to load fine-tuned causal model")
    parser.add_argument("--use-mps",
                        action="store_true",
                        help="Use MPS Apple Silicon GPU during inference")
    parser.add_argument("--use-cuda",
                        action="store_true",
                        help="Use CUDA GPU during inference")
    parser.add_argument("--left-context", default="",
                        help="left language model context for causal model")
    parser.add_argument("--left-context-file", default="",
                        help="name of file containing the left language model context for causal model. Context using --left-context takes priority.")
    parser.add_argument("--add-char", action="append", dest="extra_chars",
                        help="add character to symbol set")
    parser.add_argument("--time-outliers", action="store_true",
                        help="print time outliers at end")
    parser.add_argument("--stats-file",
                        help="write summary stats to specified file")
    parser.add_argument("--stats-extra",
                        help="extra string to write to stats file as first column")
    parser.add_argument("--phrase-limit", type=int,
                        help="max phrases to evaluate")
    parser.add_argument("--beam-width", type=int,
                        help="search beam width for causal LM, recommended value = 8")
    parser.add_argument("--max-completed", type=int,
                        help="stop causal LM search after this many completed hypotheses, recommended value = 32000")
    parser.add_argument("--ppl-file",
                        help="output sentence and ppl to a file")
    parser.add_argument("--symbol-file",
                        help="output symbol log probs to a file")
    parser.add_argument("--json-file",
                        help="Output overall model data to JSON file with specified file name.")
    parser.add_argument("--fp16", action="store_true",
                        help="convert model to fp16 (CUDA only)")
    parser.add_argument("--mixed-case-context", action="store_true", default=False,
                        help="use mixed case left context")
    parser.add_argument("--case-simple", action="store_true", default=False,
                        help="simple automatic casing of let context")
    parser.add_argument("--ngram-lm",
                        help="ngram model to load")
    parser.add_argument("--ngram-mix", type=float, default=0.5,
                        help="mixture weight for ngram in mixture models")
    parser.add_argument("--srilm-file",
                        help="output SRILM debug 2 log file")
    parser.add_argument("--skip-norm", action="store_true", default=False,
                        help="skip normalization over symbols for n-gram model, for matching SRILM output when using LM with extra symbols")
    parser.add_argument("--num-cores", type=int,
                        help="limit pytorch to specified number of cores")
    parser.add_argument("--bootstrap-samples", type=int, default=9999,
                        help="number of samples to use for bootstrap estimates")
    parser.add_argument("--bootstrap-method", default="BCa",
                        help="method to use for bootstrap, BCa | basic | percentile")
    parser.add_argument("--lora", action="store_true", default=False, help="use LoRA adapter with base model")
    parser.add_argument("--lora-path", help="huggingface or local path to LoRA adapter")
    parser.add_argument("--drop-start-end-words", action="store_true",
                        help="drop <s> and </s> words from phrases")
    parser.add_argument("--skip-oov-symbols", action="store_true",
                        help="skip symbols that aren't in our symbol set")
    args = parser.parse_args()

    verbose = args.verbose
    model = args.model
    phrases = args.phrases

    if args.lora and not model == 4:
        print("ERROR: LoRA adapter is only currently supported for causal model")
        exit(1)

    if args.lora and not args.lora_path:
        print("ERROR: To use a LoRA adapter you must specify the adapter path using --lora-path. --model-name specifies the base model.")
        exit(1)
    
    if model == 3 and not args.ngram_lm:
        print("ERROR: For n-gram model you must specify filename of model using --ngram-lm")
        exit(1)

    if model == 4 and not args.model_name:
        print("ERROR: For causal model you must specify name of model using --model-name")
        exit(1)

    if (model == 6 or model == 9) and (not args.model_name or not args.ngram_lm):
        print(f"ERROR: For mixture model you must specify name of causal LLM using --model-name and ngram LM using --ngram-lm")
        exit(1)

    if model == 7 and not args.model_name:
        print("ERROR: For causal byte model you must specify name of model using --model-name")
        exit(1)

    if model == 10 and not args.model_name:
        print("ERROR: For classifier model you must specify name of model using --model-name")
        exit(1)
        
    if args.case_simple and not args.mixed_case_context:
        print(f"WARNING: You should probably also set --mixed-case-context with --case-simple")

    # Handy stuff to print out in our log files
    print(f"START: {datetime.now()}")
    print(f"ARGS: {args}")
    print(f"HOSTNAME: {gethostname()}")

    rng = np.random.default_rng(234893458942534)

    if args.num_cores:
        # User has specified their desired number of cores
        set_num_threads(args.num_cores)
        print(f"Limiting pytorch to {args.num_cores} cores")
    elif args.use_cuda:
        # Testing showed more CPU cores did not improve inference speed when a GPU is being used
        set_num_threads(1)
        print(f"Using CUDA, limiting pytorch to 1 core. You can override with --num-cores but might no speed things up")
    else:
        # Testing showed CPU only inference didn't get faster after 32 cores
        physical_cores = cpu_count(logical=False)
        max_useful_cores = 32
        if physical_cores > max_useful_cores:
            set_num_threads(max_useful_cores)
            print(f"Limiting pytorch to {max_useful_cores} cores. You can override with --num-cores but might no speed things up")
    
    if args.left_context_file != "" and args.left_context == "":
        try:
            with open(args.left_context_file, "r", encoding="utf-8") as f:
                contents = ""
                for line in f:
                    if line not in string.whitespace:
                        contents += f" {line.rstrip()}"
                contents = contents.strip()
                args.left_context = contents
        except FileNotFoundError:
            print("CANNOT OPEN LEFT CONTEXT FILE {args.left_context_file}")
    sys.stdout.flush()

    # Allow passing in of space characters in the context using <sp> word
    args.left_context = args.left_context.replace("<sp>", " ")
    print(f"Prediction left context: '{args.left_context}'")
    sys.stdout.flush()

    device = "cpu"
    if args.use_mps:
        device = "mps"
    elif args.use_cuda:
        device = "cuda"

    # Read in the phrase file
    phrase_file = open(phrases, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()
    # Optional drop start and end words from a test set file
    if args.drop_start_end_words:
        phrases = [phrase.replace("<s> ", "").replace(" </s>", "") for phrase in phrases]

    # We may want to limit to only the first so many phrases
    if args.phrase_limit:
        while len(phrases) > args.phrase_limit:
            phrases.pop()

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

    symbol_set = DEFAULT_SYMBOL_SET
    if args.extra_chars:
        for char in args.extra_chars:
            symbol_set += char
        print(f"Modified symbol_set: {symbol_set}")

    if model == 3:
        lm = NGramLanguageModel(symbol_set, args.ngram_lm, args.skip_norm)
    elif model == 4:
        lm = CausalLanguageModel(symbol_set=symbol_set,
                                 lang_model_name=args.model_name,
                                 lm_device=device,
                                 lm_path=args.model_dir,
                                 lm_left_context=args.left_context,
                                 beam_width=args.beam_width,
                                 fp16=args.fp16,
                                 mixed_case_context=args.mixed_case_context,
                                 case_simple=args.case_simple,
                                 max_completed=args.max_completed,
                                 lora=args.lora,
                                 lora_path=args.lora_path,)
    elif model == 5:
        lm = Seq2SeqLanguageModel(symbol_set=symbol_set,
                                  lang_model_name=args.model_name,
                                  lm_device=device,
                                  lm_path=args.model_dir,
                                  lm_left_context=args.left_context)
    elif model == 6:
        lm = MixtureLanguageModel(symbol_set=symbol_set,
                                  lm_types=["CAUSAL", "NGRAM"],
                                  lm_weights=[1.0 - args.ngram_mix, args.ngram_mix],
                                  lm_params=[{"lang_model_name": args.model_name,
                                             "lm_device": device,
                                             "lm_path": args.model_dir,
                                             "lm_left_context": args.left_context,
                                             "beam_width": args.beam_width,
                                             "fp16": args.fp16,
                                             "mixed_case_context": args.mixed_case_context,
                                             "case_simple": args.case_simple,
                                             "max_completed": args.max_completed,
                                            },
                                            {"lm_path": args.ngram_lm}])
    elif model == 7:
        lm = CausalByteLanguageModel(symbol_set=symbol_set,
                                     lang_model_name=args.model_name,
                                     lm_device=device,
                                     lm_path=args.model_dir,
                                     lm_left_context=args.left_context,
                                     fp16=args.fp16,
                                     mixed_case_context=args.mixed_case_context,
                                     case_simple=args.case_simple)
    elif model == 8:
        lm = UniformLanguageModel(symbol_set=symbol_set)
    elif model == 9:
        lm = MixtureLanguageModel(symbol_set=symbol_set,
                                  lm_types=["CAUSALBYTE", "NGRAM"],
                                  lm_weights=[1.0 - args.ngram_mix, args.ngram_mix],
                                  lm_params=[{"lang_model_name": args.model_name,
                                             "lm_device": device,
                                             "lm_path": args.model_dir,
                                             "lm_left_context": args.left_context,
                                             "fp16": args.fp16,
                                             "mixed_case_context": args.mixed_case_context,
                                             "case_simple": args.case_simple,
                                            },
                                            {"lm_path": args.ngram_lm}])
    elif model == 10:
        lm = ClassifierLanguageModel(symbol_set=symbol_set,
                                     lang_model_name=args.model_name,
                                     lm_path=args.model_dir,
                                     lm_device=device,
                                     lm_left_context=args.left_context,
                                     beam_width=args.beam_width,
                                     fp16=args.fp16,
                                     mixed_case_context=args.mixed_case_context,
                                     case_simple=args.case_simple)
    else:
        parser.print_help()
        exit()

    print(f"Model load time = {timer() - start:.2f}")

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

    # Iterate over phrases
    for phrase in phrases:
        symbols = 0
        accum = 0.0
        sent_ppl = 0.0

        sentence = phrase.strip()
        if len(sentence) > 0:
            accum = 0.0

            # Phrase-level output
            if verbose >= 1:
                print(f"sentence = '{sentence}'")

            # Split into characters
            tokens = sentence.split()

            # Optional stripped of symbols not in our set
            if args.skip_oov_symbols:
                tokens_stripped = [token for token in tokens if token.upper() in symbol_set or token == "<sp>"]
                skipped_symbols += len(tokens) - len(tokens_stripped)
                tokens = tokens_stripped

            symbols = len(tokens)

            # SRILM starts with the sentence being evaluated
            if srilm_file:
                for (i, symbol) in enumerate(tokens):
                    if i > 0:
                        srilm_file.write(" ")
                    srilm_file.write(symbol.replace("_", "<sp>"))
                srilm_file.write("\n")

            # Initial previous token is the start symbol, initial context empty
            prev_token = "<s>"
            context = ""

            predict_time_arr = np.array([])
            predict_details_arr = np.array([])

            # Iterate over characters in phrase
            for (i, token) in enumerate(tokens):
                start_predict = timer()
                correct_char = ""

                # Replace the symbolic <sp> token with a space
                if (token == "<sp>"):
                    token = ' '
                    correct_char = ' '
                else:
                    correct_char = token.upper()
                score = 0.0
                next_char_pred = lm.state_update(list(context))

                predict_time = timer() - start_predict
                predict_time_arr = np.append(predict_time_arr, predict_time)
                predict_details_arr = np.append(predict_details_arr,
                                                f"sentence = {sentence}, index = {i}, p( {token} | {prev_token} )")

                # Find the probability for the correct character
                p = next_char_pred[[c[0] for c in next_char_pred].index(correct_char)][1]
                if p == 0:
                    zero_prob += 1
                    accum = 1
                    if verbose >= 2:
                        print(f"p( {token} | {prev_token} ...) = 0")
                        print(f"prediction time = {predict_time:.6f}")
                    break
                else:
                    score = log10(p)

                    # Character-level output
                    if verbose >= 2:
                        print(f"p( {token} | {prev_token} ...) = {p:.6f} [ {score:.6f} ]")
                        print(f"prediction time = {predict_time:.6f}")

                # SRILM line for a character looks like: "	p( w | <s> ) 	= [2gram] 0.095760 [ -1.018816 ]"
                if srilm_file:
                    extra = ""
                    if i > 0:
                        extra = " ..."
                    # The 1gram bit is only relevant for the n-gram, we'll just hard code to 1gram for everything
                    srilm_file.write(f"\tp( {token.replace('_', '<sp>')} | {prev_token.replace('_', '<sp>')}{extra}) \t= [1gram] {p:.6f} [ {score:.6f} ]\n")

                accum += score
                prev_token = token
                context += token
                all_symbol_log_probs.append(score)

            # Compute summary stats on prediction times for this phrase
            per_symbol_time = np.average(predict_time_arr)
            phrase_std = np.std(predict_time_arr)
            phrase_max = np.max(predict_time_arr)
            phrase_min = np.min(predict_time_arr)

            # Add this phrase's prediction times to overall array
            overall_predict_time_arr = np.append(overall_predict_time_arr, predict_time_arr, axis=None)
            overall_predict_details_arr = np.append(overall_predict_details_arr, predict_details_arr, axis=None)

            if accum == 1:
                if verbose >= 1:
                    print("Zero-prob event encountered, terminating phrase")
                    print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, "
                          f"{phrase_max:.6f}]\n")
            else:
                per_symbol_logprob = accum / symbols
                sent_ppl = pow(10, -1 * per_symbol_logprob)

                all_sentence_ppls.append(sent_ppl)

                # Phrase-level output
                if verbose >= 1:
                    print(f"sum logprob = {accum:.4f}, per-symbol logprob = {per_symbol_logprob:.4f}, ppl = {sent_ppl:.4f}")
                    print(f"per-symbol prediction time = {per_symbol_time:.6f} +/- {phrase_std:.6f} [{phrase_min:.6f}, {phrase_max:.6f}]\n")

                sum_per_symbol_logprob += per_symbol_logprob
                phrase_count += 1

                # Optional output to a file with a sentence and its ppl and log prob
                if ppl_file:
                    ppl_file.write(f"{sent_ppl:.4f}\t{accum:.4f}\t{sentence}\n")
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

        sys.stdout.flush()
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
        \nphrases = {phrase_count}, \
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
    sys.stdout.flush()

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

    if args.stats_file:
        # Single line file output, useful for running experiments
        print(f"Outputting stats to {args.stats_file}, running bootstrap on {len(all_symbol_log_probs)} samples.")
        sys.stdout.flush()
        time_bootstrap = timer()
        bootstrap_log_prob = bootstrap(data=(all_symbol_log_probs,),
                                        statistic=np.mean,
                                        confidence_level=0.95,
                                        n_resamples=args.bootstrap_samples,
                                        method=args.bootstrap_method,
                                        random_state=rng)
        print(f"Bootstrap on log probs completed in {(timer() - time_bootstrap):.2f} seconds.")
        sys.stdout.flush()

        ppl_high = pow(10, -1 * bootstrap_log_prob.confidence_interval.low)
        ppl_low = pow(10, -1 * bootstrap_log_prob.confidence_interval.high)
        error_bar = (ppl_high - ppl_low) / 2.0

        print(f"Outputting stats to {args.stats_file}, running bootstrap on {len(all_sentence_ppls)} samples.")
        sys.stdout.flush()
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

        extra = ""
        extra_col = ""
        if args.stats_extra:
            extra = args.stats_extra + "\t"
            extra_col = "\t"
        params = -1
        if model == 4 or model == 7:
            params = lm.get_num_parameters()

        exists = path.isfile(args.stats_file)
        with open(args.stats_file, 'a') as file:
            if not exists:
                # Header if the stats file doesn't already exist
                file.write(f"{extra_col}ppl\tsum_log_prob\tsum_symbols\tboot_ppl_pm\tboot_ppl_low\tboot_ppl_high\tphrases\ttime\tparams\tdate_time\tper_symbol_time\tsd_per_symbol_time\tsentence_ppl\tboot_sentence_ppl\n")
            file.write(f"{extra}"
                         f"{ppl:.6f}"
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
                         f"\t{sentence_ppl_error_bar:.6f}"
                         f"\n")

    # Prediction timing stats for the causal LLM
    if model == 4 or model == 6:
        lm.dump_predict_times()

    # Optionally print the predictions that took an abnormal amount of time
    if args.time_outliers:
        for (i, time) in enumerate(overall_predict_time_arr):
            if time < ci_floor:
                print(f"LOW OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")
            if time > ci_ceiling:
                print(f"HIGH OUTLIER: {overall_predict_details_arr[i]}, predict time = {time:.6f}\n")

