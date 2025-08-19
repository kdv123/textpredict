# Home to code we want to share from different scripts that perform evaluations using the language models.

from datasets import load_dataset
import re
from torch import set_num_threads
from psutil import cpu_count
from typing import List, Tuple
from datetime import datetime
from socket import gethostname
from timeit import default_timer as timer
from ngram import NGramLanguageModel
from causal_byte import CausalByteLanguageModel
from causal import CausalLanguageModel
from language_model import LanguageModel
from string import whitespace
from sys import stderr
from argparse import ArgumentParser, Namespace

def add_args(parser: ArgumentParser) -> None:
    """
    Add command line switches that are the same between evaluation scripts
    :param parser:
    :return:
    """
    parser.add_argument("--phrases", type=str, help="Input text file with phrases")
    parser.add_argument("--phrase-limit", type=int, help="Max phrases to evaluate")
    parser.add_argument("--dataset", type=str, help="Hugging Face dataset to load phrases from")
    parser.add_argument("--dataset-split", type=str, help="Split to use from the Hugging Face dataset")
    parser.add_argument("--dataset-phrase-col", type=str, default="text", help="Dataset column containing phrases")
    parser.add_argument("--dataset-limit-col", type=str, help="Dataset column used to limit to subset of dataset")
    parser.add_argument("--dataset-limit-val", type=str, help="Value to match to include in phrases")
    parser.add_argument("--use-mps", action="store_true", help="Use MPS Apple Silicon GPU during inference")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA GPU during inference")
    parser.add_argument("--fp16", action="store_true", help="Convert model to fp16 (CUDA only)")
    parser.add_argument("--num-cores", type=int, help="Limit pytorch to specified number of cores")
    parser.add_argument("--ngram-lm", help="N-gram model to load")
    parser.add_argument("--ngram", action="store_true", help="Use n-gram language model")
    parser.add_argument("--causal", action="store_true", help="Use causal LLM")
    parser.add_argument("--byte", action="store_true", help="LLM uses byte tokenization")
    parser.add_argument("--model-name", help="Model name of LLM")
    parser.add_argument("--model-dir", help="Local directory to load fine-tuned LLM")
    parser.add_argument("--lora", action="store_true", default=False, help="Use LoRA adapter with base model")
    parser.add_argument("--lora-path", help="Hugging Face or local path to LoRA adapter")
    parser.add_argument("--out-stats", help="Output summary stats to this tab delimited file")
    parser.add_argument("--out-extra", action="append", dest="out_extra_cols", help="Output additional column to stats file, format: COLUMN_NAME,VALUE")
    parser.add_argument("--lower", action="store_true", help="Lowercase the phrases")
    parser.add_argument("--drop-numbers", action="store_true", help="Drop phrases with numbers")
    parser.add_argument("--drop-regex", type=str, help="Drop phrases that don't match this regular expression")
    parser.add_argument("--drop-max-len", type=int, help="Drop phrases with more than this many characters")
    parser.add_argument("--strip-symbols", action="store_true", help="Strip symbols from phrases except apostrophe")
    parser.add_argument("--truncate-max-len", type=int, help="Truncate phrases longer than this many characters")
    parser.add_argument("--strip-start-end-words", action="store_true", help="Strip <s> and </s> words from phrases")
    parser.add_argument("--space-symbol", type=str, default="<sp>", help="Pseudo-word used by n-gram model for space character")
    parser.add_argument("--skip-norm", action="store_true", default=False, help="Skip normalization over symbols for n-gram model, for matching SRILM output when using LM with extra symbols")
    parser.add_argument("--left-context", help="Left language model context for transformer models")
    parser.add_argument("--left-context-file", help="File with left language model context for transformer model")
    parser.add_argument("--case-simple", action="store_true", default=False, help="Lowercase then simple automatic casing of phrases")
    parser.add_argument("--symbols", type=str, default="abcdefghijklmnopqrstuvwxyz' ", help="Symbols we make predictions over")
    parser.add_argument("--predict-lower", action="store_true", default=False, help="Prediction of lowercase characters only")
    parser.add_argument("--previous-max-len", type=int, help="Enable left context from previous phrases up to this many characters")
    parser.add_argument("--previous-add", type=str, default=" ", help="Text to add after last phrase when using context from previous phrases")
    parser.add_argument("--unstripped-context", action="store_true", default=False, help="Use left context before stripping of symbols")
    parser.add_argument("--batch-size", type=int, help="Max batch size of inferences for transformer models")

def check_args_for_errors(args: Namespace) -> None:
    """
    Check for fatal errors for command line switches shared by evaluation scripts
    :param args: Command line arguments passed to main function
    :return:
    """
    if args.ngram and not args.ngram_lm:
        print(f"ERROR: N-gram model requires n-gram model to be specified with --ngram-lm!", file = stderr)
        exit(1)
    if not args.phrases and not args.dataset:
        print(f"ERROR: Must specify either --phrases or --dataset!", file = stderr)
        exit(1)
    if args.phrases and args.dataset:
        print(f"ERROR: Can't specify both --phrases and --dataset!", file = stderr)
        exit(1)
    if (args.dataset_limit_col and not args.dataset_limit_val) or (args.dataset_limit_val and not args.dataset_limit_col):
        print(f"ERROR: Must specify both --dataset-limit-col and --dataset-limit-val!", file = stderr)
        exit(1)
    if args.out_extra_cols:
        for extra in args.out_extra_cols:
            cols = extra.split(",")
            if len(cols) != 2:
                print(f"ERROR: Invalid comma separated pair in --out-extra: {extra}!", file = stderr)
                exit(1)
    if args.lora and not args.causal:
        print("ERROR: LoRA adapter is only currently supported for causal model!", file = stderr)
        exit(1)
    if args.lora and not args.lora_path:
        print("ERROR: To use a LoRA adapter you must specify the adapter path using --lora-path. --model-name specifies the base model!", file = stderr)
        exit(1)
    if args.left_context and args.left_context_file:
        print("ERROR: Only one of --left-context or --left-context-file can be specified!", file = stderr)
        exit(1)
    if args.lower and args.case_simple:
        print("ERROR: Only one of --lower and --case-simple should be specified!", file = stderr)
        exit(1)
    if args.unstripped_context and not args.strip_symbols:
        print("ERROR: Using unstripped context requires --strip-symbols!", file = stderr)
        exit(1)
    if args.batch_size and not args.causal and not args.byte:
        print("ERROR: --batch-size only applies to transformer models!", file = stderr)
        exit(1)

def check_args_for_warnings(args: Namespace) -> None:
    """
    Check for suspicious things in the command line switches shared by evaluation scripts
    :param args: Command line arguments passed to main function
    :return:
    """
    # Check for settings that are suspicious but don't result in termination
    if args.ngram and not args.lower:
        print(f"WARNING: Unless you have a mixed case n-gram model, you should set --lower!", file = stderr)
    if args.ngram and args.case_simple:
        print(f"WARNING: Unless you have a mixed case n-gram model, you should not set --case-simple!", file = stderr)

def _load_phrases_plaintext(filename: str,
                            phrase_limit: int = None) -> List[str]:
    """
    Load phrases from a plaintext file with a phrase on each line
    :param filename: Filename containing the phrases
    :param phrase_limit: Optional limit to the first so many lines
    :return: List of phrases
    """
    # Read in a plain text file that has a sentence on each line
    phrase_file = open(filename, "r")
    phrases = phrase_file.readlines()
    phrase_file.close()

    # Get rid of any line ending in the plaintext file
    # This use to be done in the loop in the evaluation script
    phrases = [phrase.strip() for phrase in phrases]

    if phrase_limit:
        phrases = phrases[:phrase_limit]
    return phrases

def _load_phrases_dataset(name: str,
                          split: str,
                          phrase_col: str = "text",
                          phrase_limit: int = None,
                          limit_col = None,
                          limit_val = None) -> List[str]:
    """
    Load phrases from a dataset
    :param name: Name of the dataset
    :param split: Name of the split in the dataset, e.g. train or validation
    :param phrase_col: Name of the column in the dataset containing the phrases
    :param phrase_limit: Optional limit to the first so many rows
    :param limit_col: Optional column to use to select subset of rows
    :param limit_val: Value to use to select subset of rows
    :return: List of phrases
    """
    # Read the sentences from a Hugging Face dataset
    dataset = load_dataset(path=name, split=split)
    # Optional limiting to rows with a column matching a given value
    if limit_col:
        dataset = dataset.filter(
            function=lambda example:
            example[limit_col] == limit_val)
    phrases = dataset[phrase_col]

    if phrase_limit:
        phrases = phrases[:phrase_limit]
    return phrases

def _filter_phrases(phrases: List[str],
                   drop_numbers: bool = False,
                   drop_max_len: int = None,
                   drop_regex: str = None) -> List[str]:
    """
    Remove entire phrases based on different criteria
    :param phrases: Original list of all the phrases
    :param drop_numbers: Drop any phrase with a character 0-9
    :param drop_max_len: Drop any phrase with more than this many characters
    :return: List of phrases
    """
    result = []
    for phrase in phrases:
        if (not drop_max_len or len(phrase) <= drop_max_len) and \
           (not drop_numbers or not re.search(r'\d', phrase)) and \
           (not drop_regex or re.match(drop_regex, phrase)):
           result.append(phrase)
    return result

def _normalize_phrases(phrases: List[str],
                       lower: bool = False,
                       strip_symbols: bool = False,
                       truncate_max_len: int = None,
                       strip_start_end_words: bool = False) -> Tuple[List[str], List[str]]:
    """
    Perform text normalization on the phrases
    :param phrases: Original list of all the phrases
    :param lower: Lowercase all phrases
    :param strip_symbols: Converts characters besides A-Z and apostrophe to space then collapses contiguous whitespace
    :param truncate_max_len: Truncate any phrase with this many characters
    :param strip_start_end_words: Strip the start and end words <s> and </s>
    :return: Tuple(List of normalized phrases, List of context before symbol stripped, None if strip_symbols False)
    """
    result_phrases = []
    result_unstripped = []

    for phrase in phrases:
        index_to_unstripped = None
        if lower:
            phrase = phrase.lower()
        # Drop words until we meet the truncation max length
        if truncate_max_len and len(phrase) > truncate_max_len:
            # First cut it off
            phrase = phrase[:truncate_max_len]
            # Then remove characters until we reach a space
            while phrase[-1] != " ":
                phrase = phrase[:-1]
            phrase = phrase.strip()
        if strip_start_end_words:
            phrase = phrase.replace("<s> ", "").replace(" </s>", "")
        if strip_symbols:
            # Go through the string one character at-a-time stripping symbols
            # Changed from using two regular expressions since we may want to know previous context including symbols
            phrase_stripped = ""
            between_words = True
            has_output_letter = False
            index_to_unstripped = []
            for i in range(len(phrase)):
                ch = phrase[i]
                if (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z") or ch == "'":
                    if between_words and has_output_letter:
                        phrase_stripped +=  " "
                        index_to_unstripped.append(phrase[0:i-1])
                    phrase_stripped += ch
                    between_words = False
                    has_output_letter = True
                    index_to_unstripped.append(phrase[0:i])
                else:
                    between_words = True
            # Extra last one containing entire unstripped context
            index_to_unstripped.append(phrase)
            phrase = phrase_stripped

        # It could be the case we normalized it to be blank
        if len(phrase) > 0:
            result_phrases.append(phrase)
            result_unstripped.append(index_to_unstripped)
    return result_phrases, result_unstripped

def case_simple(phrase: str) -> str:
    """
    Handles the optional simple heuristic based casing of phrases
    We first lowercase the text, then use simple rules to add case back
    :param phrase: the existing left context
    :return: the simple cased version of the left context (or original if not enabled)
    """

    # The trailing spaces makes sure we don't replace in the left context until we are
    # sure it is one of these words, otherwise we might capitalize mid-sentence words like it
    simple_upper_words = {"i ": "I ",
                          "i'll ": "I'll ",
                          "i've ": "I've ",
                          "i'd ": "I'd ",
                          "i'm ": "I'm "}

    # Remove any existing casing since we are simulating automatic casing
    phrase = phrase.lower()
    if len(phrase) > 0:
        # First uppercase the first letter if it is A-Z
        if 'a' <= phrase[0] <= 'z':
            phrase = phrase[0].upper() + phrase[1:]

        for key, value in simple_upper_words.items():
            phrase = phrase.replace(key, value)
    return phrase

def count_words(phrases: List[str]) -> int:
    """
    Count the number of words in a list of phrases
    :param phrases: List of phrases
    :return: Integer count
    """
    count = 0
    for phrase in phrases:
        count += len(phrase.split())
    return count

def count_chars(phrases: List[str]) -> int:
    """
    Count the number of characters in a list of phrases
    :param phrases: List of phrases
    :return: Integer count
    """
    count = 0
    for phrase in phrases:
        count += len(phrase)
    return count

def set_cpu_cores(args: Namespace) -> None:
    """
    Optimize settings for CPU or GPU computation
    :param args: Command line arguments passed to main function
    :return:
    """
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

def get_device(args: Namespace) -> str:
    """
    Set the computation device string based on the command line switches
    :param args:
    :return: String: cpu, mps, or cuda
    """
    device = "cpu"
    if args.use_mps:
        device = "mps"
    elif args.use_cuda:
        device = "cuda"
    return device

def load_phrases(args: Namespace,
                 quiet: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load the phrases we are going to simulate writing
    :param args: Command line arguments passed to main function
    :param quiet: Can be used to supress informational output
    :return: List of phrases
    """
    if args.phrases:
        phrases = _load_phrases_plaintext(filename=args.phrases, phrase_limit=args.phrase_limit)
    else:
        phrases = _load_phrases_dataset(name=args.dataset,
                                                   split=args.dataset_split,
                                                   phrase_limit=args.phrase_limit,
                                                   phrase_col=args.dataset_phrase_col,
                                                   limit_col=args.dataset_limit_col,
                                                   limit_val=args.dataset_limit_val)
    if not quiet:
        print(f"Loaded {len(phrases)} phrases, words = {count_words(phrases)}")
        print(f"First phrase: '{phrases[0]}'")
        print(f"Last phrase: '{phrases[-1]}'")

    # First phrase is to potentially get rid of some phrases
    phrases = _filter_phrases(phrases=phrases,
                              drop_max_len=args.drop_max_len,
                              drop_numbers=args.drop_numbers,
                              drop_regex=args.drop_regex)
    if not quiet:
        print(f"After filtering: {len(phrases)} phrases, words = {count_words(phrases)}")
    if len(phrases) == 0:
        print(f"ERROR: All phrases were filtered out!", file = stderr)
        exit(1)

    # Second phrase is to normalize the text in various ways
    phrases, unstripped = _normalize_phrases(phrases=phrases,
                                             lower=args.lower,
                                             strip_symbols=args.strip_symbols,
                                             truncate_max_len=args.truncate_max_len,
                                             strip_start_end_words=args.strip_start_end_words)

    if not quiet:
        print(f"After normalization: {len(phrases)} phrases, words = {count_words(phrases)}")
        print(f"Normalized first phrase: '{phrases[0]}'")
        print(f"Normalized last phrase: '{phrases[-1]}'")

    return phrases, unstripped

def print_startup_info(args: Namespace) -> None:
    """
    Handy stuff to print out in our log files at the start of a run
    :param args: Command line arguments passed to main function
    :return: None
    """
    print(f"START: {datetime.now()}")
    print(f"ARGS: {args}")
    print(f"HOSTNAME: {gethostname()}")

def load_language_model(args: Namespace,
                        symbol_set: List[str],
                        device: str,
                        quiet: bool = False) -> LanguageModel:
    """
    Load one of the language model types shared by all evaluation scripts.
    :param args: Command line arguments passed to main function
    :param symbol_set: List of symbols to send to language model
    :param device: Device to load the model on
    :param quiet: Can be used to supress informational output
    :return: LanguageModel object, or None if type not supported
    """
    start = timer()
    lm = None

    if args.ngram_lm:
        if not quiet:
            print(f"Loading n-gram LM: {lm}")
        lm = NGramLanguageModel(symbol_set=symbol_set,
                                lm_path=args.ngram_lm,
                                skip_symbol_norm=args.skip_norm,
                                space_symbol=args.space_symbol,
                                )
    elif args.byte:
        if not quiet:
            print(f"Loading byte LLM: {args.model_name}, model directory {args.model_dir}")
        lm = CausalByteLanguageModel(symbol_set=symbol_set,
                                     lang_model_name=args.model_name,
                                     lm_device=device,
                                     lm_path=args.model_dir,
                                     lm_left_context=args.left_context,
                                     fp16=args.fp16,
                                     case_simple=args.case_simple,
                                     batch_size=args.batch_size,
                                     predict_lower=args.predict_lower,
                                     )
    elif args.causal:
        if not quiet:
            print(f"Loading causal LLM: {args.model_name}, model directory {args.model_dir}")
        lm = CausalLanguageModel(symbol_set=symbol_set,
                                 lang_model_name=args.model_name,
                                 lm_device=device,
                                 lm_path=args.model_dir,
                                 lm_left_context=args.left_context,
                                 beam_width=args.beam_width,
                                 fp16=args.fp16,
                                 max_completed=args.max_completed,
                                 lora=args.lora,
                                 lora_path=args.lora_path,
                                 batch_size=args.batch_size,
                                 predict_lower=args.predict_lower,
                                 )

    if not quiet:
        print(f"Model load time = {timer() - start:.2f}")
    return lm

def prep_left_context(args: Namespace) -> None:
    """
    Handles optional left context loading from a file and conversion of space character in provided context
    :param args: Command line arguments passed to main function
    :return:
    """
    if args.left_context_file:
        try:
            with open(args.left_context_file, "r", encoding="utf-8") as f:
                contents = ""
                for line in f:
                    if line not in whitespace:
                        contents += f" {line.rstrip()}"
                contents = contents.strip()
                args.left_context = contents
        except FileNotFoundError:
            print(f"ERROR: cannot open left context file: {args.left_context_file}!", file = stderr)
            exit(1)
    elif not args.left_context:
        # Default left context to blank string
        args.left_context = ""

    # Allow passing in of space characters in the context using the space pseudo-word
    args.left_context = args.left_context.replace(args.space_symbol, " ")

def sanity_check_symbols(symbol_set: List[str],
                         phrases: List[str],
                         predict_lower: bool) -> None:
    """
    Check all symbols in the phrases are in our symbol set
    :param symbol_set: List of all the symbols we hope to make predictions about
    :param phrases: List of all phrases we are evaluating on
    :param predict_lower: Whether we are always predicting lowercase even if phrases are mixed case
    :return:
    """
    all_phrase_symbols = set()
    for phrase in phrases:
        for ch in phrase:
            all_phrase_symbols.add(ch)
    bad_symbols = []
    for symbol in all_phrase_symbols:
        # If predictions are always lowercase then we can allow uppercase letters
        if predict_lower:
            symbol = symbol.lower()
        if symbol not in symbol_set:
            bad_symbols.append(symbol)
    if len(bad_symbols) > 0:
        bad_symbols.sort()
        print(f"ERROR: characters in phrases not in symbol set: {bad_symbols}!", file = stderr)
        exit(1)

def update_context(context: str,
                   max_len: int,
                   previous_add: str) -> str:
    """
    Drops words from the front of a string until the length is <= max_len
    :param context: String of text
    :param max_len: Maximum length of the context
    :param previous_add: Text to add to previous context to separate sentences
    :return: Shortened context
    """
    if max_len <= 0:
        return ""
    elif len(context) > 0:
        # Add a space since this is a new sentence
        if previous_add:
            context += previous_add
        # Drop words from the front of the context to get under the limit
        while len(context) > max_len:
            pos = context.find(" ")
            # If there is no space, or it is at the very end of the string, then we fall back to truncating
            if pos != -1 and pos != len(context) - 1:
                context = context[pos + 1:]
            else:
                context = context[-max_len:]
    return context