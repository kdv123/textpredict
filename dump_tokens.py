#!/usr/bin/env python
# Dumps the token text of a LLM to standard out.

from causal import CausalLanguageModel
import argparse
from language_model import alphabet

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",
                        help="Model name of causal model")
    args = parser.parse_args()

    symbol_set = alphabet()
    lm = CausalLanguageModel(symbol_set=symbol_set,
                             lang_model_name=args.model_name,
                             lm_device="cpu",
                             lm_path=args.model_dir,
                             lm_left_context=args.left_context,
                             beam_width=args.beam_width,
                             fp16=args.fp16,
                             mixed_case_context=args.mixed_case_context,
                             case_simple=args.case_simple,
                             max_completed=args.max_completed)

    tokens = lm.get_all_tokens_text()
    for i, token in enumerate(tokens):
        print(f"%d\t%s" % (i, token))