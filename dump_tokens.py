#!/usr/bin/env python
# Dumps the text of all tokens of a LLM to standard out.
# Or can show the tokenization of a specific string.

from causal import CausalLanguageModel
from causal_byte import CausalByteLanguageModel
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Model name of causal model")
    parser.add_argument("--symbols", type=str, default="abcdefghijklmnopqrstuvwxyz' ", help="Symbols we make predictions over")
    parser.add_argument("--byte", action="store_true", help="Use byte tokenized LLM")
    parser.add_argument("--str", type=str, help="Show tokenization of this string")

    args = parser.parse_args()

    if args.byte:
        lm = CausalByteLanguageModel(symbol_set=args.symbols,
                                     lang_model_name=args.model_name)
    else:
        lm = CausalLanguageModel(symbol_set=args.symbols,
                                 lang_model_name=args.model_name)
    if args.str:
        print(f"Tokenization of String: '{args.str}'")
        print(f"{lm.get_tokenization(args.str)}")
    else:
        tokens = lm.get_all_tokens_text()
        for i, token in enumerate(tokens):
            print(f"%d\t'%s'" % (i, token))