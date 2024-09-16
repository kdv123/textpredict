#!/usr/bin/env python
# Converts sentences into letter data for training or evaluating a letter language model.
# Changes spaces between words into: <sp>
#
# Optionally can:
#   Limit output to a specified maximum number of words.
#   Drop everything until the first tab (if any)
#   Lowercase
#   Remove end of sentence punctuation ?!.
#   Strip commas from sentences
#   Remove sentences with things like numbers or symbols

import sys
import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts and filters a file with one sentence per line into letter-at-a-time style")

    parser.add_argument("--lower", help="lowercase text", action="store_true")
    parser.add_argument("--drop-first-col", help="drop first column in a tab delimited text file", action="store_true")
    parser.add_argument("--strip-commas", help="strip commas from sentence (done first)", action="store_true")
    parser.add_argument("--drop-end-punc", help="drop end of sentence punctuation in [.?!]", action="store_true")
    parser.add_argument("--drop-non-alpha", help="drop sentences with characters outside [A-Za-z' ]", action="store_true")
    parser.add_argument("--max-words", type=int, help="max words to output")

    args = parser.parse_args()

    re_end_punc = re.compile("[!?.]$")
    re_all_alpha = re.compile("^[a-zA-Z ']+$")

    total_words = 0
    for line in sys.stdin:
        if args.drop_first_col:
            pos = line.find("\t")
            if pos != -1:
                line = line[pos+1:]
        
        # Nuke any existing sentence start/end symbols, plus leading/trailing whitespace
        line = line.replace("<s>", "").replace("</s>", "").strip()

        if args.strip_commas:
            line = line.replace(",", "")
        if args.lower:
            line = line.lower()
        if args.drop_end_punc:
            line = re.sub(re_end_punc, '', line)
        if not args.drop_non_alpha or re.match(re_all_alpha, line):
            words = line.split()
            output = ""
            for word in words:
                if len(output) > 0:
                    output += " <sp>"
                for i in range(len(word)):
                    if len(output) > 0:
                        output += " "
                    output += word[i]
                total_words += 1
            print(output)

            if args.max_words and total_words >= args.max_words:
                break
