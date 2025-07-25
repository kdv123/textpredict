#!/usr/bin/env python
# Average a column where each row has columns separated by tabs

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_file", required=True, type=str, help="Input tab delimited file")
    parser.add_argument("--col", type=int, default=0, help="column to average")
    parser.add_argument("--no-header", action="store_true", help="file doesn't have header row")
    args = parser.parse_args()

    total = 0.0
    num = 0
    first = True
    with open(args.in_file) as file:
        for line in file:
            if not first or args.no_header:
                cols = line.strip().split("\t")
                total += float(cols[args.col])
                num += 1
            first = False

    print(f"{(total / num):.4f}")