#!/usr/bin/env python
# Average a column where each row has columns separated by tabs.
# Default computation is to average a column.
# Can also set the column index by the header name.

from sys import stderr, exit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes something about a column in a tab delimited file, default is to average.")
    parser.add_argument("--in", dest="in_file", required=True, type=str, help="Input tab delimited file")
    parser.add_argument("--col", type=int, default=0, help="column to average (by index)")
    parser.add_argument("--col-name", type=str, help="use column with this header label")
    parser.add_argument("--no-header", action="store_true", help="file doesn't have header row")
    parser.add_argument("--count", action="store_true", help="compute count")
    parser.add_argument("--sum", action="store_true", help="compute sum")
    parser.add_argument("--min", action="store_true", help="compute min")
    parser.add_argument("--max", action="store_true", help="compute max")

    args = parser.parse_args()

    if args.col and args.col_name:
        print(f"ERROR: set either --col or --col-name, not both!", file=stderr)
        exit(1)
    if args.no_header and args.col_name:
        print(f"ERROR: can't use --col-name with headerless files!", file=stderr)
        exit(1)
    if sum([args.count, args.sum]) > 1:
        print(f"ERROR: can't set more than one type of computation", file=stderr)
        exit(1)

    total = 0.0
    num = 0
    first = True
    col_index = args.col
    min_val = float("+inf")
    max_val = float("-inf")

    with open(args.in_file) as file:
        for line in file:
            if first and args.col_name:
                # Find the column index we should use
                cols = line.split("\t")
                try:
                    col_index = cols.index(args.col_name)
                except ValueError:
                    print(f"ERROR: couldn't find column '{args.col_name}'!", file=stderr)
                    exit(1)

            if not first or args.no_header:
                cols = line.strip().split("\t")
                total += float(cols[col_index])
                num += 1
                min_val = min(min_val, float(cols[col_index]))
                max_val = max(max_val, float(cols[col_index]))
            first = False

    if args.sum:
        print(f"{total:.6f}")
    elif args.count:
        print(f"{num}")
    elif args.min:
        print(f"{min_val:.6f}")
    elif args.max:
        print(f"{max_val:.6f}")
    else:
        print(f"{(total / num):.6f}")