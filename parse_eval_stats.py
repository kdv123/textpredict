# Parse out importat stats from the end of the output of the lm_eval.py script.

import sys

if __name__ == "__main__":
    in_overall = False

    results = ["", "", "", "", ""]

    for line in sys.stdin:
        line = line.strip()
        if line.startswith("OVERALL"):
            in_overall = True
        elif in_overall and line.startswith("zero-prob events ="):
            results[0] = line.split()[-1]
        elif in_overall and line.startswith("ppl = "):
            results[1] = line.split()[-1]
        elif in_overall and line.startswith("sum logprob = "):
            results[2] = line.split()[-1]
        elif in_overall and line.startswith("inference time = "):
            results[3] = line.split()[-1]
        elif in_overall and line.startswith("per-symbol prediction time = "):
            results[4] = line.split()[4]
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            print(f"{sys.argv[i]}", end="\t")
    print("\t".join(results))
