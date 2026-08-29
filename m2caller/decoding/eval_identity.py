import argparse, os, subprocess, tempfile
from typing import List
import edlib

def identity_stats(hyp: str, ref: str):
    # Levenshtein via edlib; identity = 1 - dist/len(ref)
    res = edlib.align(hyp, ref, task="distance")
    dist = res["editDistance"]
    L = max(1, len(ref))
    ident = 1.0 - dist / L
    return ident, dist, L

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--ref", required=True, help="reference fasta (single sequence)")
    args = p.parse_args()
    # naive parsing
    def read_fasta(path):
        seqs = []
        cur = []
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        return seqs
    hyps = read_fasta(args.fasta)
    refs = read_fasta(args.ref)
    ref = refs[0]
    totals = [identity_stats(h, ref)[0] for h in hyps]
    print(f"Mean identity over {len(hyps)} reads: {sum(totals)/max(1,len(totals)):.4f}")
