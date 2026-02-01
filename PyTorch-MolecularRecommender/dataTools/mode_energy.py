#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import pandas as pd
from glob import glob
import csv

parser = argparse.ArgumentParser(
    description="Compute BINNED MODE (peak) of 7th numeric column from .out files using fixed NUMBER of bins"
)
parser.add_argument("--in_dir", default='data/MissingSpecificityOutFolder',
                    help="Directory containing .out files")
parser.add_argument("--out_path", default="data/mode_out_large.csv",
                    help="Output CSV path")
parser.add_argument("--num_bins", type=int, default=60,
                    help="Number of bins for histogram (e.g. 40, 60, 80). Higher = finer resolution. Default: 60")
args = parser.parse_args()

# Regex for numbers (including scientific notation)
num_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')

def binned_mode_of_7th(path, num_bins=60):
    vals = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            toks = num_re.findall(line)
            if len(toks) >= 7:
                try:
                    v = float(toks[6])
                    vals.append(v)
                except:
                    continue
    
    if len(vals) == 0:
        return float("nan")
    if len(vals) == 1:
        return round(vals[0], 3)

    vals = np.array(vals)
    
    # Use fixed number of bins — this automatically adapts bin width to data range
    hist, bin_edges = np.histogram(vals, bins=num_bins, range=(vals.min(), vals.max()))
    
    # Find bin with maximum count
    max_idx = np.argmax(hist)
    mode_center = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0
    
    return round(mode_center, 3)

# Find .out files
patterns = ["*.out", "*.OUT"]
file_paths = [p for pat in patterns for p in glob(os.path.join(args.in_dir, pat))]
file_paths = sorted(set(file_paths))

if not file_paths:
    print(f"No .out files found in {args.in_dir}")
    exit()

print(f"Found {len(file_paths)} .out files. Processing with {args.num_bins} bins per file...")

results = []
for p in file_paths:
    fname = os.path.basename(p)
    mode_val = binned_mode_of_7th(p, num_bins=args.num_bins)
    results.append({"Filename": fname, "Mode7": mode_val})

df = pd.DataFrame(results)
df.to_csv(args.out_path, index=False, quoting=csv.QUOTE_ALL)

print(f"\nSaved {len(df)} entries to {args.out_path}")
print(f"   Column 'Mode7' = peak of distribution (binned mode using {args.num_bins} bins)")