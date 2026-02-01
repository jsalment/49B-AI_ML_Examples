import os
import re
import argparse
import numpy as np
import pandas as pd
from glob import glob

parser = argparse.ArgumentParser(description="Compute mean of 7th numeric column from .out files")
parser.add_argument("--in_dir", default='data/MissingSpecificityOutFolder', help="Directory containing .out files")
parser.add_argument("--out_path", default="data/mean_out_large.csv", help="Output CSV path")
args = parser.parse_args()

# regex: floats and ints, including scientific notation
num_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')

def mean_of_7th_from_file(path):
    vals = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            # find numeric tokens on the line
            toks = num_re.findall(line)
            if len(toks) >= 7:
                try:
                    v = float(toks[6])  # 7th numeric token (0-based index 6)
                    vals.append(v)
                except Exception:
                    # Skip if conversion fails for some token
                    continue
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))

# find .out files (case-insensitive)
patterns = ["*.out", "*.OUT"]
file_paths = []
for p in patterns:
    file_paths.extend(glob(os.path.join(args.in_dir, p)))
file_paths = sorted(set(file_paths))

if not file_paths:
    print(f"No .out files found in {args.in_dir}")
else:
    print(f"Found {len(file_paths)} .out files. Processing...")

results = []
for p in file_paths:
    fname = os.path.basename(p)
    mean7 = mean_of_7th_from_file(p)
    results.append({"Filename": fname, "Mean7": mean7})

df = pd.DataFrame(results)
df.to_csv(args.out_path, index=False)
print(f"\nSaved {len(df)} entries to {args.out_path}")