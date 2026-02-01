#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Join Mode7 (from CSV1) into CSV2 where (AptamerID,Target) pairs match (case-insensitive)."
)
parser.add_argument("--csv1", default="data/mode_out.csv", help="Path to CSV1 (contains Mode7 and AptamerID, Target).")
parser.add_argument("--csv2", default="data/seq_SMILES_large.csv", help="Path to CSV2 (to receive Mode7 column).")
parser.add_argument("--out", default="data/seq_SMILES_large_Mode7.csv", help="Output path for CSV2 augmented with Mode7.")
parser.add_argument("--drop-unmatched-rows", action="store_true",
                    help="If set, remove all rows from CSV2 that do not have a matching Mode7 value (i.e. keep only fully matched rows).")
parser.add_argument("--mean-names", nargs="*", default=["Mode7", "Mean", "Mode7"],
                    help="Candidate column names in CSV1 that might contain the value (searched in order).")
args = parser.parse_args()

# Load as strings to avoid dtype surprises
df1 = pd.read_csv(args.csv1, dtype=str, keep_default_na=False)
df2 = pd.read_csv(args.csv2, dtype=str, keep_default_na=False)

# Required columns check
required = ["AptamerID", "Target"]
for c in required:
    if c not in df1.columns:
        print(f"ERROR: CSV1 ({args.csv1}) missing required column '{c}'", file=sys.stderr)
        sys.exit(1)
    if c not in df2.columns:
        print(f"ERROR: CSV2 ({args.csv2}) missing required column '{c}'", file=sys.stderr)
        sys.exit(1)

# Check that CSV1 has Mode7 column
if "Mode7" not in df1.columns:
    print(f"ERROR: CSV1 ({args.csv1}) does not contain 'Mode7' column (exact name required).", file=sys.stderr)
    print("Available columns in CSV1:", list(df1.columns), file=sys.stderr)
    sys.exit(1)

# Create lowercase keys for case-insensitive matching
df1["_apt_key"] = df1["AptamerID"].astype(str).str.strip().str.lower()
df1["_tgt_key"] = df1["Target"].astype(str).str.strip().str.lower()

df2["_apt_key"] = df2["AptamerID"].astype(str).str.strip().str.lower()
df2["_tgt_key"] = df2["Target"].astype(str).str.strip().str.lower()

# Coerce Mode7 to numeric
df1["_Mode7_num"] = pd.to_numeric(df1["Mode7"].astype(str).str.replace(",", ""), errors="coerce")
df1["_Mode7_val"] = df1["_Mode7_num"].where(df1["_Mode7_num"].notna(), df1["Mode7"])

# Deduplicate CSV1
df1_keys = df1[["_apt_key", "_tgt_key", "_Mode7_val"]].drop_duplicates(subset=["_apt_key", "_tgt_key"], keep="first")

# Perform left join
merged = df2.merge(
    df1_keys,
    left_on=("_apt_key", "_tgt_key"),
    right_on=("_apt_key", "_tgt_key"),
    how="left",
    suffixes=("", "_from_csv1")
)

# Add Mode7 column
merged["Mode7"] = merged["_Mode7_val"]

# Convert to numeric and round
merged['Affinity'] = pd.to_numeric(merged['Affinity'], errors='coerce').round(2)
merged["Mode7"] = pd.to_numeric(merged["Mode7"], errors='coerce').round(2)

# Clean up temporary columns
for col in ["_apt_key", "_tgt_key", "_Mode7_num", "_Mode7_val"]:
    if col in merged.columns:
        merged.drop(columns=[col], inplace=True)

# === NEW: Optionally drop unmatched rows ===
initial_count = len(merged)
if args.drop_unmatched_rows:
    before_drop = len(merged)
    merged = merged[merged["Mode7"].notna()].reset_index(drop=True)
    dropped = before_drop - len(merged)
    print(f"Dropped {dropped} rows without matching Mode7 (kept only matched).")
else:
    print("Kept all rows from CSV2 (including unmatched ones with Mode7 = NaN).")

# Save result
merged.to_csv(args.out, index=False)

# Final report
matched = merged["Mode7"].notna().sum()
total = len(merged)
print(f"\nFinal dataset: {total} rows")
print(f"   → {matched} rows have Mode7 value ({matched/total*100:.1f}%)")
print(f"Saved to: {args.out}")