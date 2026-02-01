import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter csv1 by rows that match Sequence and Target columns in csv2")
    parser.add_argument("--csv1", type=str, help="Path to first CSV file (to filter)")
    parser.add_argument("--csv2", type=str, help="Path to second CSV file (with rows to match)")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    # Load the two CSV files
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # Ensure required columns exist
    required_cols = {"Sequence", "Target"}
    if not required_cols.issubset(df1.columns):
        raise ValueError(f"{args.csv1} must contain columns: {required_cols}")
    if not required_cols.issubset(df2.columns):
        raise ValueError(f"{args.csv2} must contain columns: {required_cols}")

    # Perform an inner merge to keep only matching Sequence and Target pairs
    filtered = df1.merge(df2[["Sequence", "Target"]], on=["Sequence", "Target"], how="inner")

    # Save the result
    filtered.to_csv(args.output, index=False)
    print(f"Filtered CSV written to: {args.output}")

    # Find Targets from csv2 that did not make it into the output
    matched_targets = set(filtered["Target"].unique())
    all_targets = set(df2["Target"].unique())
    missing_targets = sorted(all_targets - matched_targets)

    if missing_targets:
        print("\nTargets from csv2 not found in the output:")
        for t in missing_targets:
            print(t)
    else:
        print("\nAll targets from csv2 were found in the output.")

if __name__ == "__main__":
    main()
