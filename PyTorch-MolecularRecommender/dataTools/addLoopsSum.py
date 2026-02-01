import pandas as pd
import argparse

"""
This script is inteneded to add an extra column to sequence.csv called 
LoopSum which contains the total number of '.' characters in the
loops of the two-dimensional structure.
"""

def count_all_periods(structure: str) -> int:
    """
    Returns the total number of '.' characters in the structure string.
    """
    if not isinstance(structure, str):
        return 0
    return structure.count('.')


def main():
    parser = argparse.ArgumentParser(description="Count total number of periods in TwoDStructure column.")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output CSV file")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input)

    if "TwoDStructure" not in df.columns:
        raise ValueError("Input CSV must contain a 'TwoDStructure' column.")

    # Compute total dot count
    df["LoopSum"] = df["TwoDStructure"].apply(count_all_periods)

    # Write output
    df.to_csv(args.output, index=False)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
