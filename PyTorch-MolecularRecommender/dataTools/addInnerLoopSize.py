import pandas as pd
import argparse
import re

"""
This script is inteneded to add an extra column to sequence.csv called 
InnerLoopSize which contains the number of '.' characters in the innermost
loop of the two-dimensional structure.
"""

def count_periods_in_innermost_parentheses(structure: str) -> int:
    """
    Returns the number of '.' characters inside the innermost parentheses pair.
    If no parentheses exist, returns 0.
    """
    if not isinstance(structure, str):
        return 0

    # Use regex to find all parenthetical groups
    matches = re.findall(r'\([^()]*\)', structure)
    if not matches:
        return 0

    # Take the *last* match — the innermost group
    innermost = matches[-1]
    return innermost.count('.')


def main():
    parser = argparse.ArgumentParser(description="Compute inner loop size from TwoDStructure column.")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output CSV file")
    args = parser.parse_args()

    # Read the input CSV
    df = pd.read_csv(args.input)

    if "TwoDStructure" not in df.columns:
        raise ValueError("Input CSV must contain a 'TwoDStructure' column.")

    # Compute InnerLoopSize
    df["InnerLoopSize"] = df["TwoDStructure"].apply(count_periods_in_innermost_parentheses)

    # Save to output CSV
    df.to_csv(args.output, index=False)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
