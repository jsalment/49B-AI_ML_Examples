import pandas as pd
import argparse
import re

def longest_dot_run(structure: str) -> int:
    """
    Returns the length of the longest continuous run of '.' characters in the structure string.
    If there are no '.' characters, returns 0.
    """
    if not isinstance(structure, str):
        return 0
    matches = re.findall(r'\.+', structure)
    return max((len(m) for m in matches), default=0)


def main():
    parser = argparse.ArgumentParser(description="Find the longest continuous string of periods in TwoDStructure column.")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output CSV file")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input)

    if "TwoDStructure" not in df.columns:
        raise ValueError("Input CSV must contain a 'TwoDStructure' column.")

    # Compute max continuous dot run
    df["LongestLoopSize"] = df["TwoDStructure"].apply(longest_dot_run)

    # Write output
    df.to_csv(args.output, index=False)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
