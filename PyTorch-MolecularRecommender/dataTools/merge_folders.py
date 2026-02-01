import os
import shutil
import argparse

def merge_unique_files(folder1, folder2, output):
    """
    Combine all unique files by name from folder1 and folder2 into output.
    If two files have the same name, the one from folder1 is kept.
    """
    os.makedirs(output, exist_ok=True)
    copied_files = set()

    def copy_unique(src_folder):
        for root, _, files in os.walk(src_folder):
            for file in files:
                if file not in copied_files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output, file)
                    shutil.copy2(src_path, dst_path)
                    copied_files.add(file)

    copy_unique(folder1)
    copy_unique(folder2)

    print(f"✅ Merged unique files into: {output}")
    print(f"Total unique files: {len(copied_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two folders into one, keeping only unique filenames."
    )
    parser.add_argument("--folder1", help="Path to the first folder")
    parser.add_argument("--folder2", help="Path to the second folder")
    parser.add_argument("--output", help="Path to the output folder")

    args = parser.parse_args()
    merge_unique_files(args.folder1, args.folder2, args.output)
