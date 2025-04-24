#!/usr/bin/env python3
"""
delete_duplicates.py

Script to delete duplicate files in a fixed working directory based on numeric prefix.
Keeps only the file with the smallest numeric prefix for each basename.
"""
import sys
from pathlib import Path

# Configuration: set the directory to process
WORKING_DIR = Path(__file__).parent.parent / 'data' / 'cnn_english' / 'article_download_bak4'


def delete_duplicates(directory: Path):
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Gather all files and group by basename (filename without numeric prefix)
    duplicates = {}
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        filename = file_path.name
        if '_' not in filename:
            continue
        prefix, rest = filename.split('_', 1)
        try:
            num = int(prefix)
        except ValueError:
            continue
        duplicates.setdefault(rest, []).append((num, file_path))

    # For each group, keep only the lowest-prefix file and delete others
    for basename, entries in duplicates.items():
        entries.sort(key=lambda x: x[0])  # sort by numeric prefix
        keep_num, keep_path = entries[0]
        for num, path in entries[1:]:
            try:
                path.unlink()
                print(f"Deleted {path.name} (duplicate of {keep_path.name})")
            except Exception as e:
                print(f"Failed to delete {path.name}: {e}")


def main():
    print(f"Processing directory: {WORKING_DIR}")
    delete_duplicates(WORKING_DIR)


if __name__ == "__main__":
    main()
