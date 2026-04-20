#!/usr/bin/env python3

import pandas as pd
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILENAME = "GeneratedDatasetPython.jsonl"

def explore_jsonl(file_path: Path, max_char: int = 100):
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    with file_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            print(f"File {file_path.name} is empty.")
            return
        data = json.loads(first_line)

    s = pd.Series(data)

    def truncate_value(val):
        if isinstance(val, list):
            return str(val[:1])[:max_char], len(val) if val else []
        
        str_val = str(val)
        if len(str_val) > max_char:
            return str_val[:max_char] + "..."
        
        return val

    explored = s.apply(truncate_value)

    print(f"\n--- Exploring First Row of: {file_path.name} ---")
    print(explored.to_string())
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Explore the first row of a JSONL dataset.")
    
    # Adding the filename argument. 
    # Defaulting to your newly generated file, but allows override.
    parser.add_argument(
        "filename", 
        nargs="?", 
        default=FILENAME, 
        help="The name of the .jsonl file to explore (default: generatedDatasetPython.jsonl)"
    )

    args = parser.parse_args()
    
    # Resolve the path relative to the script's directory
    file_to_open = BASE_DIR / args.filename
    explore_jsonl(file_to_open)

if __name__ == "__main__":
    main()