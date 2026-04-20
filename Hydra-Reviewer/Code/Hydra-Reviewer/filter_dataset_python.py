#!/usr/bin/env python3

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "unseen_dataset.jsonl"
TARGET = BASE_DIR / "unseen_dataset_python.jsonl"
LANGUAGE = "python"


def main() -> None:
    kept = 0

    with SOURCE.open("r", encoding="utf-8") as src, TARGET.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("lang") != LANGUAGE:
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept > 100:
                break

    print(f"Wrote {kept} {LANGUAGE} items to {TARGET}")


if __name__ == "__main__":
    main()
