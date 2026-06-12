#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets from a Hugging Face Dataset repo.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. owner/name.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    target = Path(args.local_dir)
    if not target.is_absolute():
        target = ROOT / target
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "raw/**",
            "processed/**",
            "README.md",
            "*.json",
            "*.jsonl",
            "*.md",
        ],
    )
    print(f"Downloaded dataset repo {args.repo_id} to {target}")


if __name__ == "__main__":
    main()
