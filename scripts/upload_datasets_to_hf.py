#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDES = [
    ".DS_Store",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "user_uploads/**",
    "user_risk_uploads/**",
    "laws/source_docs/**",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local project datasets to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. owner/name.")
    parser.add_argument("--source-dir", default=str(ROOT / "data"))
    parser.add_argument("--private", action="store_true", help="Create or keep the dataset repo private.")
    parser.add_argument("--commit-message", default="Upload contract review RAG datasets")
    args = parser.parse_args()

    source = Path(args.source_dir)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(source),
        path_in_repo=".",
        ignore_patterns=DEFAULT_EXCLUDES,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {source} to dataset repo {args.repo_id}")


if __name__ == "__main__":
    main()
