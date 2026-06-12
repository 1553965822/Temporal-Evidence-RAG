#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_rag.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measured dataset checks and local lightweight baselines.")
    parser.add_argument("--mode", choices=["measured"], default="measured")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    result = run_experiment(mode=args.mode, config_path=args.config, force_data=args.force_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
