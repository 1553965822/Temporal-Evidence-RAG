#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_rag.data_builder import build_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-aligned GLTRD/GrassRisk/CUADRisk datasets.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--force", action="store_true", help="Regenerate existing processed datasets.")
    args = parser.parse_args()
    built = build_all(args.config, force=args.force)
    for name, count in built.items():
        print(f"{name}: {count} samples")


if __name__ == "__main__":
    main()
