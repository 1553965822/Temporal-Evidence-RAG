#!/usr/bin/env python
from __future__ import annotations

import os
import webbrowser
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / "outputs" / "component_experiments").glob("*/results.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        report = candidates[0]
    else:
        report = root / "outputs" / "measured" / "tables" / "measured_local_baseline.md"
    if not report.exists():
        raise SystemExit("Report not found. Run a real experiment first, for example: python scripts/run_component_experiments.py --mode all")
    webbrowser.open(report.as_uri())
    print(report)


if __name__ == "__main__":
    main()
