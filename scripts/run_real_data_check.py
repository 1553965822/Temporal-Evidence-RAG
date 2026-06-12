#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_rag.runner import binary_metrics
from paper_rag.tables import write_table


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    base = ROOT / "data/processed/GrassRiskReal"
    if not (base / "train.jsonl").exists():
        raise SystemExit("GrassRiskReal not found. Run scripts\\ingest_real_data.py first.")
    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")
    if not train or not test:
        raise SystemExit("GrassRiskReal split is empty.")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=3000),
        LogisticRegression(max_iter=300, class_weight="balanced"),
    )
    model.fit([row["clause_text"] for row in train], [row["label"] for row in train])
    pred = model.predict([row["clause_text"] for row in test])
    y = [row["label"] for row in test]
    metrics = binary_metrics(y, list(map(int, pred)))

    out_dir = ROOT / "outputs/real_data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ["GrassRiskReal", "Local TF-IDF + LogisticRegression", round(metrics["precision"], 2), round(metrics["recall"], 2), round(metrics["f1"], 2), len(train), len(test)]
    ]
    write_table(
        "grassrisk_real_local_baseline",
        "GrassRisk真实标注本地轻量基线检查",
        ["Dataset", "Model", "Precision/%", "Recall/%", "F1/%", "Train", "Test"],
        rows,
        out_dir,
    )
    summary = {
        "dataset": "GrassRiskReal",
        "train": len(train),
        "test": len(test),
        "metrics": metrics,
        "output_table": str(out_dir / "grassrisk_real_local_baseline.md"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
