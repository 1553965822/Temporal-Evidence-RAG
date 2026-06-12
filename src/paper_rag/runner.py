from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ensure_dir, load_yaml, project_path
from .data_builder import build_all, load_dataset
from .tables import export_dataset_stats_table, write_table


def run_measured_local_baseline(output_dir: Path) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.pipeline import make_pipeline

    rows = []
    for dataset_name in ["GrassRisk", "CUADRisk"]:
        train = load_dataset(dataset_name, "train")
        test = load_dataset(dataset_name, "test")
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000), LogisticRegression(max_iter=300))
        model.fit([x["clause_text"] for x in train], [int(x["label"]) for x in train])
        pred = model.predict([x["clause_text"] for x in test])
        y = [int(x["label"]) for x in test]
        rows.append(
            [
                dataset_name,
                "Local TF-IDF + LogisticRegression",
                round(precision_score(y, pred, zero_division=0) * 100, 2),
                round(recall_score(y, pred, zero_division=0) * 100, 2),
                round(f1_score(y, pred, zero_division=0) * 100, 2),
                len(train),
                len(test),
            ]
        )
    write_table(
        "measured_local_baseline",
        "真实本地轻量基线结果",
        ["Dataset", "Model", "Precision/%", "Recall/%", "F1/%", "Train", "Test"],
        rows,
        output_dir / "tables",
    )
    return {"measured_rows": len(rows)}


def run_experiment(
    mode: str = "measured",
    config_path: str | Path = "configs/experiment.yaml",
    force_data: bool = False,
) -> dict[str, Any]:
    if mode != "measured":
        raise ValueError("Only measured mode is supported. Use scripts/run_component_experiments.py for component experiments.")

    config = load_yaml(config_path)
    built = build_all(config_path, force=force_data)
    output_dir = ensure_dir(project_path("outputs", "measured"))
    table_dir = ensure_dir(output_dir / "tables")
    dataset_stats = export_dataset_stats_table(config, table_dir)
    result: dict[str, Any] = {"mode": mode, "datasets": built, "tables": len(dataset_stats)}
    result.update(run_measured_local_baseline(output_dir))

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
