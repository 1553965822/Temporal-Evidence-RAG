#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    env.update(os.environ)
    return env


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }


def tune_threshold(y_true: list[int], scores: np.ndarray, objective: str = "balanced") -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_score = -float("inf")
    true_positive_rate = float(sum(y_true) / len(y_true)) if y_true else 0.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (scores >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if objective == "balanced":
            tn = sum(1 for gold, got in zip(y_true, pred) if gold == 0 and got == 0)
            fp = sum(1 for gold, got in zip(y_true, pred) if gold == 0 and got == 1)
            specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
            balanced_accuracy = (recall + specificity) / 2
            pred_positive_rate = float(pred.sum() / len(pred)) if len(pred) else 0.0
            score = min(precision, recall) + 0.25 * f1 + 0.25 * balanced_accuracy - 0.20 * abs(pred_positive_rate - true_positive_rate)
        else:
            score = f1
        if score > best_score or (abs(score - best_score) < 1e-9 and f1 > best_f1):
            best_score = score
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def run_tfidf(train: list[dict], val: list[dict], test: list[dict], threshold_objective: str) -> dict:
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=12000),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )
    model.fit([row["clause_text"] for row in train], [row["label"] for row in train])
    val_scores = model.predict_proba([row["clause_text"] for row in val])[:, 1]
    threshold, val_f1 = tune_threshold([row["label"] for row in val], val_scores, threshold_objective)
    test_scores = model.predict_proba([row["clause_text"] for row in test])[:, 1]
    pred = (test_scores >= threshold).astype(int).tolist()
    result = metrics([row["label"] for row in test], pred)
    result.update({"threshold": threshold, "val_f1_at_threshold": val_f1 * 100})
    return {"model": "TF-IDF + LogisticRegression", "status": "ok", "metrics": result, "predictions": pred}


def mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_roberta(texts: list[str], model_path: str, batch_size: int = 8) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def run_roberta(train: list[dict], val: list[dict], test: list[dict], model_path: str | None, threshold_objective: str) -> dict | None:
    if not model_path or not Path(model_path).exists():
        print("RoBERTa model path missing, omitting RoBERTa row.")
        return None
    try:
        x_train = encode_roberta([row["clause_text"] for row in train], model_path)
        x_val = encode_roberta([row["clause_text"] for row in val], model_path)
        x_test = encode_roberta([row["clause_text"] for row in test], model_path)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(x_train, [row["label"] for row in train])
        val_scores = clf.predict_proba(x_val)[:, 1]
        threshold, val_f1 = tune_threshold([row["label"] for row in val], val_scores, threshold_objective)
        test_scores = clf.predict_proba(x_test)[:, 1]
        pred = (test_scores >= threshold).astype(int).tolist()
        result = metrics([row["label"] for row in test], pred)
        result.update({"threshold": threshold, "val_f1_at_threshold": val_f1 * 100})
        return {"model": "RoBERTa frozen embeddings + LogisticRegression", "status": "ok", "metrics": result, "predictions": pred}
    except Exception as exc:
        print(f"RoBERTa evaluation failed, omitting RoBERTa row: {type(exc).__name__}: {exc}")
        return None


def write_markdown(results: dict, path: Path) -> None:
    dataset_name = results["dataset"].get("name", "GrassRisk")
    lines = [f"# {dataset_name} Real Model Evaluation", ""]
    lines.append("| Model | Status | Accuracy/% | Precision/% | Recall/% | F1/% | Threshold |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in results["classification_results"]:
        values = item.get("metrics", {})
        lines.append(
            "| {model} | {status} | {accuracy} | {precision} | {recall} | {f1} | {threshold} |".format(
                model=item["model"],
                status=item["status"],
                accuracy=f"{values.get('accuracy', 0):.2f}",
                precision=f"{values.get('precision', 0):.2f}",
                recall=f"{values.get('recall', 0):.2f}",
                f1=f"{values.get('f1', 0):.2f}",
                threshold=f"{values.get('threshold', 0):.2f}",
            )
        )
    lines.append("")
    lines.append("## Dataset")
    for key, value in results["dataset"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real local baselines on a processed risk-review dataset.")
    parser.add_argument("--dataset", default="GrassRisk", help="Processed dataset folder under data/processed.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-objective", choices=["balanced", "f1"], default="balanced")
    args = parser.parse_args()

    env = load_env()
    base = ROOT / "data" / "processed" / args.dataset
    train = load_jsonl(base / "train.jsonl")
    val = load_jsonl(base / "val.jsonl")
    test = load_jsonl(base / "test.jsonl")
    classification_results = [
        run_tfidf(train, val, test, args.threshold_objective),
        run_roberta(train, val, test, env.get("ROBERTA_MODEL_PATH"), args.threshold_objective),
    ]
    results = {
        "dataset": {
            "name": args.dataset,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_positive": sum(int(row["label"]) for row in train),
            "val_positive": sum(int(row["label"]) for row in val),
            "test_positive": sum(int(row["label"]) for row in test),
        },
        "classification_results": [item for item in classification_results if item is not None],
    }
    out_dir = args.output_dir or (ROOT / "outputs" / "real_model_eval" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_model_eval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, out_dir / "real_model_eval_results.md")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
