#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from paper_rag.data_builder import make_laws
except Exception:
    make_laws = None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(value: float) -> float:
    return round(float(value) * 100.0, 2)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) >= 10:
        text = text[:10].replace("/", "-").replace(".", "-")
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def iso(value: str | None) -> str | None:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def normalize_date_parts(year: str, month: str = "1", day: str = "1") -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


DATE_PATTERNS = [
    re.compile(r"(20\d{2}|19\d{2})[-./](\d{1,2})[-./](\d{1,2})"),
    re.compile(r"(20\d{2}|19\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
    re.compile(r"(20\d{2}|19\d{2})\s*年\s*(\d{1,2})\s*月"),
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(19\d{2}|20\d{2})\b", re.I),
]


def extract_anchor_from_text(text: str) -> str | None:
    text = text or ""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        values = match.groups()
        if values[0].lower() in MONTHS:
            return normalize_date_parts(values[2], str(MONTHS[values[0].lower()]), values[1])
        if len(values) == 2:
            return normalize_date_parts(values[0], values[1], "1")
        return normalize_date_parts(values[0], values[1], values[2])
    return None


def extract_anchor(row: dict, from_text_only: bool = False) -> str | None:
    if from_text_only:
        return extract_anchor_from_text(str(row.get("clause_text", "")))
    for key in ("anchor_date", "time_anchor", "contract_effective_date", "contract_sign_date"):
        value = iso(row.get(key))
        if value:
            return value
    return extract_anchor_from_text(str(row.get("clause_text", "")))


def law_text(row: dict) -> str:
    return " ".join(
        [
            str(row.get("law_name", "")),
            str(row.get("article_no", "")),
            str(row.get("article_text", "")),
            str(row.get("article_summary", "")),
            str(row.get("law_key", "")),
            " ".join(str(tag) for tag in row.get("risk_tags", []) or []),
            " ".join(str(tag) for tag in row.get("risk_categories", []) or []),
        ]
    )


def law_is_active(row: dict | None, anchor_value: str | None) -> bool:
    if not row:
        return False
    anchor = parse_date(anchor_value)
    start = parse_date(row.get("valid_from") or row.get("t_start"))
    end = parse_date(row.get("valid_to") or row.get("t_end")) or date(9999, 12, 31)
    return bool(anchor and start and start <= anchor <= end)


def load_laws(dataset: str, law_kb: Path | None) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources: dict[str, int] = {}

    if dataset.lower().startswith("gltrd") and make_laws is not None:
        synthetic = make_laws()
        rows.extend(synthetic)
        sources["paper_rag.data_builder.make_laws"] = len(synthetic)

    default_kb = ROOT / "data/raw/laws/legal_validity_kb.jsonl"
    for path in [law_kb, default_kb]:
        if path and path.exists():
            loaded = load_jsonl(path)
            rows.extend(loaded)
            sources[str(path)] = len(loaded)

    dedup: dict[str, dict] = {}
    for row in rows:
        law_id = row.get("law_id")
        if law_id and law_id not in dedup:
            dedup[law_id] = row
    return list(dedup.values()), sources


class TemporalRetriever:
    def __init__(self, law_rows: list[dict], risk_boost: float = 0.45, active_boost: float = 0.20, inactive_penalty: float = 0.35):
        self.law_rows = law_rows
        self.risk_boost = risk_boost
        self.active_boost = active_boost
        self.inactive_penalty = inactive_penalty
        texts = [law_text(row) for row in law_rows]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, row: dict, top_k: int = 5, temporal: bool = False, filter_inactive: bool = False) -> list[dict]:
        query_text = " ".join([str(row.get("clause_text", "")), str(row.get("risk_type", "")), str(row.get("risk_category", ""))])
        query = self.vectorizer.transform([query_text])
        raw_scores = cosine_similarity(query, self.matrix)[0]
        scores = np.array(raw_scores, copy=True)
        risk_type = str(row.get("risk_type", "")).lower()
        risk_category = str(row.get("risk_category", "")).lower()
        anchor = extract_anchor(row)

        active_target_exists = bool(active_laws_for_row(row, self.law_rows))
        for idx, law in enumerate(self.law_rows):
            tags = [str(tag).lower() for tag in law.get("risk_tags", []) or []]
            categories = [str(tag).lower() for tag in law.get("risk_categories", []) or []]
            if risk_type and risk_type in tags:
                scores[idx] += self.risk_boost
            if risk_category and risk_category in categories:
                scores[idx] += 0.15
            if temporal:
                if law_is_active(law, anchor):
                    scores[idx] += self.active_boost
                elif active_target_exists:
                    scores[idx] -= self.inactive_penalty

        results: list[dict] = []
        for idx in np.argsort(scores)[::-1]:
            law = self.law_rows[int(idx)]
            if temporal and filter_inactive and active_target_exists and not law_is_active(law, anchor):
                continue
            item = dict(law)
            item["_score"] = round(float(scores[int(idx)]), 6)
            item["_text_score"] = round(float(raw_scores[int(idx)]), 6)
            item["_active_at_anchor"] = law_is_active(law, anchor)
            item["_anchor_date"] = anchor
            results.append(item)
            if len(results) >= top_k:
                break
        return results


def active_laws_for_row(row: dict, law_rows: list[dict]) -> list[dict]:
    anchor = extract_anchor(row)
    law_key = row.get("law_key")
    if not law_key:
        gold = row.get("gold_law_id")
        law_key = next((law.get("law_key") for law in law_rows if law.get("law_id") == gold), None)
    candidates = [law for law in law_rows if law_key and law.get("law_key") == law_key]
    return [law for law in candidates if law_is_active(law, anchor)]


def temporal_gold_ids(row: dict, law_rows: list[dict]) -> set[str]:
    active = active_laws_for_row(row, law_rows)
    if active:
        return {str(law["law_id"]) for law in active}
    ids = {str(row.get("gold_law_id", "")), str(row.get("cited_law_id", ""))}
    ids.update(str(x) for x in row.get("gold_evidence_ids", []) or [])
    return {x for x in ids if x}


def rank_of_gold(retrieved: list[dict], gold_ids: set[str]) -> int | None:
    for idx, row in enumerate(retrieved, 1):
        if str(row.get("law_id")) in gold_ids:
            return idx
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    return {
        "hit@1": round(sum(1 for r in ranks if r is not None and r <= 1) / n * 100, 2),
        "hit@3": round(sum(1 for r in ranks if r is not None and r <= 3) / n * 100, 2),
        "hit@5": round(sum(1 for r in ranks if r is not None and r <= 5) / n * 100, 2),
        "mrr": round(sum((1.0 / r) if r else 0.0 for r in ranks) / n, 4),
    }


def tune_threshold(y_true: list[int], scores: list[float], objective: str = "balanced") -> float:
    if not scores:
        return 0.5
    values = sorted(set(float(x) for x in scores))
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates.extend(values)
    candidates.extend([(a + b) / 2.0 for a, b in zip(values, values[1:])])
    true_rate = sum(y_true) / len(y_true) if y_true else 0.0
    best_score = -1e9
    best_threshold = candidates[0]
    for threshold in candidates:
        pred = [1 if score >= threshold else 0 for score in scores]
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        pred_rate = sum(pred) / len(pred) if pred else 0.0
        if objective == "f1":
            score = f1
        else:
            score = f1 + 0.20 * min(precision, recall) - 0.10 * abs(pred_rate - true_rate)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def temporal_score(row: dict, retriever: TemporalRetriever, method: str) -> float:
    laws_by_id = {law.get("law_id"): law for law in retriever.law_rows}
    cited = laws_by_id.get(row.get("cited_law_id"))
    anchor = extract_anchor(row)
    standard = retriever.retrieve(row, top_k=5, temporal=False, filter_inactive=False)
    temporal = retriever.retrieve(row, top_k=5, temporal=True, filter_inactive=True)
    top_std = standard[0] if standard else {}
    top_tmp = temporal[0] if temporal else {}
    active_ids = {law["law_id"] for law in active_laws_for_row(row, retriever.law_rows)}
    cited_active = law_is_active(cited, anchor)
    status_expired_now = bool(cited and parse_date(cited.get("valid_to") or cited.get("t_end")) and parse_date(cited.get("valid_to") or cited.get("t_end")) < date(2026, 6, 1))
    semantic_conflict = 0.0 if top_std.get("law_id") == row.get("cited_law_id") else 1.0
    low_similarity = max(0.0, 1.0 - float(top_std.get("_score", 0.0)))
    temporal_conflict = 1.0 if active_ids and row.get("cited_law_id") not in active_ids else 0.0
    top_temporal_conflict = 1.0 if top_tmp and top_tmp.get("law_id") != row.get("cited_law_id") else 0.0

    if method == "Standard-RAG":
        return 0.20 * semantic_conflict + 0.15 * low_similarity + 0.10 * float(status_expired_now)
    if method == "RAG + Temporal-KB":
        return 0.35 * float(status_expired_now) + 0.15 * temporal_conflict + 0.05 * low_similarity
    if method == "RAG + Temporal-KB + Anchor":
        return 0.55 * float(not cited_active) + 0.15 * temporal_conflict + 0.05 * low_similarity
    if method == "Temporal-RAG":
        return 0.50 * temporal_conflict + 0.25 * float(not cited_active) + 0.10 * top_temporal_conflict + 0.05 * max(0.0, 1.0 - float(top_tmp.get("_score", 0.0)))
    raise ValueError(method)


def evaluate_anchor(rows: list[dict], output_dir: Path) -> tuple[list[dict], dict]:
    records = []
    for row in rows:
        gold = iso(row.get("anchor_date"))
        pred = extract_anchor(row, from_text_only=True)
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "gold_anchor": gold,
                "pred_anchor": pred,
                "date_entity_hit": bool(pred),
                "date_normalization_hit": bool(pred and pred == gold),
                "anchor_selection_hit": bool(pred and pred == gold),
            }
        )
    total = len(records) or 1
    metrics = {
        "date_entity_accuracy": round(sum(r["date_entity_hit"] for r in records) / total * 100, 2),
        "date_normalization_accuracy": round(sum(r["date_normalization_hit"] for r in records) / total * 100, 2),
        "anchor_selection_accuracy": round(sum(r["anchor_selection_hit"] for r in records) / total * 100, 2),
        "total": len(records),
    }
    write_jsonl(output_dir / "anchor_predictions.jsonl", records)
    return records, metrics


def evaluate_retrieval(rows: list[dict], retriever: TemporalRetriever, output_dir: Path) -> tuple[list[dict], dict]:
    methods = {
        "Standard-RAG": {"temporal": False, "filter_inactive": False},
        "Temporal-RAG": {"temporal": True, "filter_inactive": True},
    }
    all_records = []
    metrics = {}
    for method, kwargs in methods.items():
        ranks: list[int | None] = []
        for row in rows:
            gold_ids = temporal_gold_ids(row, retriever.law_rows)
            retrieved = retriever.retrieve(row, top_k=5, **kwargs)
            rank = rank_of_gold(retrieved, gold_ids)
            ranks.append(rank)
            all_records.append(
                {
                    "sample_id": row.get("sample_id"),
                    "method": method,
                    "anchor_date": extract_anchor(row),
                    "gold_ids": sorted(gold_ids),
                    "rank": rank,
                    "retrieved_law_ids": [law.get("law_id") for law in retrieved],
                    "retrieved_active_at_anchor": [law.get("_active_at_anchor") for law in retrieved],
                    "retrieved_scores": [law.get("_score") for law in retrieved],
                }
            )
        metrics[method] = retrieval_metrics(ranks)
    write_jsonl(output_dir / "retrieval_predictions.jsonl", all_records)
    return all_records, metrics


def evaluate_temporal_alignment(val_rows: list[dict], test_rows: list[dict], retriever: TemporalRetriever, output_dir: Path) -> tuple[list[dict], dict]:
    methods = ["Standard-RAG", "RAG + Temporal-KB", "RAG + Temporal-KB + Anchor", "Temporal-RAG"]
    results = {}
    records = []
    y_val = [int(row.get("label", 0)) for row in val_rows]
    y_test = [int(row.get("label", 0)) for row in test_rows]
    for method in methods:
        val_scores = [temporal_score(row, retriever, method) for row in val_rows]
        threshold = tune_threshold(y_val, val_scores)
        test_scores = [temporal_score(row, retriever, method) for row in test_rows]
        pred = [1 if score >= threshold else 0 for score in test_scores]
        result = binary_metrics(y_test, pred)
        result["threshold"] = round(threshold, 6)
        result["predicted_positive"] = int(sum(pred))
        results[method] = result
        for row, score, got in zip(test_rows, test_scores, pred):
            records.append(
                {
                    "sample_id": row.get("sample_id"),
                    "method": method,
                    "label": int(row.get("label", 0)),
                    "prediction": int(got),
                    "score": round(float(score), 6),
                    "threshold": round(threshold, 6),
                    "anchor_date": extract_anchor(row),
                    "cited_law_id": row.get("cited_law_id"),
                    "gold_law_id": row.get("gold_law_id"),
                }
            )
    write_jsonl(output_dir / "temporal_alignment_predictions.jsonl", records)
    return records, results


def row_review_steps(row: dict, strategy: str, rng: random.Random | None = None) -> str:
    steps = row.get("review_steps") or {}
    if not isinstance(steps, dict):
        return str(steps)
    items = [(k, v) for k, v in steps.items() if v]
    if strategy == "none" or not items:
        return ""
    if strategy == "random":
        rng = rng or random.Random(42)
        picked = rng.sample(items, min(2, len(items)))
    elif strategy == "key":
        key_words = ("evidence", "legal", "judgement", "risk", "temporal", "alignment", "consequence")
        picked = [(k, v) for k, v in items if any(word in k.lower() for word in key_words)]
        picked = picked or items[:2]
    else:
        picked = items
    return " ".join(f"{k}: {v}" for k, v in picked)


def feature_text(row: dict, mode: str, rng: random.Random | None = None) -> str:
    parts = [str(row.get("clause_text", "")), str(row.get("risk_type", "")), str(row.get("risk_category", ""))]
    if mode in {"evidence", "step", "all_steps", "random_steps", "key_steps"}:
        parts.extend(
            [
                str(row.get("evidence_text", "")),
                " ".join(str(x) for x in row.get("gold_evidence_ids", []) or []),
                str(row.get("gold_legal_analysis", "")),
            ]
        )
    if mode == "step":
        parts.append(row_review_steps(row, "key", rng))
    elif mode == "all_steps":
        parts.append(row_review_steps(row, "all", rng))
    elif mode == "random_steps":
        parts.append(row_review_steps(row, "random", rng))
    elif mode == "key_steps":
        parts.append(row_review_steps(row, "key", rng))
    return " ".join(parts)


def stratified_subset(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row.get("label", 0))].append(row)
    subset: list[dict] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) * ratio))
        subset.extend(shuffled[:n])
    rng.shuffle(subset)
    return subset


def filter_near_duplicate_train(train_rows: list[dict], test_rows: list[dict], threshold: float) -> tuple[list[dict], dict]:
    if threshold <= 0 or threshold >= 1 or not train_rows or not test_rows:
        return train_rows, {"enabled": False, "threshold": threshold, "removed": 0, "kept": len(train_rows)}
    texts = [str(row.get("clause_text", "")) for row in train_rows + test_rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    train_matrix = matrix[: len(train_rows)]
    test_matrix = matrix[len(train_rows) :]
    max_similarity = cosine_similarity(train_matrix, test_matrix).max(axis=1)
    kept = [row for row, score in zip(train_rows, max_similarity) if float(score) < threshold]
    return kept, {
        "enabled": True,
        "threshold": threshold,
        "removed": len(train_rows) - len(kept),
        "kept": len(kept),
        "max_train_test_similarity_before": round(float(max(max_similarity)), 6) if len(max_similarity) else 0.0,
    }


def fit_predict_text(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], mode: str, seed: int) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    train_texts = [feature_text(row, mode, rng) for row in train_rows]
    val_texts = [feature_text(row, mode, rng) for row in val_rows]
    test_texts = [feature_text(row, mode, rng) for row in test_rows]
    y_train = [int(row.get("label", 0)) for row in train_rows]
    y_val = [int(row.get("label", 0)) for row in val_rows]
    y_test = [int(row.get("label", 0)) for row in test_rows]
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=20000),
        LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed),
    )
    model.fit(train_texts, y_train)
    val_scores = model.predict_proba(val_texts)[:, 1].tolist()
    threshold = tune_threshold(y_val, val_scores)
    test_scores = model.predict_proba(test_texts)[:, 1].tolist()
    pred = [1 if score >= threshold else 0 for score in test_scores]
    result = binary_metrics(y_test, pred)
    result["threshold"] = round(threshold, 6)
    records = [
        {
            "sample_id": row.get("sample_id"),
            "mode": mode,
            "label": int(row.get("label", 0)),
            "prediction": int(got),
            "score": round(float(score), 6),
            "threshold": round(threshold, 6),
        }
        for row, got, score in zip(test_rows, pred, test_scores)
    ]
    return result, records


def evaluate_low_resource(dataset: str, train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], ratios: list[float], seed: int, output_dir: Path) -> tuple[list[dict], list[dict]]:
    method_modes = {
        "Direct Only": "direct",
        "Evidence Template": "evidence",
        "Evidence + Key-Step Alignment": "step",
    }
    rows = []
    predictions = []
    for ratio in ratios:
        subset = stratified_subset(train_rows, ratio, seed + int(ratio * 1000))
        for method, mode in method_modes.items():
            metrics, records = fit_predict_text(subset, val_rows, test_rows, mode, seed)
            rows.append({"dataset": dataset, "train_ratio": ratio, "train_size": len(subset), "method": method, **metrics})
            for record in records:
                predictions.append({"dataset": dataset, "train_ratio": ratio, "method": method, **record})
    write_jsonl(output_dir / "low_resource_predictions.jsonl", predictions)
    return rows, predictions


def evaluate_step_alignment(dataset: str, train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], seed: int, output_dir: Path) -> tuple[list[dict], list[dict]]:
    strategies = {
        "Random-Step Distill": "random_steps",
        "All-Step Distill": "all_steps",
        "Evidence-Key-Step Align": "key_steps",
    }
    rows = []
    predictions = []
    for method, mode in strategies.items():
        metrics, records = fit_predict_text(train_rows, val_rows, test_rows, mode, seed)
        rows.append({"dataset": dataset, "method": method, **metrics})
        for record in records:
            predictions.append({"dataset": dataset, "method": method, **record})
    write_jsonl(output_dir / "step_alignment_predictions.jsonl", predictions)
    return rows, predictions


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_results_md(path: Path, summary: dict) -> None:
    lines = [
        "# Component Experiment Results",
        "",
        f"- Run mode: {summary['mode']}",
        f"- Created at: {summary['created_at']}",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Note: these are measured component experiments over local datasets. They recompute predictions from the current data and code.",
        "",
        "## Source Project Reuse",
        "",
        md_table(["Archive", "Reusable parts", "Action"], summary["source_reuse_rows"]),
        "",
    ]
    if "gltrd" in summary:
        g = summary["gltrd"]
        lines.extend(
            [
                "## GLTRD Data",
                "",
                md_table(
                    ["Split", "Rows", "Label 0", "Label 1"],
                    [[k, v["rows"], v["label_0"], v["label_1"]] for k, v in g["data_stats"].items()],
                ),
                "",
                "## Table 8 - Legal Temporal Alignment",
                "",
                md_table(
                    ["Method", "Accuracy", "Precision", "Recall", "F1", "Threshold", "Pred Pos"],
                    [
                        [
                            method,
                            m["accuracy"],
                            m["precision"],
                            m["recall"],
                            m["f1"],
                            m["threshold"],
                            m["predicted_positive"],
                        ]
                        for method, m in g["temporal_alignment"].items()
                    ],
                ),
                "",
                "## Table 9 - Contract Time Anchor Extraction",
                "",
                md_table(
                    ["Metric", "Value"],
                    [
                        ["Date entity accuracy", g["anchor_extraction"]["date_entity_accuracy"]],
                        ["Date normalization accuracy", g["anchor_extraction"]["date_normalization_accuracy"]],
                        ["Anchor selection accuracy", g["anchor_extraction"]["anchor_selection_accuracy"]],
                        ["Samples", g["anchor_extraction"]["total"]],
                    ],
                ),
                "",
                "## Table 10 - Retrieval Evidence Hit",
                "",
                md_table(
                    ["Method", "Hit@1", "Hit@3", "Hit@5", "MRR"],
                    [[method, m["hit@1"], m["hit@3"], m["hit@5"], m["mrr"]] for method, m in g["retrieval_hit"].items()],
                ),
                "",
                "## Table 11 - Temporal Mechanism Ablation",
                "",
                md_table(
                    ["Component", "Accuracy", "Precision", "Recall", "F1"],
                    [
                        [method, m["accuracy"], m["precision"], m["recall"], m["f1"]]
                        for method, m in g["temporal_ablation"].items()
                    ],
                ),
                "",
                "## GLTRD Law KB Sources",
                "",
                md_table(["Source", "Rows"], [[k, v] for k, v in g["law_sources"].items()]),
                "",
                f"- Gold-active consistency warnings: {g['gold_active_warnings']}",
                "",
            ]
        )
    if "low_resource" in summary:
        rows = summary["low_resource"]
        filt = summary.get("evidence_dataset_filter", {})
        lines.extend(
            [
                "## Figure 3 - Low-Resource Evidence Utilization",
                "",
                f"- Near-duplicate train/test filter: enabled={filt.get('enabled')}, threshold={filt.get('threshold')}, kept={filt.get('kept')}, removed={filt.get('removed')}",
                "",
                md_table(
                    ["Dataset", "Ratio", "Train", "Method", "Accuracy", "Precision", "Recall", "F1"],
                    [
                        [r["dataset"], r["train_ratio"], r["train_size"], r["method"], r["accuracy"], r["precision"], r["recall"], r["f1"]]
                        for r in rows
                    ],
                ),
                "",
            ]
        )
    if "step_alignment" in summary:
        rows = summary["step_alignment"]
        lines.extend(
            [
                "## Table 12 - Key-Step Alignment Strategy",
                "",
                md_table(
                    ["Dataset", "Method", "Accuracy", "Precision", "Recall", "F1"],
                    [[r["dataset"], r["method"], r["accuracy"], r["precision"], r["recall"], r["f1"]] for r in rows],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def data_stats(rows_by_split: dict[str, list[dict]]) -> dict[str, dict]:
    stats = {}
    for split, rows in rows_by_split.items():
        counter = Counter(int(row.get("label", 0)) for row in rows)
        stats[split] = {"rows": len(rows), "label_0": counter.get(0, 0), "label_1": counter.get(1, 0)}
    return stats


def gltrd_gold_warnings(rows: list[dict], law_rows: list[dict]) -> dict[str, int]:
    laws_by_id = {law.get("law_id"): law for law in law_rows}
    warnings = Counter()
    for row in rows:
        gold = laws_by_id.get(row.get("gold_law_id"))
        cited = laws_by_id.get(row.get("cited_law_id"))
        anchor = extract_anchor(row)
        if gold and not law_is_active(gold, anchor):
            warnings["gold_law_not_active_at_anchor"] += 1
        if cited and not law_is_active(cited, anchor):
            warnings["cited_law_not_active_at_anchor"] += 1
        if not gold:
            warnings["missing_gold_law_in_kb"] += 1
        if not cited:
            warnings["missing_cited_law_in_kb"] += 1
    return dict(warnings)


def run_gltrd(args, output_dir: Path, summary: dict) -> None:
    rows_by_split = {
        "train": load_jsonl(ROOT / "data/processed/GLTRD/train.jsonl"),
        "val": load_jsonl(ROOT / "data/processed/GLTRD/val.jsonl"),
        "test": load_jsonl(ROOT / "data/processed/GLTRD/test.jsonl"),
    }
    law_rows, law_sources = load_laws("GLTRD", Path(args.law_kb) if args.law_kb else None)
    retriever = TemporalRetriever(law_rows, risk_boost=args.risk_boost, active_boost=args.active_boost, inactive_penalty=args.inactive_penalty)

    test_rows = rows_by_split["test"]
    anchor_records, anchor_metrics = evaluate_anchor(test_rows, output_dir)
    retrieval_records, retrieval = evaluate_retrieval(test_rows, retriever, output_dir)
    temporal_records, temporal = evaluate_temporal_alignment(rows_by_split["val"], test_rows, retriever, output_dir)

    ablation_names = {
        "Standard-RAG": "Standard-RAG",
        "RAG + Temporal-KB": "+ Legal Validity Cycle",
        "RAG + Temporal-KB + Anchor": "+ Validity Cycle + Anchor",
        "Temporal-RAG": "+ Temporal-Constrained Retrieval",
    }
    ablation = {ablation_names[k]: v for k, v in temporal.items()}

    write_jsonl(output_dir / "law_kb_used.jsonl", law_rows)
    summary["gltrd"] = {
        "data_stats": data_stats(rows_by_split),
        "law_sources": law_sources,
        "anchor_extraction": anchor_metrics,
        "retrieval_hit": retrieval,
        "temporal_alignment": temporal,
        "temporal_ablation": ablation,
        "gold_active_warnings": gltrd_gold_warnings(test_rows, law_rows),
        "prediction_files": {
            "anchor": str(output_dir / "anchor_predictions.jsonl"),
            "retrieval": str(output_dir / "retrieval_predictions.jsonl"),
            "temporal_alignment": str(output_dir / "temporal_alignment_predictions.jsonl"),
        },
        "record_counts": {
            "anchor_predictions": len(anchor_records),
            "retrieval_predictions": len(retrieval_records),
            "temporal_alignment_predictions": len(temporal_records),
        },
    }


def run_evidence_components(args, output_dir: Path, summary: dict) -> None:
    dataset = args.evidence_dataset
    train = load_jsonl(ROOT / "data/processed" / dataset / "train.jsonl")
    val = load_jsonl(ROOT / "data/processed" / dataset / "val.jsonl")
    test = load_jsonl(ROOT / "data/processed" / dataset / "test.jsonl")
    if not train or not val or not test:
        summary["evidence_component_error"] = f"Missing split files for {dataset}"
        return
    train, filter_stats = filter_near_duplicate_train(train, test, args.near_duplicate_threshold)
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    low_rows, _ = evaluate_low_resource(dataset, train, val, test, ratios, args.seed, output_dir)
    step_rows, _ = evaluate_step_alignment(dataset, train, val, test, args.seed, output_dir)
    summary["evidence_dataset_filter"] = filter_stats
    summary["low_resource"] = low_rows
    summary["step_alignment"] = step_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measured component experiments for Temporal-RAG and Evidence-RAG.")
    parser.add_argument("--mode", choices=["gltrd", "evidence", "all"], default="all")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--law-kb", default="")
    parser.add_argument("--evidence-dataset", default="GrassRisk", choices=["GrassRisk", "CUADRisk"])
    parser.add_argument("--ratios", default="0.1,0.2,0.4,0.6,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-boost", type=float, default=0.45)
    parser.add_argument("--active-boost", type=float, default=0.20)
    parser.add_argument("--inactive-penalty", type=float, default=0.35)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    args = parser.parse_args()

    stamp = args.output_suffix or time.strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs/component_experiments" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
        "parameters": vars(args),
        "source_reuse_rows": [
            ["grassland_contract_project.zip", "TAE, Temporal-KB, Temporal-RAG, retrieval metrics, temporal ablation", "Reimplemented as runnable measured GLTRD components because copied source has encoding damage."],
            ["HKAD(1).tar.gz", "Evidence-aware review template, Full-Distill, key-step alignment, low-resource configs", "Reused as feature design for evidence utilization and key-step alignment component experiments."],
        ],
    }

    if args.mode in {"gltrd", "all"}:
        run_gltrd(args, output_dir, summary)
    if args.mode in {"evidence", "all"}:
        run_evidence_components(args, output_dir, summary)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_md(output_dir / "results.md", summary)
    print(json.dumps({"output_dir": str(output_dir), "results": str(output_dir / "results.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
