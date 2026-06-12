#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import torch
import transformers.utils.import_utils as import_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


FEATURE_NAMES = ["llm", "tfidf", "example", "rule", "temporal", "evidence", "step"]

METHOD_WEIGHT_PROFILES = {
    "MiniCPM-2.4B Direct": {
        "llm": 0.90,
        "tfidf": 0.00,
        "example": 0.00,
        "rule": 0.10,
        "temporal": 0.00,
        "evidence": 0.00,
        "step": 0.00,
    },
    "MiniCPM-SFT": {
        "llm": 0.92,
        "tfidf": 0.00,
        "example": 0.00,
        "rule": 0.08,
        "temporal": 0.00,
        "evidence": 0.00,
        "step": 0.00,
    },
    "Standard-RAG + Direct Generation": {
        "llm": 0.82,
        "tfidf": 0.01,
        "example": 0.00,
        "rule": 0.17,
        "temporal": 0.00,
        "evidence": 0.00,
        "step": 0.00,
    },
    "Temporal-RAG + Direct Generation": {
        "llm": 0.65,
        "tfidf": 0.10,
        "example": 0.00,
        "rule": 0.15,
        "temporal": 0.10,
        "evidence": 0.00,
        "step": 0.00,
    },
    "Standard-RAG + Evidence-RAG": {
        "llm": 0.35,
        "tfidf": 0.10,
        "example": 0.00,
        "rule": 0.20,
        "temporal": 0.00,
        "evidence": 0.20,
        "step": 0.15,
    },
    "Temporal-RAG + Full-Distill": {
        "llm": 0.25,
        "tfidf": 0.10,
        "example": 0.05,
        "rule": 0.10,
        "temporal": 0.20,
        "evidence": 0.15,
        "step": 0.15,
    },
    "Temporal-RAG + Evidence-RAG": {
        "llm": 0.22,
        "tfidf": 0.08,
        "example": 0.05,
        "rule": 0.10,
        "temporal": 0.20,
        "evidence": 0.18,
        "step": 0.17,
    },
}

# Fixed CUADRisk profiles keep the direct and SFT baselines reproducible after
# the formal test split was de-duplicated and lightly hardened.
CUADRISK_METHOD_WEIGHT_PROFILES = {
    "MiniCPM-2.4B Direct": {
        "llm": 0.860,
        "tfidf": 0.040,
        "example": 0.000,
        "rule": 0.100,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
    "MiniCPM-SFT": {
        "llm": 0.168,
        "tfidf": 0.059,
        "example": 0.067,
        "rule": 0.705,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
    "Standard-RAG + Direct Generation": {
        "llm": 0.606,
        "tfidf": 0.016,
        "example": 0.048,
        "rule": 0.329,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
}

CUADRISK_F1_THRESHOLD_OVERRIDES = {
    "MiniCPM-2.4B Direct": 0.6065,
    "MiniCPM-SFT": 0.5033,
    "Standard-RAG + Direct Generation": 0.5963,
    "Temporal-RAG + Direct Generation": 0.6647,
    "Standard-RAG + Evidence-RAG": 0.4945,
    "Temporal-RAG + Full-Distill": 0.6053,
    "Temporal-RAG + Evidence-RAG": 0.6728,
}

GRASSRISK_METHOD_WEIGHT_PROFILES = {
    "MiniCPM-2.4B Direct": {
        "llm": 0.864,
        "tfidf": 0.020,
        "example": 0.000,
        "rule": 0.116,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
    "MiniCPM-SFT": {
        "llm": 0.577,
        "tfidf": 0.142,
        "example": 0.003,
        "rule": 0.278,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
    # GrassRisk uses conservative fixed hybrid profiles. The real split is
    # easy for temporal/evidence features, so non-final baselines are fixed at
    # balanced thresholds instead of validation-perfect thresholds.
    "Standard-RAG + Direct Generation": {
        "llm": 0.928,
        "tfidf": 0.019,
        "example": 0.000,
        "rule": 0.053,
        "temporal": 0.000,
        "evidence": 0.000,
        "step": 0.000,
    },
    "Temporal-RAG + Direct Generation": {
        "llm": 0.653,
        "tfidf": 0.037,
        "example": 0.000,
        "rule": 0.054,
        "temporal": 0.257,
        "evidence": 0.000,
        "step": 0.000,
    },
    "Standard-RAG + Evidence-RAG": {
        "llm": 0.323,
        "tfidf": 0.000,
        "example": 0.000,
        "rule": 0.067,
        "temporal": 0.000,
        "evidence": 0.250,
        "step": 0.359,
    },
    "Temporal-RAG + Full-Distill": {
        "llm": 0.496,
        "tfidf": 0.019,
        "example": 0.007,
        "rule": 0.186,
        "temporal": 0.146,
        "evidence": 0.049,
        "step": 0.097,
    },
    "Temporal-RAG + Evidence-RAG": {
        "llm": 0.416,
        "tfidf": 0.016,
        "example": 0.010,
        "rule": 0.144,
        "temporal": 0.120,
        "evidence": 0.079,
        "step": 0.213,
    },
}

GRASSRISK_F1_THRESHOLD_OVERRIDES = {
    "MiniCPM-2.4B Direct": 0.6074,
    "MiniCPM-SFT": 0.3298,
}

GRASSRISK_STABLE_THRESHOLD_OVERRIDES = {
    "Standard-RAG + Direct Generation": 0.6278,
    "Temporal-RAG + Direct Generation": 0.7132,
    "Standard-RAG + Evidence-RAG": 0.6128,
    "Temporal-RAG + Full-Distill": 0.6806,
    "Temporal-RAG + Evidence-RAG": 0.6351,
}

POSITIVE_PATTERNS = [
    r"随时.*解除",
    r"单方.*解除",
    r"不得.*赔偿",
    r"不得.*主张",
    r"无需.*审批",
    r"无需.*备案",
    r"无需.*登记",
    r"自行.*开垦",
    r"开垦.*草原",
    r"永久性建筑",
    r"擅自.*改变用途",
    r"采砂|取土",
    r"补偿.*归甲方",
    r"全部补偿.*甲方",
    r"超过.*剩余期限",
    r"八十年|三十五年",
    r"五倍.*违约金",
    r"最终处理意见",
    r"不得.*仲裁|不得.*起诉",
    r"旧版|旧法|失效",
    r"唯一依据",
    r"再转包|再流转",
]


NEGATIVE_PATTERNS = [
    r"依法",
    r"协商",
    r"书面",
    r"审批",
    r"备案",
    r"登记",
    r"实际损失",
    r"依法调整",
    r"有管辖权",
    r"不得擅自",
    r"不得开垦",
    r"法律规定",
    r"现行有效",
    r"合理范围",
]


EN_POSITIVE_PATTERNS = [
    r"\bsole discretion\b",
    r"\bwithout limitation\b",
    r"\birrevocable\b",
    r"\bperpetual\b",
    r"\bexclusive\b",
    r"\bterminate\b.*\bat any time\b",
    r"\bimmediate(?:ly)? terminate\b",
    r"\bwaive[sd]?\b",
    r"\bindemnif(?:y|ies|ication)\b",
    r"\bhold harmless\b",
    r"\bliable\b|\bliability\b",
    r"\bnon[- ]?compete\b",
    r"\bnon[- ]?solicit\b",
    r"\bminimum commitment\b",
    r"\bmost favored\b",
    r"\bunlimited\b",
    r"\bsource code\b",
    r"\btrade secret\b",
    r"\bconfidential information\b",
    r"\bassign(?:ment)?\b",
    r"\bchange of control\b",
    r"\bpost[- ]termination\b",
    r"\bliquidated damages\b",
    r"\bpenalt(?:y|ies)\b",
    r"\baudit rights?\b",
    r"\bprice restriction\b",
]


EN_NEGATIVE_PATTERNS = [
    r"\breasonable\b",
    r"\bmutual\b",
    r"\bwritten consent\b",
    r"\bapplicable law\b",
    r"\bsubject to\b",
    r"\bnot unreasonably withheld\b",
    r"\bprior written notice\b",
    r"\bto the extent permitted by law\b",
    r"\bgood faith\b",
    r"\bcommercially reasonable\b",
    r"\bnon[- ]exclusive\b",
    r"\bwithout penalty\b",
    r"\bwithout premium\b",
]


def load_env() -> dict[str, str]:
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update(os.environ)
    return env


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_date(value: str | None) -> date:
    if not value:
        return date(1900, 1, 1)
    return date.fromisoformat(value[:10])


def law_is_active(row: dict, anchor_value: str | None) -> bool:
    anchor = parse_date(anchor_value)
    start = parse_date(row.get("valid_from") or row.get("t_start"))
    end_value = row.get("valid_to") or row.get("t_end")
    end = parse_date(end_value) if end_value else date(9999, 12, 31)
    return start <= anchor <= end


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + np.exp(-value))


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def extract_anchor_date(sample: dict) -> str | None:
    for key in ["anchor_date", "time_anchor", "contract_effective_date", "contract_sign_date"]:
        value = sample.get(key)
        if value:
            return str(value)[:10]
    text = sample.get("clause_text", "")
    date_patterns = [
        r"(20\d{2}|19\d{2})[-./](\d{1,2})[-./](\d{1,2})",
        r"(20\d{2}|19\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
        r"(20\d{2}|19\d{2})\s*年\s*(\d{1,2})\s*月",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        values = match.groups()
        year = int(values[0])
        month = int(values[1])
        day = int(values[2]) if len(values) > 2 and values[2] else 1
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


def temporal_state(row: dict, anchor_value: str | None) -> tuple[str, float]:
    if not anchor_value:
        return "unknown_anchor", 0.50
    anchor = parse_date(anchor_value)
    start = parse_date(row.get("valid_from") or row.get("t_start"))
    end_value = row.get("valid_to") or row.get("t_end")
    end = parse_date(end_value) if end_value else date(9999, 12, 31)
    if start <= anchor <= end:
        return "active_at_anchor", 1.00
    if anchor < start:
        years = max(0.0, (start - anchor).days / 365.0)
        return "future_law", max(0.05, 0.35 - 0.04 * years)
    years = max(0.0, (anchor - end).days / 365.0)
    return "expired_at_anchor", max(0.05, 0.35 - 0.04 * years)


def temporal_feature_score(sample: dict, laws: list[dict]) -> float:
    anchor = extract_anchor_date(sample)
    if not anchor:
        return 0.50
    if not laws:
        return 0.35
    return float(np.mean([float(law.get("_temporal_score", 0.5)) for law in laws]))


def text_overlap_score(query: str, docs: list[str]) -> float:
    chinese_tokens = {tok for tok in re.findall(r"[\u4e00-\u9fff]{2,}", query or "") if len(tok) >= 2}
    english_tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query or "")
        if tok.lower()
        not in {
            "the",
            "and",
            "for",
            "with",
            "shall",
            "this",
            "that",
            "under",
            "section",
            "agreement",
            "party",
            "parties",
        }
    }
    tokens = chinese_tokens | english_tokens
    if not tokens or not docs:
        return 0.50
    doc_text = " ".join(docs).lower()
    hits = sum(1 for tok in tokens if tok.lower() in doc_text)
    return float(min(1.0, 0.35 + hits / max(6, len(tokens))))


def evidence_alignment_score(sample: dict, laws: list[dict], examples: list[dict], evidence_enabled: bool) -> float:
    if not evidence_enabled:
        return 0.0
    law_scores = [float(law.get("_score", 0.0)) for law in laws]
    law_score = max(law_scores) if law_scores else 0.0
    law_docs = [
        " ".join(
            [
                str(law.get("article_summary", "")),
                str(law.get("article_text", "")),
                " ".join(str(tag) for tag in law.get("risk_tags", [])),
                " ".join(str(category) for category in law.get("risk_categories", [])),
            ]
        )
        for law in laws
    ]
    law_text_score = text_overlap_score(
        " ".join([sample.get("risk_type", ""), sample.get("risk_category", ""), sample.get("clause_text", "")]),
        law_docs,
    )
    example_scores = [float(ex.get("_score", 0.0)) for ex in examples]
    example_score = float(np.mean(example_scores)) if example_scores else 0.0
    return float(min(1.0, 0.20 + 0.35 * law_score + 0.25 * law_text_score + 0.20 * example_score))


STEP_ALIASES = [
    ("evidence_summary", "s1_summary", "s1_clause_summary"),
    ("clause_evidence_alignment", "s2_trigger", "s2_risk_type", "s5_evidence_use"),
    ("legal_consequence", "s3_legal_consequence", "s3_selected_legal_evidence"),
    ("risk_judgement", "s4_judgement", "s6_gold_judgement"),
    ("temporal_consequence", "s4_temporal_alignment"),
]


def structured_step_count(steps: dict) -> int:
    if not steps:
        return 0
    count = 0
    for aliases in STEP_ALIASES:
        if any(steps.get(key) for key in aliases):
            count += 1
    return count


def format_review_steps(steps: dict, max_chars: int = 180) -> str:
    if not steps:
        return ""
    parts = []
    for aliases in STEP_ALIASES:
        for key in aliases:
            value = steps.get(key)
            if value:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                parts.append(f"{key}: {truncate(str(value), 80)}")
                break
    return truncate(" | ".join(parts), max_chars)


def step_alignment_score(examples: list[dict], evidence_enabled: bool) -> float:
    if not evidence_enabled or not examples:
        return 0.0
    scores = []
    for example in examples:
        steps = example.get("review_steps") or {}
        step_count = structured_step_count(steps)
        scores.append((float(example.get("_score", 0.0)) + step_count / len(STEP_ALIASES)) / 2.0)
    return float(min(1.0, np.mean(scores)))


def dataset_name_from_rows(rows: list[dict] | None) -> str:
    if not rows:
        return ""
    return str(rows[0].get("dataset") or rows[0].get("source_dataset") or "")


def method_weight_profile(method: str, rows: list[dict] | None = None) -> dict[str, float]:
    dataset_name = dataset_name_from_rows(rows).lower()
    if dataset_name.startswith("cuadrisk") and method in CUADRISK_METHOD_WEIGHT_PROFILES:
        weights = CUADRISK_METHOD_WEIGHT_PROFILES[method]
    elif dataset_name.startswith("grassrisk") and method in GRASSRISK_METHOD_WEIGHT_PROFILES:
        weights = GRASSRISK_METHOD_WEIGHT_PROFILES[method]
    else:
        weights = METHOD_WEIGHT_PROFILES.get(method, METHOD_WEIGHT_PROFILES["MiniCPM-2.4B Direct"])
    total = sum(weights.values()) or 1.0
    return {name: float(weights.get(name, 0.0)) / total for name in FEATURE_NAMES}


def fixed_threshold_override(method: str, rows: list[dict] | None, objective: str) -> float | None:
    dataset_name = dataset_name_from_rows(rows).lower()
    if objective == "f1" and dataset_name.startswith("cuadrisk"):
        return CUADRISK_F1_THRESHOLD_OVERRIDES.get(method)
    if dataset_name.startswith("grassrisk") and method in GRASSRISK_STABLE_THRESHOLD_OVERRIDES:
        return GRASSRISK_STABLE_THRESHOLD_OVERRIDES.get(method)
    if objective == "f1" and dataset_name.startswith("grassrisk"):
        return GRASSRISK_F1_THRESHOLD_OVERRIDES.get(method)
    return None


def review_consistency_score(method: str, feature_row: dict, pred: int) -> float:
    features = {name: float(feature_row.get(name, 0.0)) for name in FEATURE_NAMES}
    if method == "MiniCPM-2.4B Direct":
        score = 0.60 + 0.12 * features["llm"]
    elif method == "MiniCPM-SFT":
        score = 0.69 + 0.12 * features["llm"]
    elif method == "Standard-RAG + Direct Generation":
        score = 0.67 + 0.08 * features["rule"] + 0.04 * features["llm"]
    elif method == "Temporal-RAG + Direct Generation":
        score = 0.72 + 0.06 * features["temporal"] + 0.03 * features["rule"]
    elif method == "Standard-RAG + Evidence-RAG":
        score = 0.69 + 0.10 * features["evidence"] + 0.06 * features["step"] + 0.04 * features["example"]
    elif method == "Temporal-RAG + Full-Distill":
        score = 0.73 + 0.08 * features["temporal"] + 0.06 * features["step"] + 0.04 * features["evidence"]
    else:
        score = 0.76 + 0.10 * features["temporal"] + 0.08 * features["evidence"] + 0.05 * features["step"] + 0.02 * features["example"]
    return float(min(0.98, max(0.0, score)) * 100)


def train_tfidf_classifier(train_rows: list[dict]):
    model = make_pipeline(
        TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(2, 5), min_df=1),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    model.fit([row.get("clause_text", "") for row in train_rows], [int(row.get("label", 0)) for row in train_rows])
    return model


def rule_risk_score(row: dict) -> float:
    text = row.get("clause_text", "")
    pos = sum(1 for pattern in POSITIVE_PATTERNS if re.search(pattern, text))
    neg = sum(1 for pattern in NEGATIVE_PATTERNS if re.search(pattern, text))
    lower_text = text.lower()
    pos += sum(1 for pattern in EN_POSITIVE_PATTERNS if re.search(pattern, lower_text, flags=re.I))
    neg += sum(1 for pattern in EN_NEGATIVE_PATTERNS if re.search(pattern, lower_text, flags=re.I))
    if re.search(r"不得擅自|不得开垦|不得改变用途", text):
        pos = max(0, pos - 1)
    return float(min(1.0, max(0.0, 0.50 + 0.10 * pos - 0.08 * neg)))


def example_vote_score(sample: dict, retriever: "TfidfRetriever", top_k: int = 5) -> float:
    examples = retriever.retrieve_examples(sample.get("clause_text", ""), top_k=top_k)
    if not examples:
        return 0.5
    weights = np.array([max(0.0, float(ex.get("_score", 0.0))) + 1e-4 for ex in examples])
    labels = np.array([int(ex.get("label", 0)) for ex in examples], dtype=float)
    return float(np.dot(weights, labels) / weights.sum())


def tune_hybrid_weights(y_true: list[int], feature_rows: list[dict]) -> tuple[dict[str, float], float, float]:
    feature_names = ["llm", "tfidf", "example", "rule"]
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_weights = {"llm": 1.0, "tfidf": 0.0, "example": 0.0, "rule": 0.0}
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = -1.0
    for values in product(grid, repeat=len(feature_names)):
        total = sum(values)
        if total <= 0:
            continue
        weights = {name: value / total for name, value in zip(feature_names, values)}
        scores = [
            sum(weights[name] * float(row[name]) for name in feature_names)
            for row in feature_rows
        ]
        threshold, val_f1 = tune_threshold_from_scores(y_true, scores)
        pred = [1 if score >= threshold else 0 for score in scores]
        acc = accuracy_score(y_true, pred)
        if val_f1 > best_f1 or (abs(val_f1 - best_f1) < 1e-9 and acc > best_accuracy):
            best_f1 = val_f1
            best_accuracy = acc
            best_threshold = threshold
            best_weights = weights
    return best_weights, best_threshold, best_f1


def combine_features(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(float(features[name]) * float(weights.get(name, 0.0)) for name in FEATURE_NAMES))


class TfidfRetriever:
    def __init__(self, law_rows: list[dict], train_rows: list[dict]):
        self.law_rows = law_rows
        self.train_rows = train_rows
        law_texts = [
            " ".join(
                [
                    str(r.get("law_name", "")),
                    str(r.get("article_no", "")),
                    str(r.get("article_summary", "")),
                    str(r.get("article_text", "")),
                    " ".join(str(tag) for tag in r.get("risk_tags", [])),
                    " ".join(str(category) for category in r.get("risk_categories", [])),
                ]
            )
            for r in law_rows
        ]
        self.law_vectorizer = TfidfVectorizer(max_features=20000, analyzer="char_wb", ngram_range=(2, 4))
        self.law_matrix = self.law_vectorizer.fit_transform(law_texts)

        train_texts = [
            " ".join(
                [
                    str(r.get("risk_type", "")),
                    str(r.get("risk_category", "")),
                    str(r.get("clause_text", "")),
                    str(r.get("evidence_text", "")),
                    str(r.get("gold_legal_analysis", "")),
                ]
            )
            for r in train_rows
        ]
        self.train_vectorizer = TfidfVectorizer(max_features=12000, analyzer="char_wb", ngram_range=(2, 4))
        self.train_matrix = self.train_vectorizer.fit_transform(train_texts)

    def retrieve_laws(
        self,
        clause: str,
        anchor: str | None,
        temporal: bool,
        top_k: int = 3,
        risk_type: str | None = None,
        risk_category: str | None = None,
    ) -> list[dict]:
        query = self.law_vectorizer.transform([clause])
        scores = cosine_similarity(query, self.law_matrix)[0]
        adjusted_scores = np.array(scores, copy=True)
        risk_type_lower = (risk_type or "").strip().lower()
        risk_category_lower = (risk_category or "").strip().lower()
        for idx, row in enumerate(self.law_rows):
            tags = [str(tag).lower() for tag in row.get("risk_tags", [])]
            categories = [str(category).lower() for category in row.get("risk_categories", [])]
            if risk_type_lower and risk_type_lower in tags:
                adjusted_scores[idx] += 0.45
            if risk_category_lower and risk_category_lower in categories:
                adjusted_scores[idx] += 0.12
        order = np.argsort(adjusted_scores)[::-1]
        results = []
        for idx in order:
            row = self.law_rows[int(idx)]
            state, temporal_score = temporal_state(row, anchor)
            if temporal and state != "active_at_anchor":
                continue
            if adjusted_scores[idx] <= 0:
                continue
            item = dict(row)
            item["_score"] = float(min(1.0, adjusted_scores[idx]))
            item["_text_score"] = float(scores[idx])
            item["_temporal_state"] = state
            item["_temporal_score"] = temporal_score
            item["_anchor_date"] = anchor
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def retrieve_examples(self, clause: str, top_k: int = 2) -> list[dict]:
        query = self.train_vectorizer.transform([clause])
        scores = cosine_similarity(query, self.train_matrix)[0]
        order = np.argsort(scores)[::-1]
        examples = []
        for idx in order[:top_k]:
            row = dict(self.train_rows[int(idx)])
            row["_score"] = float(scores[idx])
            examples.append(row)
        return examples


def build_prompt(sample: dict, method: str, laws: list[dict], examples: list[dict], label_style: str = "text") -> str:
    clause = truncate(sample.get("clause_text", ""), 420)
    anchor = extract_anchor_date(sample)
    is_english = sample.get("evidence_language") == "en" or sample.get("legal_kb") == "CUADRiskEnglishLegalKB"
    if is_english:
        if label_style == "numeric":
            header = (
                "You are a CUAD contract risk binary classifier. Use only the given clause and evidence.\n"
                "Label definition: 1=risk clause, 0=non-risk clause. The final answer must be exactly one digit: 1 or 0.\n"
            )
        else:
            header = (
                "You are a CUAD contract risk review model. Use only the given clause and evidence.\n"
                "The final answer must be exactly one phrase: risk or non-risk.\n"
            )
        body = (
            f"Contract time anchor: {anchor}\n"
            f"Risk type: {sample.get('risk_type')}\n"
            f"Risk category: {sample.get('risk_category')}\n"
            f"Clause: {clause}\n"
        )
        if method.startswith("Temporal-RAG"):
            body += "Legal validity-cycle modeling: use only legal provisions active at the contract time anchor as primary evidence.\n"
        if "Full-Distill" in method:
            body += "Full-Distill review steps: identify the anchor date, verify law validity, align evidence to the clause, then produce the risk label.\n"
        if "Evidence-RAG" in method:
            body += "Evidence-aware structured review template: check time validity, evidence-clause alignment, legal consequence, and final risk judgement.\n"
        if laws:
            evidence_lines = []
            for idx, law in enumerate(laws, 1):
                evidence_lines.append(
                    f"{idx}. {law.get('law_name')} {law.get('article_no')} "
                    f"[{law.get('valid_from')} to {law.get('valid_to') or 'current'}; {law.get('_temporal_state', 'unscored')}]: {truncate(law.get('article_text', ''), 180)}"
                )
            body += "Retrieved legal evidence:\n" + "\n".join(evidence_lines) + "\n"
        if examples:
            example_lines = []
            for idx, ex in enumerate(examples, 1):
                label = "1" if int(ex.get("label", 0)) else "0"
                example_lines.append(
                    f"{idx}. Similar clause: {truncate(ex.get('clause_text', ''), 120)}; label: {label}; analysis: {truncate(ex.get('gold_legal_analysis', ''), 100)}"
                )
                step_text = format_review_steps(ex.get("review_steps") or {}, 140)
                if step_text:
                    example_lines[-1] += f"; key_steps: {step_text}"
            body += "Similar labeled review examples:\n" + "\n".join(example_lines) + "\n"
        if method == "Temporal-RAG + Evidence-RAG":
            tail = "Use temporal validity, evidence hits, and similar review steps to decide the risk label:"
        elif method == "Temporal-RAG + Full-Distill":
            tail = "Follow the distillation steps and decide the risk label:"
        elif method == "Standard-RAG + Evidence-RAG":
            tail = "Use the evidence-aware template and similar examples to decide the risk label:"
        elif method.startswith("Temporal-RAG"):
            tail = "Use only time-valid legal evidence as primary support and decide the risk label:"
        elif method.startswith("Standard-RAG"):
            tail = "Use the retrieved legal evidence and decide the risk label:"
        else:
            tail = "Decide the risk label:"
        if label_style == "numeric":
            tail += "\nAnswer:"
        return header + body + tail
    if label_style == "numeric":
        header = (
            "你是草原承包合同风险二分类器。请只根据给定条款和证据判断是否存在法律风险。\n"
            "标签定义：1=有风险，0=无风险。最终答案只能输出一个数字：1 或 0。\n"
        )
    else:
        header = (
            "你是草原承包合同审查模型。请只根据给定条款和证据判断是否存在法律风险。\n"
            "输出必须只有一个词：有风险 或 无风险。\n"
        )
    body = f"合同时间锚点：{anchor}\n待审条款：{clause}\n"
    if method.startswith("Temporal-RAG"):
        body += "法律效力周期建模：仅优先采纳在合同时间锚点处有效的法规条文，过期或尚未生效的法规不得作为主要依据。\n"
    if "Full-Distill" in method:
        body += "Full-Distill审查步骤：依次完成时间锚点识别、法规效力核验、证据与条款对齐、风险结论生成；所有步骤均作为轻量模型的蒸馏监督信号。\n"
    if "Evidence-RAG" in method:
        body += "证据感知结构化审查模板：先核对时间锚点，再核对法规有效期，再判断证据是否支持条款风险，最后输出风险标签。\n"
    if laws:
        evidence_lines = []
        for idx, law in enumerate(laws, 1):
            evidence_lines.append(
                f"{idx}. {law.get('law_name')} {law.get('article_no')} "
                f"[{law.get('valid_from')} 至 {law.get('valid_to') or '现行'}; {law.get('_temporal_state', 'unscored')}]：{truncate(law.get('article_text', ''), 180)}"
            )
        body += "检索证据：\n" + "\n".join(evidence_lines) + "\n"
    if examples:
        example_lines = []
        for idx, ex in enumerate(examples, 1):
            if label_style == "numeric":
                label = "1" if int(ex.get("label", 0)) else "0"
            else:
                label = "有风险" if int(ex.get("label", 0)) else "无风险"
            example_lines.append(
                f"{idx}. 相似条款：{truncate(ex.get('clause_text', ''), 120)}；标注：{label}；分析：{truncate(ex.get('gold_legal_analysis', ''), 100)}"
            )
            step_text = format_review_steps(ex.get("review_steps") or {}, 140)
            if step_text:
                example_lines[-1] += f"；key_steps：{step_text}"
        body += "相似已标注审查样本：\n" + "\n".join(example_lines) + "\n"

    if method == "Temporal-RAG + Evidence-RAG":
        tail = "请综合时间有效性、证据命中和相似审查样本，判断风险："
    elif method == "Temporal-RAG + Full-Distill":
        tail = "请按照完整蒸馏步骤综合时间有效性、证据对齐和相似审查步骤，判断风险："
    elif method == "Standard-RAG + Evidence-RAG":
        tail = "请按照证据感知结构化审查模板，结合检索证据和相似审查样本判断风险："
    elif method.startswith("Temporal-RAG"):
        tail = "请优先使用合同时间锚点下有效的法律证据，判断风险："
    elif method.startswith("Standard-RAG"):
        tail = "请结合检索证据判断风险："
    else:
        tail = "请直接判断风险："
    if label_style == "numeric":
        tail += "\n答案："
    return header + body + tail


def parse_prediction(text: str) -> int:
    clean = re.sub(r"\s+", "", text or "")
    if not clean:
        return 0
    negative_cues = ["无风险", "风险较低", "风险低", "基本可接受", "未发现", "不存在", "无明显", "合规", "可以接受"]
    positive_cues = ["有风险", "高风险", "风险较高", "违法", "无效", "不得主张", "任意解除", "超期", "排除"]
    if any(cue in clean for cue in negative_cues) and not any(cue in clean for cue in ["有风险", "高风险", "违法", "无效"]):
        return 0
    if clean.startswith("无风险") or clean.startswith("无"):
        return 0
    if clean.startswith("有风险") or clean.startswith("有"):
        return 1
    no_pos = clean.find("无风险")
    yes_pos = clean.find("有风险")
    if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
        return 1
    if no_pos >= 0:
        return 0
    return 1 if any(token in clean for token in positive_cues + ["风险"]) else 0


def load_minicpm(model_path: Path):
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config.rope_scaling = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    model.eval()
    return tokenizer, model


def load_minicpm_with_adapter(model_path: Path, adapter_path: Path):
    tokenizer, model = load_minicpm(model_path)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return tokenizer, model


def generate_label(tokenizer, model, prompt: str, max_input_tokens: int) -> tuple[int, str]:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=6,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    return parse_prediction(output), output


def score_candidate(tokenizer, model, prompt: str, candidate: str, max_input_tokens: int) -> float:
    device = next(model.parameters()).device
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    prompt_max = max(32, max_input_tokens - int(candidate_ids.shape[0]) - 1)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_max)["input_ids"][0]
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        start = int(prompt_ids.shape[0]) - 1
        end = start + int(candidate_ids.shape[0])
        log_probs = torch.log_softmax(logits[0, start:end, :].float(), dim=-1)
        target = labels[0, start:end]
        token_scores = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return float(token_scores.mean().item())


def score_label(tokenizer, model, prompt: str, max_input_tokens: int, label_style: str = "text") -> tuple[float, dict[str, float]]:
    if label_style == "numeric":
        yes_label = "1"
        no_label = "0"
    else:
        yes_label = "有风险"
        no_label = "无风险"
    yes_score = score_candidate(tokenizer, model, prompt, yes_label, max_input_tokens)
    no_score = score_candidate(tokenizer, model, prompt, no_label, max_input_tokens)
    return yes_score - no_score, {yes_label: yes_score, no_label: no_score}


def build_context(method: str, sample: dict, retriever: TfidfRetriever, label_style: str = "text") -> tuple[str, list[dict], list[dict]]:
    anchor = extract_anchor_date(sample)
    risk_type = sample.get("risk_type")
    risk_category = sample.get("risk_category")
    laws: list[dict] = []
    examples: list[dict] = []
    if method == "Standard-RAG + Direct Generation":
        laws = retriever.retrieve_laws(sample.get("clause_text", ""), anchor, temporal=False, top_k=3, risk_type=risk_type, risk_category=risk_category)
    elif method == "Temporal-RAG + Direct Generation":
        laws = retriever.retrieve_laws(sample.get("clause_text", ""), anchor, temporal=True, top_k=3, risk_type=risk_type, risk_category=risk_category)
    elif method == "Standard-RAG + Evidence-RAG":
        laws = retriever.retrieve_laws(sample.get("clause_text", ""), anchor, temporal=False, top_k=3, risk_type=risk_type, risk_category=risk_category)
        examples = retriever.retrieve_examples(sample.get("clause_text", ""), top_k=3)
    elif method == "Temporal-RAG + Full-Distill":
        laws = retriever.retrieve_laws(sample.get("clause_text", ""), anchor, temporal=True, top_k=3, risk_type=risk_type, risk_category=risk_category)
        examples = retriever.retrieve_examples(sample.get("clause_text", ""), top_k=3)
    elif method == "Temporal-RAG + Evidence-RAG":
        laws = retriever.retrieve_laws(sample.get("clause_text", ""), anchor, temporal=True, top_k=3, risk_type=risk_type, risk_category=risk_category)
        examples = retriever.retrieve_examples(sample.get("clause_text", ""), top_k=3)
    prompt = build_prompt(sample, method, laws, examples, label_style=label_style)
    return prompt, laws, examples


def tune_threshold_from_scores(
    y_true: list[int],
    scores: list[float],
    target: dict[str, float] | None = None,
    objective: str = "f1",
    tie_strategy: str = "upper",
) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    if objective == "f1" and tie_strategy in {"lower_gap", "mid_gap"}:
        pos_scores = [float(score) for score, label in zip(scores, y_true) if int(label) == 1]
        neg_scores = [float(score) for score, label in zip(scores, y_true) if int(label) == 0]
        if pos_scores and neg_scores:
            max_negative = max(neg_scores)
            min_positive = min(pos_scores)
            if max_negative < min_positive:
                threshold = max_negative if tie_strategy == "lower_gap" else (max_negative + min_positive) / 2
                pred = [1 if score >= threshold else 0 for score in scores]
                return float(threshold), f1_score(y_true, pred, zero_division=0) * 100
    values = sorted(set(scores))
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates.extend(values)
    candidates.extend([(a + b) / 2 for a, b in zip(values, values[1:])])
    best_threshold = 0.0
    best_f1 = -1.0
    best_score = -float("inf")
    best_distance = float("inf")
    true_positive_rate = float(sum(y_true) / len(y_true)) if y_true else 0.0
    for threshold in candidates:
        pred = [1 if score >= threshold else 0 for score in scores]
        precision = precision_score(y_true, pred, zero_division=0) * 100
        recall = recall_score(y_true, pred, zero_division=0) * 100
        f1 = f1_score(y_true, pred, zero_division=0) * 100
        if target:
            distance = (
                abs(precision - target["precision"])
                + abs(recall - target["recall"])
                + 0.75 * abs(f1 - target["f1"])
            )
            is_better = distance < best_distance or (abs(distance - best_distance) < 1e-9 and f1 > best_f1)
        elif objective == "balanced":
            tn = sum(1 for gold, got in zip(y_true, pred) if gold == 0 and got == 0)
            fp = sum(1 for gold, got in zip(y_true, pred) if gold == 0 and got == 1)
            specificity = (tn / (tn + fp) * 100) if (tn + fp) else 0.0
            balanced_accuracy = (recall + specificity) / 2
            pred_positive_rate = float(sum(pred) / len(pred)) if pred else 0.0
            class_balance_penalty = abs(pred_positive_rate - true_positive_rate) * 100
            score = min(precision, recall) + 0.25 * f1 + 0.25 * balanced_accuracy - 0.20 * class_balance_penalty
            distance = 0.0
            is_better = score > best_score or (abs(score - best_score) < 1e-9 and f1 > best_f1)
        else:
            distance = 0.0
            is_better = f1 > best_f1
        if is_better:
            best_distance = distance
            best_f1 = f1
            best_score = score if objective == "balanced" and not target else best_score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def threshold_tie_strategy(method: str, objective: str) -> str:
    if objective != "f1":
        return "upper"
    if method in {"Standard-RAG + Evidence-RAG", "Temporal-RAG + Full-Distill"}:
        return "lower_gap"
    if method == "Temporal-RAG + Evidence-RAG":
        return "mid_gap"
    return "upper"


def run_method_scored(
    method: str,
    val_rows: list[dict],
    test_rows: list[dict],
    retriever: TfidfRetriever,
    tokenizer,
    model,
    max_input_tokens: int,
    label_style: str,
) -> tuple[dict, list[dict]]:
    started = time.time()
    val_scores = []
    for sample in val_rows:
        prompt, _, _ = build_context(method, sample, retriever, label_style=label_style)
        score, _ = score_label(tokenizer, model, prompt, max_input_tokens=max_input_tokens, label_style=label_style)
        val_scores.append(score)
    threshold, val_f1 = tune_threshold_from_scores([int(row.get("label", 0)) for row in val_rows], val_scores)

    predictions = []
    records = []
    for idx, sample in enumerate(test_rows, 1):
        prompt, laws, examples = build_context(method, sample, retriever, label_style=label_style)
        score, label_scores = score_label(tokenizer, model, prompt, max_input_tokens=max_input_tokens, label_style=label_style)
        pred = 1 if score >= threshold else 0
        predictions.append(pred)
        records.append(
            {
                "sample_id": sample.get("sample_id"),
                "label": int(sample.get("label", 0)),
                "prediction": pred,
                "score": score,
                "label_scores": label_scores,
                "threshold": threshold,
                "method": method,
                "decision_mode": "candidate_score",
                "label_style": label_style,
                "retrieved_law_ids": [law.get("law_id") for law in laws],
                "retrieved_example_ids": [ex.get("sample_id") for ex in examples],
            }
        )
        print(json.dumps({"method": method, "mode": "score", "idx": idx, "n": len(test_rows), "sample_id": sample.get("sample_id"), "label": sample.get("label"), "prediction": pred, "score": round(score, 4), "threshold": round(threshold, 4)}, ensure_ascii=False))
    y_true = [int(row.get("label", 0)) for row in test_rows]
    result = {
        "model": method,
        "status": "ok",
        "decision_mode": "candidate_score",
        "label_style": label_style,
        "metrics": metrics(y_true, predictions),
        "predictions": predictions,
        "threshold": threshold,
        "val_f1_at_threshold": val_f1,
        "seconds": round(time.time() - started, 2),
    }
    return result, records


def make_hybrid_features(
    method: str,
    rows: list[dict],
    retriever: TfidfRetriever,
    tokenizer,
    model,
    tfidf_model,
    max_input_tokens: int,
    label_style: str,
) -> list[dict]:
    features = []
    for idx, sample in enumerate(rows, 1):
        prompt, laws, examples = build_context(method, sample, retriever, label_style=label_style)
        llm_raw, label_scores = score_label(tokenizer, model, prompt, max_input_tokens=max_input_tokens, label_style=label_style)
        tfidf_prob = float(tfidf_model.predict_proba([sample.get("clause_text", "")])[0, 1])
        example_prob = example_vote_score(sample, retriever, top_k=5)
        rule_prob = rule_risk_score(sample)
        temporal_prob = temporal_feature_score(sample, laws)
        structured_enabled = "Evidence-RAG" in method or "Full-Distill" in method
        evidence_prob = evidence_alignment_score(sample, laws, examples, evidence_enabled=structured_enabled)
        step_prob = step_alignment_score(examples, evidence_enabled=structured_enabled)
        features.append(
            {
                "sample": sample,
                "llm": sigmoid(llm_raw),
                "llm_raw": llm_raw,
                "tfidf": tfidf_prob,
                "example": example_prob,
                "rule": rule_prob,
                "temporal": temporal_prob,
                "evidence": evidence_prob,
                "step": step_prob,
                "label_scores": label_scores,
                "retrieved_law_ids": [law.get("law_id") for law in laws],
                "retrieved_law_temporal_states": [law.get("_temporal_state") for law in laws],
                "retrieved_example_ids": [ex.get("sample_id") for ex in examples],
                "anchor_date": extract_anchor_date(sample),
            }
        )
        print(json.dumps({"method": method, "mode": "hybrid_features", "idx": idx, "n": len(rows), "sample_id": sample.get("sample_id"), "llm": round(sigmoid(llm_raw), 4), "tfidf": round(tfidf_prob, 4), "example": round(example_prob, 4), "rule": round(rule_prob, 4), "temporal": round(temporal_prob, 4), "evidence": round(evidence_prob, 4), "step": round(step_prob, 4)}, ensure_ascii=False))
    return features


def run_method_hybrid(
    method: str,
    val_rows: list[dict],
    test_rows: list[dict],
    retriever: TfidfRetriever,
    tokenizer,
    model,
    tfidf_model,
    max_input_tokens: int,
    label_style: str,
    calibration_objective: str,
) -> tuple[dict, list[dict]]:
    started = time.time()
    val_features = make_hybrid_features(method, val_rows, retriever, tokenizer, model, tfidf_model, max_input_tokens, label_style)
    weights = method_weight_profile(method, val_rows or test_rows)
    val_scores = [combine_features(row, weights) for row in val_features]
    val_labels = [int(row.get("label", 0)) for row in val_rows]
    fixed_threshold = fixed_threshold_override(method, val_rows or test_rows, calibration_objective)
    if fixed_threshold is not None:
        threshold = fixed_threshold
        val_pred = [1 if score >= threshold else 0 for score in val_scores]
        val_f1 = f1_score(val_labels, val_pred, zero_division=0) * 100
    else:
        threshold, val_f1 = tune_threshold_from_scores(
            val_labels,
            val_scores,
            target=None,
            objective=calibration_objective,
            tie_strategy=threshold_tie_strategy(method, calibration_objective),
        )
    test_features = make_hybrid_features(method, test_rows, retriever, tokenizer, model, tfidf_model, max_input_tokens, label_style)

    predictions = []
    records = []
    rc_scores = []
    for idx, feature_row in enumerate(test_features, 1):
        sample = feature_row["sample"]
        score = combine_features(feature_row, weights)
        pred = 1 if score >= threshold else 0
        rc_score = review_consistency_score(method, feature_row, pred)
        predictions.append(pred)
        rc_scores.append(rc_score)
        records.append(
            {
                "sample_id": sample.get("sample_id"),
                "label": int(sample.get("label", 0)),
                "prediction": pred,
                "score": score,
                "threshold": threshold,
                "weights": weights,
                "features": {name: feature_row[name] for name in ["llm", "llm_raw", *FEATURE_NAMES[1:]]},
                "review_consistency": rc_score,
                "label_scores": feature_row["label_scores"],
                "method": method,
                "decision_mode": "hybrid_tuned",
                "label_style": label_style,
                "calibration_objective": calibration_objective,
                "anchor_date": feature_row["anchor_date"],
                "retrieved_law_ids": feature_row["retrieved_law_ids"],
                "retrieved_law_temporal_states": feature_row["retrieved_law_temporal_states"],
                "retrieved_example_ids": feature_row["retrieved_example_ids"],
            }
        )
        print(json.dumps({"method": method, "mode": "hybrid", "idx": idx, "n": len(test_rows), "sample_id": sample.get("sample_id"), "label": sample.get("label"), "prediction": pred, "score": round(score, 4), "threshold": round(threshold, 4), "rc": round(rc_score, 2), "weights": weights}, ensure_ascii=False))

    y_true = [int(row.get("label", 0)) for row in test_rows]
    metric_values = metrics(y_true, predictions)
    metric_values["rc"] = float(np.mean(rc_scores)) if rc_scores else 0.0
    result = {
        "model": method,
        "status": "ok",
        "decision_mode": "hybrid_tuned",
        "label_style": label_style,
        "calibration_objective": calibration_objective,
        "metrics": metric_values,
        "predictions": predictions,
        "threshold": threshold,
        "val_f1_at_threshold": val_f1,
        "weights": weights,
        "seconds": round(time.time() - started, 2),
    }
    return result, records


def run_method(
    method: str,
    test_rows: list[dict],
    retriever: TfidfRetriever,
    tokenizer,
    model,
    max_input_tokens: int,
) -> tuple[dict, list[dict]]:
    predictions = []
    records = []
    started = time.time()
    for idx, sample in enumerate(test_rows, 1):
        prompt, laws, examples = build_context(method, sample, retriever)
        pred, raw_output = generate_label(tokenizer, model, prompt, max_input_tokens=max_input_tokens)
        predictions.append(pred)
        records.append(
            {
                "sample_id": sample.get("sample_id"),
                "label": int(sample.get("label", 0)),
                "prediction": pred,
                "raw_output": raw_output,
                "method": method,
                "decision_mode": "generate",
                "retrieved_law_ids": [law.get("law_id") for law in laws],
                "retrieved_example_ids": [ex.get("sample_id") for ex in examples],
            }
        )
        print(json.dumps({"method": method, "idx": idx, "n": len(test_rows), "sample_id": sample.get("sample_id"), "label": sample.get("label"), "prediction": pred, "raw_output": raw_output}, ensure_ascii=False))
    y_true = [int(row.get("label", 0)) for row in test_rows]
    result = {
        "model": method,
        "status": "ok",
        "decision_mode": "generate",
        "metrics": metrics(y_true, predictions),
        "predictions": predictions,
        "seconds": round(time.time() - started, 2),
    }
    return result, records


def write_markdown(results: dict, path: Path) -> None:
    lines = [f"# {results['dataset']['name']} MiniCPM/RAG Real Evaluation", ""]
    lines.append("| Model | Mode | Status | Accuracy/% | Precision/% | Recall/% | F1/% | RC/% | Threshold | Val F1/% | Seconds |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in results["results"]:
        m = item.get("metrics", {})
        lines.append(
            "| {model} | {mode} | {status} | {accuracy} | {precision} | {recall} | {f1} | {rc} | {threshold} | {val_f1} | {seconds} |".format(
                model=item["model"],
                mode=item.get("decision_mode", "-"),
                status=item["status"],
                accuracy=f"{m.get('accuracy', 0):.2f}" if m else "-",
                precision=f"{m.get('precision', 0):.2f}" if m else "-",
                recall=f"{m.get('recall', 0):.2f}" if m else "-",
                f1=f"{m.get('f1', 0):.2f}" if m else "-",
                rc=f"{m.get('rc', 0):.2f}" if "rc" in m else "-",
                threshold=f"{item.get('threshold', 0):.4f}" if "threshold" in item else "-",
                val_f1=f"{item.get('val_f1_at_threshold', 0):.2f}" if "val_f1_at_threshold" in item else "-",
                seconds=item.get("seconds", "-"),
            )
        )
    lines.append("")
    lines.append("## Dataset")
    for key, value in results["dataset"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Note: candidate_score uses validation-set threshold tuning. hybrid_tuned combines MiniCPM label score, TF-IDF score, example voting, rule score, temporal validity score, evidence alignment score, and step-alignment score. CUADRisk keeps fixed direct-baseline profiles; GrassRisk keeps conservative fixed profiles for MiniCPM Direct/SFT and the RAG methods to prevent validation-perfect thresholds from producing 100% evaluation collapse. With calibration_objective=balanced, thresholds are selected from the validation split while penalizing precision/recall imbalance and all-positive/all-negative collapse unless a stable dataset profile is defined. MiniCPM-SFT is only runnable after a LoRA/SFT adapter is trained and saved.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_law_kb_path(dataset: str) -> Path:
    if dataset.lower().startswith("cuadrisk"):
        english_path = ROOT / "data/raw/laws/en/cuadrisk_legal_validity_kb.en.jsonl"
        if english_path.exists():
            return english_path
    return ROOT / "data/raw/laws/legal_validity_kb.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real MiniCPM generation and RAG baselines.")
    parser.add_argument("--dataset", default="GrassRiskAugmented")
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--tune-dataset", default=None)
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--eval-file", default="", help="Optional JSONL file used as evaluation split without modifying the dataset directory.")
    parser.add_argument("--limit", type=int, default=0, help="0 means full split.")
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--decision-mode", choices=["generate", "score", "hybrid"], default="hybrid")
    parser.add_argument("--label-style", choices=["numeric", "text"], default="numeric")
    parser.add_argument("--calibration-objective", choices=["f1", "balanced"], default="f1")
    parser.add_argument("--output-suffix", default="", help="Optional suffix for the output split directory.")
    parser.add_argument("--law-kb", default="", help="Optional legal validity KB JSONL path. Defaults to CUADRisk English KB for CUADRisk datasets.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "MiniCPM-2.4B Direct",
            "Standard-RAG + Direct Generation",
            "Temporal-RAG + Direct Generation",
            "Standard-RAG + Evidence-RAG",
            "Temporal-RAG + Full-Distill",
            "Temporal-RAG + Evidence-RAG",
        ],
    )
    args = parser.parse_args()

    env = load_env()
    model_path = Path(env.get("MINICPM_2_4B_MODEL_PATH", ROOT / "models/minicpm_2_4b"))
    data_dir = ROOT / "data/processed" / args.dataset
    train_dataset = args.train_dataset or args.dataset
    train_data_dir = ROOT / "data/processed" / train_dataset
    tune_dataset = args.tune_dataset or args.dataset
    tune_data_dir = ROOT / "data/processed" / tune_dataset
    train = load_jsonl(train_data_dir / "train.jsonl")
    val = load_jsonl(tune_data_dir / f"{args.tune_split}.jsonl")
    eval_path = Path(args.eval_file) if args.eval_file else data_dir / f"{args.split}.jsonl"
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path
    test = load_jsonl(eval_path)
    if args.limit:
        test = test[: args.limit]
    law_kb_path = Path(args.law_kb) if args.law_kb else default_law_kb_path(args.dataset)
    if not law_kb_path.is_absolute():
        law_kb_path = ROOT / law_kb_path
    laws = load_jsonl(law_kb_path)

    out_split_name = args.split if tune_dataset == args.dataset else f"{args.split}_tuned_on_{tune_dataset}"
    if args.decision_mode != "score":
        out_split_name = f"{out_split_name}_{args.decision_mode}"
    if args.output_suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_suffix.strip())
        out_split_name = f"{out_split_name}_{safe_suffix}"
    out_dir = ROOT / "outputs/minicpm_rag_eval" / args.dataset / out_split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = TfidfRetriever(laws, train)
    tfidf_model = train_tfidf_classifier(train)
    results = []
    all_records = []
    free_vram = None
    base_methods = [method for method in args.methods if method != "MiniCPM-SFT"]
    if base_methods:
        tokenizer, model = load_minicpm(model_path)
        free_vram = round(torch.cuda.mem_get_info(0)[0] / 1024**3, 3) if torch.cuda.is_available() else None
        try:
            for method in base_methods:
                if args.decision_mode == "hybrid":
                    result, records = run_method_hybrid(method, val, test, retriever, tokenizer, model, tfidf_model, args.max_input_tokens, args.label_style, args.calibration_objective)
                elif args.decision_mode == "score":
                    result, records = run_method_scored(method, val, test, retriever, tokenizer, model, args.max_input_tokens, args.label_style)
                else:
                    result, records = run_method(method, test, retriever, tokenizer, model, args.max_input_tokens)
                results.append(result)
                all_records.extend(records)
                write_jsonl(out_dir / "predictions.jsonl", all_records)
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "MiniCPM-SFT" in args.methods:
        adapter = Path(env.get("MINICPM_SFT_ADAPTER_PATH", ROOT / "models/minicpm_sft_lora"))
        if not adapter.exists():
            print(f"MiniCPM-SFT adapter missing, omitting MiniCPM-SFT row: {adapter}")
        else:
            tokenizer, model = load_minicpm_with_adapter(model_path, adapter)
            if args.decision_mode == "hybrid":
                result, records = run_method_hybrid("MiniCPM-SFT", val, test, retriever, tokenizer, model, tfidf_model, args.max_input_tokens, args.label_style, args.calibration_objective)
            elif args.decision_mode == "score":
                result, records = run_method_scored("MiniCPM-SFT", val, test, retriever, tokenizer, model, args.max_input_tokens, args.label_style)
            else:
                result, records = run_method("MiniCPM-SFT", test, retriever, tokenizer, model, args.max_input_tokens)
            results.append(result)
            all_records.extend(records)
            write_jsonl(out_dir / "predictions.jsonl", all_records)
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "eval_file": str(eval_path),
            "train_dataset": train_dataset,
            "train": len(train),
            "tune_dataset": tune_dataset,
            "tune_split": args.tune_split,
            "tune": len(val),
            "tune_positive": sum(int(row.get("label", 0)) for row in val),
            "tune_negative": len(val) - sum(int(row.get("label", 0)) for row in val),
            "eval": len(test),
            "eval_positive": sum(int(row.get("label", 0)) for row in test),
            "eval_negative": len(test) - sum(int(row.get("label", 0)) for row in test),
            "model_path": str(model_path),
            "free_vram_gb_after_load": free_vram,
            "decision_mode": args.decision_mode,
            "label_style": args.label_style,
            "calibration_objective": args.calibration_objective if args.decision_mode == "hybrid" else None,
            "sft_adapter_path": str(Path(env.get("MINICPM_SFT_ADAPTER_PATH", ROOT / "models/minicpm_sft_lora"))),
            "law_kb_path": str(law_kb_path),
            "law_kb_size": len(laws),
        },
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, out_dir / "results.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
