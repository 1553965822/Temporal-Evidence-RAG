#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short(text: str, n: int = 900) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


def step_text(row: dict) -> str:
    steps = row.get("review_steps") or {}
    parts = []
    for key in [
        "s1_clause_summary",
        "s2_risk_type",
        "s3_selected_legal_evidence",
        "s4_temporal_alignment",
        "s5_evidence_use",
        "s6_gold_judgement",
    ]:
        value = steps.get(key)
        if value:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(f"{key}: {short(str(value), 220)}")
    return " | ".join(parts)


def make_prompt(row: dict, variant: str) -> str:
    label_def = "Label definition: 1=risk clause, 0=non-risk clause. The final answer must be exactly one digit."
    base = [
        "You are a CUAD contract risk review model.",
        label_def,
        f"Contract time anchor: {row.get('anchor_date')}",
        f"Risk type: {row.get('risk_type')}",
        f"Risk category: {row.get('risk_category')}",
        f"Clause: {short(row.get('clause_text', ''), 1300)}",
    ]
    if variant in {"evidence", "step"}:
        base.append(f"Legal evidence active at the time anchor: {short(row.get('evidence_text', ''), 900)}")
    if variant == "step":
        base.append("Evidence-aware structured review steps:")
        base.append(short(step_text(row), 1200))
        base.append("Follow the key steps: anchor extraction, validity-cycle filtering, evidence-clause alignment, consequence analysis, final risk judgement.")
    elif variant == "evidence":
        base.append("Use the legal evidence to decide whether the clause wording creates overbroad, unilateral, unclear, or non-compliant risk.")
    else:
        base.append("Decide from the clause wording and risk type.")
    base.append("Answer:")
    return "\n".join(base)


def clone(row: dict, variant: str, idx: int) -> dict:
    out = dict(row)
    out["sample_id"] = f"{row.get('sample_id')}-SFT-{variant}-{idx}"
    out["sft_variant"] = variant
    out["sft_prompt"] = make_prompt(row, variant)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="CUADRisk")
    parser.add_argument("--output", default="CUADRiskTrainValEnhanced")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = ROOT / "data/processed" / args.source
    train = load_jsonl(src / "train.jsonl")
    val = load_jsonl(src / "val.jsonl")
    test = load_jsonl(src / "test.jsonl")
    train_val = train + val

    enhanced = []
    for idx, row in enumerate(train_val, 1):
        enhanced.append(clone(row, "direct", idx))
        enhanced.append(clone(row, "evidence", idx))
        enhanced.append(clone(row, "step", idx))

    positives = [row for row in train_val if int(row.get("label", 0)) == 1]
    negatives = [row for row in train_val if int(row.get("label", 0)) == 0]
    for idx, row in enumerate(rng.sample(positives, min(len(positives), len(positives))), 1):
        enhanced.append(clone(row, "step", 100000 + idx))
    for idx, row in enumerate(rng.sample(negatives, min(len(negatives), len(negatives))), 1):
        enhanced.append(clone(row, "evidence", 200000 + idx))
    rng.shuffle(enhanced)

    out = ROOT / "data/processed" / args.output
    write_jsonl(out / "train.jsonl", enhanced)
    write_jsonl(out / "val.jsonl", train_val)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "all.jsonl", enhanced + test)
    audit = {
        "dataset": args.output,
        "source": args.source,
        "policy": "Training/calibration dataset only. Original CUADRisk test split is copied unchanged and is not used for training rows.",
        "train_source_rows": len(train),
        "val_source_rows": len(val),
        "train_val_source_rows": len(train_val),
        "enhanced_train_rows": len(enhanced),
        "test_rows_copied_unchanged": len(test),
        "positive_train_rows": sum(int(row.get("label", 0)) for row in enhanced),
        "negative_train_rows": len(enhanced) - sum(int(row.get("label", 0)) for row in enhanced),
        "variants": {
            "direct": "Clause/risk-type only prompt.",
            "evidence": "Prompt includes active English legal evidence.",
            "step": "Prompt includes evidence-aware structured review steps.",
        },
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

