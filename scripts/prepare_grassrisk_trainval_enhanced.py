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


def steps_text(row: dict) -> str:
    steps = row.get("review_steps") or {}
    if not isinstance(steps, dict):
        return short(str(steps), 900)
    keys = [
        "s1_summary",
        "s2_trigger",
        "s3_legal_consequence",
        "s4_judgement",
        "temporal_consequence",
    ]
    parts = []
    for key in keys:
        value = steps.get(key)
        if value:
            parts.append(f"{key}: {short(str(value), 240)}")
    return " | ".join(parts)


def make_prompt(row: dict, variant: str) -> str:
    label_def = "标签定义：1=风险条款，0=非风险条款。最终答案只能输出一个数字。"
    base = [
        "你是草原承包合同风险审查模型。",
        label_def,
        f"合同时间锚点：{row.get('anchor_date')}",
        f"风险类型：{row.get('risk_type')}",
        f"风险类别：{row.get('risk_category')}",
        f"待审条款：{short(row.get('clause_text', ''), 1200)}",
    ]
    if variant in {"evidence", "step", "temporal"}:
        base.append(f"有效法律证据摘要：{short(row.get('evidence_text', ''), 900)}")
    if variant in {"step", "temporal"}:
        base.extend(
            [
                "证据感知结构化审查步骤：",
                short(steps_text(row), 1200),
                "请按关键步骤判断：合同时间锚点提取、法律效力周期过滤、证据与条款对齐、法律后果分析、最终风险结论。",
            ]
        )
    elif variant == "evidence":
        base.append("请重点判断条款是否存在权利义务失衡、标准不清、程序缺失、生态保护或救济不足等风险。")
    else:
        base.append("请根据条款文字、风险类型和合同语境判断是否存在法律风险。")
    base.append("答案：")
    return "\n".join(base)


def clone(row: dict, variant: str, idx: int) -> dict:
    out = dict(row)
    out["sample_id"] = f"{row.get('sample_id')}-SFT-{variant}-{idx}"
    out["dataset"] = "GrassRiskTrainValEnhancedV2"
    out["sft_variant"] = variant
    out["sft_prompt"] = make_prompt(row, variant)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enhanced GrassRisk training data without changing the test labels/content.")
    parser.add_argument("--source", default="GrassRisk")
    parser.add_argument("--output", default="GrassRiskTrainValEnhancedV2")
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_dir = ROOT / "data" / "processed" / args.source
    train = load_jsonl(source_dir / "train.jsonl")
    val = load_jsonl(source_dir / "val.jsonl")
    test = load_jsonl(source_dir / "test.jsonl")
    train_val = train + val

    enhanced: list[dict] = []
    for idx, row in enumerate(train_val, 1):
        enhanced.append(clone(row, "direct", idx))
        enhanced.append(clone(row, "evidence", idx))
        enhanced.append(clone(row, "step", idx))
        enhanced.append(clone(row, "temporal", idx))

    positives = [row for row in train_val if int(row.get("label", 0)) == 1]
    negatives = [row for row in train_val if int(row.get("label", 0)) == 0]
    for idx, row in enumerate(rng.sample(positives, len(positives)), 1):
        enhanced.append(clone(row, "step", 100000 + idx))
    for idx, row in enumerate(rng.sample(negatives, len(negatives)), 1):
        enhanced.append(clone(row, "evidence", 200000 + idx))
    rng.shuffle(enhanced)

    out_dir = ROOT / "data" / "processed" / args.output
    write_jsonl(out_dir / "train.jsonl", enhanced)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "all.jsonl", enhanced + val + test)
    audit = {
        "dataset": args.output,
        "source": args.source,
        "policy": "Training data is enhanced from GrassRisk train+val. Original GrassRisk test rows are copied unchanged and are not used to create train rows.",
        "seed": args.seed,
        "source_train_rows": len(train),
        "source_val_rows": len(val),
        "source_test_rows": len(test),
        "enhanced_train_rows": len(enhanced),
        "val_rows": len(val),
        "test_rows_copied_unchanged": len(test),
        "train_positive": sum(int(row.get("label", 0)) for row in enhanced),
        "train_negative": len(enhanced) - sum(int(row.get("label", 0)) for row in enhanced),
        "val_positive": sum(int(row.get("label", 0)) for row in val),
        "test_positive": sum(int(row.get("label", 0)) for row in test),
        "variants": ["direct", "evidence", "step", "temporal", "positive_step_oversample", "negative_evidence_oversample"],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
