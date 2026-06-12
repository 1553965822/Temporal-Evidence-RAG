#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def take_stratified(rows: list[dict], count: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    positives = [r for r in rows if int(r.get("label", 0)) == 1]
    negatives = [r for r in rows if int(r.get("label", 0)) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_n = min(len(positives), count // 2)
    neg_n = min(len(negatives), count - pos_n)
    selected = positives[:pos_n] + negatives[:neg_n]
    if len(selected) < count:
        remaining_pool = positives[pos_n:] + negatives[neg_n:]
        selected.extend(remaining_pool[: count - len(selected)])
    selected_ids = {id(r) for r in selected}
    remaining = [r for r in rows if id(r) not in selected_ids]
    rng.shuffle(selected)
    return selected, remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an expanded synthetic evaluation split from GrassRiskAugmented.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed/GrassRiskAugmented")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/GrassRiskExpandedEval")
    parser.add_argument("--extra-val", type=int, default=40)
    parser.add_argument("--extra-test", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = load_jsonl(args.input_dir / "train.jsonl")
    val = load_jsonl(args.input_dir / "val.jsonl")
    test = load_jsonl(args.input_dir / "test.jsonl")
    synthetic = [r for r in train if r.get("label_source") == "rule_augmented"]
    real_train = [r for r in train if r.get("label_source") != "rule_augmented"]

    extra_test, remaining = take_stratified(synthetic, args.extra_test, rng)
    extra_val, remaining = take_stratified(remaining, args.extra_val, rng)
    new_train = real_train + remaining
    new_val = val + extra_val
    new_test = test + extra_test
    rng.shuffle(new_train)
    rng.shuffle(new_val)
    rng.shuffle(new_test)

    for row in extra_val + extra_test:
        row["evaluation_note"] = "rule_augmented_eval_label"
    for row in new_train:
        row.setdefault("evaluation_note", "train")

    all_rows = new_train + new_val + new_test
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", new_train)
    write_jsonl(args.output_dir / "val.jsonl", new_val)
    write_jsonl(args.output_dir / "test.jsonl", new_test)
    write_jsonl(args.output_dir / "all.jsonl", all_rows)

    meta = {
        "dataset": "GrassRiskExpandedEval",
        "source": str(args.input_dir),
        "description": "Expanded evaluation split. It keeps the 93 real annotations and moves part of rule-augmented samples into val/test. Use only for stability checks, not as newly collected human ground truth.",
        "split_counts": {
            name: {
                "total": len(rows),
                "positive": sum(int(r.get("label", 0)) for r in rows),
                "negative": len(rows) - sum(int(r.get("label", 0)) for r in rows),
                "rule_augmented": sum(1 for r in rows if r.get("label_source") == "rule_augmented"),
                "human_real": sum(1 for r in rows if r.get("label_source") != "rule_augmented"),
            }
            for name, rows in [("train", new_train), ("val", new_val), ("test", new_test)]
        },
        "label_source_counts": dict(Counter(r.get("label_source", "unknown") for r in all_rows)),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
