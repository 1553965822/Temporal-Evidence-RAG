#!/usr/bin/env python
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
ANNOTATIONS = ROOT / "data/raw/annotations"
CONTRACTS = ROOT / "data/raw/contracts_generated"
SUMMARY_DIR = ROOT / "outputs/dataset_audit/FinalDatasets"

SOURCE_TO_FINAL = {
    "GrassRiskUser": "GrassRisk",
    "CUADRiskUser": "CUADRisk",
}

FINAL_FIELDS = [
    "sample_id",
    "contract_id",
    "dataset",
    "task",
    "clause_text",
    "anchor_date",
    "risk_type",
    "risk_category",
    "label",
    "label_name",
    "gold_evidence_ids",
    "evidence_text",
    "review_steps",
    "gold_legal_analysis",
]

REMOVE_PROCESSED = [
    "GrassRiskUser",
    "CUADRiskUser",
    "GrassRiskAugmented",
    "GrassRiskExpandedEval",
    "GrassRiskReal",
    "CUADRiskHard",
    "CUADRiskTemporalCN",
    "GLTRD",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rename_id(value: str | None, dataset: str, index: int, is_contract: bool = False) -> str:
    if value:
        renamed = re.sub(r"^(GrassRiskUser|CUADRiskUser)-", f"{dataset}-", str(value))
        if renamed != value:
            return renamed
    if is_contract:
        return f"{dataset}-C{(index - 1) // 8 + 1:04d}"
    return f"{dataset}-{index:04d}"


def convert_row(row: dict, dataset: str, index: int) -> dict:
    out = dict(row)
    out["sample_id"] = rename_id(out.get("sample_id"), dataset, index)
    out["contract_id"] = rename_id(out.get("contract_id"), dataset, index, is_contract=True)
    out["dataset"] = dataset
    out.setdefault("task", "contract_risk_review")
    out.setdefault("risk_category", "generic")
    out.setdefault("gold_legal_analysis", " ".join(str(v) for v in out.get("review_steps", {}).values()))
    return {field: out.get(field) for field in FINAL_FIELDS}


def copy_contracts(src_dataset: str, dst_dataset: str) -> None:
    src_dir = CONTRACTS / src_dataset
    dst_dir = CONTRACTS / dst_dataset
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    if not src_dir.exists():
        return
    shutil.copytree(src_dir, dst_dir)
    for path in list(dst_dir.glob("*.txt")):
        new_name = re.sub(r"^(GrassRiskUser|CUADRiskUser)-", f"{dst_dataset}-", path.name)
        text = path.read_text(encoding="utf-8")
        text = text.replace(src_dataset, dst_dataset)
        text = re.sub(r"(GrassRiskUser|CUADRiskUser)-", f"{dst_dataset}-", text)
        path.write_text(text, encoding="utf-8")
        if new_name != path.name:
            path.rename(path.with_name(new_name))


def finalize_dataset(src_dataset: str, dst_dataset: str) -> dict:
    src_dir = PROCESSED / src_dataset
    dst_dir = PROCESSED / dst_dataset
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    all_rows: list[dict] = []
    split_counts = {}
    for split in ["train", "val", "test"]:
        rows = []
        for row in read_jsonl(src_dir / f"{split}.jsonl"):
            rows.append(convert_row(row, dst_dataset, index))
            index += 1
        write_jsonl(dst_dir / f"{split}.jsonl", rows)
        all_rows.extend(rows)
        labels = [int(row["label"]) for row in rows]
        split_counts[split] = {
            "total": len(rows),
            "positive": sum(labels),
            "negative": len(labels) - sum(labels),
        }
    write_jsonl(dst_dir / "all.jsonl", all_rows)

    annotations = [
        {
            "sample_id": row["sample_id"],
            "contract_id": row["contract_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "risk_type": row["risk_type"],
            "risk_category": row["risk_category"],
            "anchor_date": row["anchor_date"],
            "gold_evidence_ids": row["gold_evidence_ids"],
            "evidence_text": row["evidence_text"],
        }
        for row in all_rows
    ]
    annotation_path = ANNOTATIONS / f"{dst_dataset}_annotations.jsonl"
    write_jsonl(annotation_path, annotations)
    copy_contracts(src_dataset, dst_dataset)

    labels = [int(row["label"]) for row in all_rows]
    audit = {
        "dataset": dst_dataset,
        "source_dataset": src_dataset,
        "total": len(all_rows),
        "label_counts": dict(Counter(labels)),
        "split_counts": split_counts,
        "fields": FINAL_FIELDS,
        "processed_dir": str(dst_dir),
        "annotations": str(annotation_path),
        "contracts_dir": str(CONTRACTS / dst_dataset),
        "note": "Final experiment dataset. Intermediate provenance and supplement marker fields are intentionally omitted from rows.",
    }
    (dst_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def remove_intermediate_dirs() -> list[str]:
    removed = []
    for name in REMOVE_PROCESSED:
        path = PROCESSED / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    for name in ["GrassRiskUser", "CUADRiskUser"]:
        path = CONTRACTS / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def write_summary(audits: list[dict], removed: list[str]) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Final Experiment Datasets", ""]
    lines.append("最终实验只保留两个数据集目录：`data/processed/GrassRisk` 和 `data/processed/CUADRisk`。")
    lines.append("数据行中不再保留 `is_generated_supplement` 等中间标记字段。")
    lines.append("")
    for audit in audits:
        lines.append(f"## {audit['dataset']}")
        lines.append("")
        lines.append(f"- processed_dir: `{audit['processed_dir']}`")
        lines.append(f"- annotations: `{audit['annotations']}`")
        lines.append(f"- contracts_dir: `{audit['contracts_dir']}`")
        lines.append(f"- total: {audit['total']}")
        lines.append(f"- label_counts: {audit['label_counts']}")
        lines.append(f"- split_counts: {audit['split_counts']}")
        lines.append(f"- fields: {', '.join(audit['fields'])}")
        sample = read_jsonl(Path(audit["processed_dir"]) / "train.jsonl")[0]
        lines.append("")
        lines.append("### Sample Row")
        lines.append("```json")
        lines.append(json.dumps(sample, ensure_ascii=False, indent=2)[:2600])
        lines.append("```")
        lines.append("")
    lines.append("## Removed Intermediate Dataset Paths")
    for item in removed:
        lines.append(f"- `{item}`")
    path = SUMMARY_DIR / "DATASETS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SUMMARY_DIR / "audit.json").write_text(
        json.dumps({"datasets": audits, "removed": removed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    audits = [finalize_dataset(src, dst) for src, dst in SOURCE_TO_FINAL.items()]
    removed = remove_intermediate_dirs()
    summary = write_summary(audits, removed)
    print(json.dumps({"datasets": audits, "removed": removed, "summary": str(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
