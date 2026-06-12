#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(os.environ.get("USER_RISK_DATA_DIR", ROOT / "data/raw/user_risk_uploads"))

CUAD_TARGET = {"total": 1012, "positive": 497, "negative": 515}
GRASS_TARGET = {"total": 620, "positive": 312, "negative": 308}

SPLIT_COUNTS = {
    "CUADRiskUser": {
        1: {"train": 298, "val": 99, "test": 100},
        0: {"train": 310, "val": 103, "test": 102},
    },
    "GrassRiskUser": {
        1: {"train": 187, "val": 62, "test": 63},
        0: {"train": 185, "val": 62, "test": 61},
    },
}

ANCHOR_POOL = [
    "2012-06-01",
    "2017-05-01",
    "2019-06-01",
    "2021-02-01",
    "2024-01-15",
]

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

EVIDENCE_BY_CATEGORY = {
    "transfer": "承包经营权流转、转包、转让或委托经营应当符合法律规定和合同约定，并保留同意、备案、登记和生态保护审查程序。",
    "term": "合同期限、续期、延包和终止时间应结合合同时间锚点适用当时有效法律，避免自动续期或期限不明。",
    "payment": "价款、收益分配、补偿、扣款和费用调整应遵循公平、诚信、证据、通知和书面确认程序。",
    "audit": "审计、检查、资料披露和合规监督条款应明确范围、频率、证据留存和双方配合义务。",
    "liability": "违约、赔偿、责任限制、保险和补救条款应明确触发条件、损失证明、整改机会和责任边界。",
    "dispute": "适用法律、管辖、仲裁、诉讼和争议解决条款不得排除法定救济，并应明确外部审查路径。",
    "ecology": "草原经营利用应遵守生态保护、修复、禁牧休牧、防火、防虫害和载畜量控制义务。",
    "use": "草原用途变更、建设、采挖、取土、旅游或商业开发应依法取得批准并完成用途审查。",
    "generic": "合同条款应结合现行有效法律、合同时间锚点、证据材料和审查步骤判断是否存在法律风险。",
}

KEYWORDS_BY_CATEGORY = {
    "transfer": ["流转", "转包", "转让", "承包经营权"],
    "term": ["承包期", "期限", "续期", "延包"],
    "payment": ["价款", "补偿", "费用", "支付"],
    "audit": ["审计", "检查", "监督", "资料"],
    "liability": ["违约", "赔偿", "责任", "解除"],
    "dispute": ["争议", "仲裁", "诉讼", "法院"],
    "ecology": ["草原", "生态", "修复", "载畜量"],
    "use": ["草原", "用途", "建设", "采挖", "审批"],
    "generic": ["合同", "履行", "诚信", "公平"],
}


def stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def read_maybe_broken_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r'(?m)^\{"id"\s*:', text)]
    if not starts:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    starts.append(len(text))
    rows = []
    for idx in range(len(starts) - 1):
        segment = text[starts[idx] : starts[idx + 1]].strip()
        segment = re.sub(r"\r?\n", " ", segment)
        rows.append(json.loads(segment))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_laws() -> list[dict]:
    path = ROOT / "data/raw/laws/legal_validity_kb.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_category(risk_type: str, clause: str) -> str:
    text = f"{risk_type} {clause}".lower()
    rules = [
        ("transfer", ["assign", "assignment", "transfer", "sublicense", "affiliate license", "non-transferable", "转包", "转让", "流转"]),
        ("term", ["effective date", "expiration", "renewal", "term", "期限", "续期", "延包"]),
        ("payment", ["price", "fee", "royalty", "revenue", "profit", "minimum commitment", "payment", "补偿", "费用", "价款"]),
        ("audit", ["audit", "inspection", "books", "records", "审计", "监管", "检查"]),
        ("liability", ["liability", "indemn", "warranty", "insurance", "termination", "违约", "解除", "责任"]),
        ("dispute", ["governing law", "jurisdiction", "dispute", "arbitration", "争议", "仲裁", "诉讼"]),
        ("ecology", ["生态", "修复", "防火", "病虫害", "载畜量", "草原保护"]),
        ("use", ["用途", "建设", "采挖", "取土", "旅游", "审批"]),
    ]
    for category, tokens in rules:
        if any(token in text for token in tokens):
            return category
    return "generic"


def find_evidence_ids(laws: list[dict], category: str, top_k: int = 3) -> list[str]:
    if not laws:
        return []
    keywords = KEYWORDS_BY_CATEGORY.get(category, KEYWORDS_BY_CATEGORY["generic"])
    scored = []
    for row in laws:
        text = f"{row.get('law_name', '')} {row.get('article_no', '')} {row.get('article_text', '')}"
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            active_bonus = 1 if not row.get("valid_to") else 0
            scored.append((score, active_bonus, row.get("law_id", "")))
    scored.sort(reverse=True)
    return [law_id for _, _, law_id in scored[:top_k] if law_id]


def extract_anchor_date(text: str, row_id: str) -> str:
    text = text or ""
    m = re.search(r"(20\d{2}|19\d{2})[-/.年]\s*(\d{1,2})[-/.月]\s*(\d{1,2})", text)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(19\d{2}|20\d{2})", text, re.I)
    if m:
        mo, d, y = m.groups()
        return f"{int(y):04d}-{MONTHS[mo.lower()]}-{int(d):02d}"
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m:
        return f"{int(m.group(1)):04d}-01-01"
    return ANCHOR_POOL[stable_int(row_id or text[:80]) % len(ANCHOR_POOL)]


def detect_language(text: str) -> str:
    zh = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return "zh" if zh >= max(3, latin // 4) else "en"


def normalize_row(raw: dict, dataset: str, index: int, laws: list[dict], source_files: list[str], generated: bool = False) -> dict:
    original_id = str(raw.get("id") or raw.get("sample_id") or f"{dataset}-RAW-{index:04d}")
    clause = re.sub(r"\s+", " ", str(raw.get("clause_text") or raw.get("text") or "")).strip()
    label = int(raw.get("label", 0))
    risk_type = str(raw.get("risk_type") or raw.get("category") or "generic")
    category = infer_category(risk_type, clause)
    anchor_date = extract_anchor_date(clause, original_id)
    sample_id = f"{dataset}-{index:04d}"
    contract_match = re.search(r"_c(\d+)_", original_id)
    contract_id = f"{dataset}-C{int(contract_match.group(1)):04d}" if contract_match else f"{dataset}-C{(index - 1) // 8 + 1:04d}"
    s1 = raw.get("s1_summary") or f"该条款主要约定：{clause[:120]}"
    s2 = raw.get("s2_trigger") or ("存在可能导致权利义务失衡、程序缺失或救济不足的风险触发点。" if label else "未发现明显的程序缺失、责任倒置或救济不足触发点。")
    s3 = raw.get("s3_legal_consequence") or ("该问题可能导致履约争议、责任承担范围不明或一方权利受损。" if label else "现有约定通常能够支持双方理解和履行，法律后果较为确定。")
    s4 = raw.get("s4_judgement") or ("综合判断，该条款存在风险。" if label else "综合判断，该条款不存在明显风险。")
    evidence_text = raw.get("evidence_text") or EVIDENCE_BY_CATEGORY.get(category, EVIDENCE_BY_CATEGORY["generic"])
    return {
        "sample_id": sample_id,
        "contract_id": contract_id,
        "dataset": dataset,
        "task": "contract_risk_review",
        "clause_text": clause,
        "anchor_date": anchor_date,
        "risk_type": risk_type,
        "risk_category": category,
        "label": label,
        "label_name": "风险条款" if label else "非风险条款",
        "gold_evidence_ids": find_evidence_ids(laws, category),
        "evidence_text": evidence_text,
        "review_steps": {
            "s1_summary": s1,
            "s2_trigger": s2,
            "s3_legal_consequence": s3,
            "s4_judgement": s4,
            "temporal_consequence": "以 anchor_date 为合同时间锚点，仅使用该日期有效的法律证据进行审查。",
        },
        "gold_legal_analysis": f"{s1} {s2} {s3} {s4}",
        "source_id": original_id,
        "source_dataset": raw.get("dataset"),
        "source_files": source_files,
        "label_source": "generated_compliant_negative_from_user_grassrisk" if generated else "user_uploaded_jsonl",
        "is_generated_supplement": generated,
        "text_language": detect_language(clause),
    }


def dedupe_rows(paths: list[Path], dataset_filter: str | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    rows_by_id: dict[str, dict] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for row in read_maybe_broken_jsonl(path):
            if dataset_filter and str(row.get("dataset")) != dataset_filter:
                continue
            key = str(row.get("id") or hashlib.md5(str(row.get("clause_text", "")).encode("utf-8")).hexdigest())
            sources[key].append(path.name)
            rows_by_id.setdefault(key, row)
    return list(rows_by_id.values()), sources


def sample_by_label(rows: list[dict], positive: int, negative: int, rng: random.Random) -> list[dict]:
    pos = [row for row in rows if int(row.get("label", 0)) == 1]
    neg = [row for row in rows if int(row.get("label", 0)) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < positive or len(neg) < negative:
        raise ValueError(f"not enough rows: pos {len(pos)}/{positive}, neg {len(neg)}/{negative}")
    selected = pos[:positive] + neg[:negative]
    rng.shuffle(selected)
    return selected


def make_compliant_negative(raw: dict, idx: int) -> dict:
    risk_type = str(raw.get("risk_type") or "合规补充条款")
    base_clause = re.sub(r"\s+", " ", str(raw.get("clause_text") or "")).strip()
    clause = (
        f"针对“{risk_type}”事项，双方确认：{base_clause[:80]}。"
        "该事项须依法经发包方书面同意、主管部门审批或备案、证据材料留存和补偿/整改程序确认后方可执行，任何一方不得单方免除法定义务。"
    )
    out = dict(raw)
    out["id"] = f"generated_grass_neg_{idx:03d}_from_{raw.get('id', 'unknown')}"
    out["clause_text"] = clause
    out["label"] = 0
    out["source"] = "generated_from_user_uploaded_grassrisk_positive"
    out["s1_summary"] = f"该条款围绕“{risk_type}”事项补充审批、备案、证据留存和补偿/整改程序。"
    out["s2_trigger"] = "条款保留书面同意、主管部门审批或备案、证据留存和法定义务，不构成明显风险触发点。"
    out["s3_legal_consequence"] = "该约定有助于降低履约争议和程序违法风险。"
    out["s4_judgement"] = "综合判断，该补充条款为非风险条款。"
    return out


def build_grass_source(rng: random.Random) -> tuple[list[dict], dict[str, list[str]], dict]:
    paths = [TEMP / "grassrisk_train.jsonl", TEMP / "grassrisk_dev.jsonl", TEMP / "grassrisk_test.jsonl"]
    rows, sources = dedupe_rows(paths, dataset_filter="GrassRisk")
    pos = [row for row in rows if int(row.get("label", 0)) == 1]
    neg = [row for row in rows if int(row.get("label", 0)) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    needed_generated_neg = max(0, GRASS_TARGET["negative"] - len(neg))
    generated = [make_compliant_negative(row, i + 1) for i, row in enumerate(pos[:needed_generated_neg])]
    for row in generated:
        sources[row["id"]] = ["generated_from_grassrisk_source"]
    selected_pos = pos[: GRASS_TARGET["positive"]]
    selected_neg = neg[: GRASS_TARGET["negative"] - needed_generated_neg] + generated
    selected = selected_pos + selected_neg
    rng.shuffle(selected)
    audit = {
        "raw_unique_grassrisk": len(rows),
        "raw_positive": len(pos),
        "raw_negative": len(neg),
        "generated_negative_supplements": needed_generated_neg,
    }
    return selected, sources, audit


def stratified_split(rows: list[dict], dataset: str, rng: random.Random) -> dict[str, list[dict]]:
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)
    for items in by_label.values():
        rng.shuffle(items)
    split_rows = {"train": [], "val": [], "test": []}
    for label, counts in SPLIT_COUNTS[dataset].items():
        cursor = 0
        for split in ["train", "val", "test"]:
            take = counts[split]
            split_rows[split].extend(by_label[label][cursor : cursor + take])
            cursor += take
    for split in split_rows:
        rng.shuffle(split_rows[split])
    return split_rows


def write_contracts(dataset: str, rows: list[dict]) -> Path:
    out_dir = ROOT / "data/raw/contracts_generated" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    by_contract: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_contract[row["contract_id"]].append(row)
    for contract_id, items in by_contract.items():
        lines = [
            f"合同编号：{contract_id}",
            f"数据集：{dataset}",
            "说明：该文件由用户上传 JSONL 条款重组生成，用于本地实验测试；不是外部新增真实合同原件。",
            "",
        ]
        for idx, item in enumerate(items, 1):
            lines.append(f"第{idx}条 [{item['sample_id']}] {item['clause_text']}")
            lines.append(f"标注：{item['label']}（{item['label_name']}）；风险类型：{item['risk_type']}；时间锚点：{item['anchor_date']}")
            lines.append("")
        (out_dir / f"{contract_id}.txt").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def write_dataset(dataset: str, raw_rows: list[dict], sources: dict[str, list[str]], laws: list[dict], rng: random.Random, extra_audit: dict) -> dict:
    out_dir = ROOT / "data/processed" / dataset
    split_raw = stratified_split(raw_rows, dataset, rng)
    normalized_by_split = {}
    all_rows = []
    index = 1
    for split in ["train", "val", "test"]:
        normalized = []
        for raw in split_raw[split]:
            row_id = str(raw.get("id") or raw.get("sample_id") or "")
            source_files = sources.get(row_id, ["unknown_or_generated"])
            row = normalize_row(raw, dataset, index, laws, source_files, generated=bool(str(row_id).startswith("generated_grass_neg_")))
            normalized.append(row)
            all_rows.append(row)
            index += 1
        normalized_by_split[split] = normalized
        write_jsonl(out_dir / f"{split}.jsonl", normalized)
    write_jsonl(out_dir / "all.jsonl", all_rows)

    annotations = [
        {
            "sample_id": row["sample_id"],
            "contract_id": row["contract_id"],
            "label": row["label"],
            "label_name": row["label_name"],
            "risk_type": row["risk_type"],
            "anchor_date": row["anchor_date"],
            "gold_evidence_ids": row["gold_evidence_ids"],
            "evidence_text": row["evidence_text"],
            "source_id": row["source_id"],
            "label_source": row["label_source"],
        }
        for row in all_rows
    ]
    annotation_path = ROOT / "data/raw/annotations" / f"{dataset}_annotations.jsonl"
    write_jsonl(annotation_path, annotations)
    contract_dir = write_contracts(dataset, all_rows)

    audit = {
        "dataset": dataset,
        **extra_audit,
        "total": len(all_rows),
        "label_counts": dict(Counter(row["label"] for row in all_rows)),
        "split_counts": {
            split: {
                "total": len(rows),
                "positive": sum(row["label"] for row in rows),
                "negative": len(rows) - sum(row["label"] for row in rows),
            }
            for split, rows in normalized_by_split.items()
        },
        "language_counts": dict(Counter(row["text_language"] for row in all_rows)),
        "generated_supplements": sum(1 for row in all_rows if row["is_generated_supplement"]),
        "processed_dir": str(out_dir),
        "annotations": str(annotation_path),
        "contracts_dir": str(contract_dir),
        "sample_rows": {
            split: normalized_by_split[split][:2]
            for split in ["train", "val", "test"]
        },
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def write_summary(audits: list[dict]) -> None:
    out_dir = ROOT / "outputs/dataset_audit/UserUploadedRiskDatasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# User Uploaded Risk Dataset Conversion", ""]
    for audit in audits:
        lines.append(f"## {audit['dataset']}")
        lines.append("")
        lines.append(f"- processed_dir: `{audit['processed_dir']}`")
        lines.append(f"- annotations: `{audit['annotations']}`")
        lines.append(f"- contracts_dir: `{audit['contracts_dir']}`")
        lines.append(f"- total: {audit['total']}")
        lines.append(f"- label_counts: {audit['label_counts']}")
        lines.append(f"- split_counts: {audit['split_counts']}")
        lines.append(f"- language_counts: {audit['language_counts']}")
        lines.append(f"- generated_supplements: {audit['generated_supplements']}")
        lines.append("")
        lines.append("### Sample")
        lines.append("```json")
        lines.append(json.dumps(audit["sample_rows"]["train"][0], ensure_ascii=False, indent=2)[:2500])
        lines.append("```")
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare user-uploaded CUADRisk and GrassRisk datasets for paper-aligned RAG experiments.")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    laws = load_laws()

    cuad_paths = [TEMP / "cuad_risk_train.jsonl", TEMP / "cuad_risk_dev.jsonl", TEMP / "cuad_risk_test.jsonl"]
    cuad_rows_all, cuad_sources = dedupe_rows(cuad_paths)
    cuad_selected = sample_by_label(cuad_rows_all, CUAD_TARGET["positive"], CUAD_TARGET["negative"], rng)
    cuad_audit = write_dataset(
        "CUADRiskUser",
        cuad_selected,
        cuad_sources,
        laws,
        rng,
        {
            "raw_unique": len(cuad_rows_all),
            "raw_label_counts": dict(Counter(int(row.get("label", 0)) for row in cuad_rows_all)),
            "target": CUAD_TARGET,
        },
    )

    grass_selected, grass_sources, grass_extra_audit = build_grass_source(rng)
    grass_audit = write_dataset(
        "GrassRiskUser",
        grass_selected,
        grass_sources,
        laws,
        rng,
        {
            **grass_extra_audit,
            "target": GRASS_TARGET,
        },
    )
    write_summary([cuad_audit, grass_audit])
    print(json.dumps([cuad_audit, grass_audit], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
