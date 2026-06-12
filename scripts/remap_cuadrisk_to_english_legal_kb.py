#!/usr/bin/env python
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "CUADRisk"
ANNOTATION_PATH = ROOT / "data" / "raw" / "annotations" / "CUADRisk_annotations.jsonl"
KB_PATH = ROOT / "data" / "raw" / "laws" / "en" / "cuadrisk_legal_validity_kb.en.jsonl"
MAPPING_PATH = ROOT / "data" / "raw" / "laws" / "en" / "cuadrisk_risk_type_to_law_ids.json"
AUDIT_DIR = ROOT / "outputs" / "dataset_audit" / "CUADRiskEnglishEvidenceRemap"


RISK_CATEGORY_FALLBACK = {
    "Agreement Date": "generic",
    "Document Name": "generic",
    "Effective Date": "term",
    "Expiration Date": "term",
    "Renewal Term": "term",
    "Post-Termination Services": "term",
    "Anti-Assignment": "transfer",
    "License Grant": "transfer",
    "Non-Transferable License": "transfer",
    "Exclusivity": "transfer",
    "Rofr/Rofo/Rofn": "transfer",
    "Affiliate License-Licensee": "transfer",
    "Affiliate License-Licensor": "transfer",
    "Minimum Commitment": "payment",
    "Revenue/Profit Sharing": "payment",
    "Most Favored Nation": "payment",
    "Price Restrictions": "payment",
    "Warranty Duration": "liability",
    "Uncapped Liability": "liability",
    "Liquidated Damages": "liability",
    "Insurance": "liability",
    "Governing Law": "dispute",
    "Third Party Beneficiary": "generic",
    "Audit Rights": "audit",
    "Source Code Escrow": "generic",
    "Non-Compete": "generic",
    "Non-Disparagement": "generic",
    "Competitive Restriction Exception": "generic",
    "Volume Restriction": "payment",
    "Unlimited/All-You-Can-Eat-License": "generic",
    "Parties": "generic",
}

CURRENT_YEAR = 2026


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def temporal_state(law: dict, anchor_value: str | None) -> str:
    anchor = parse_date(anchor_value)
    if anchor is None:
        return "unknown_anchor"
    start = parse_date(law.get("valid_from") or law.get("t_start"))
    end = parse_date(law.get("valid_to") or law.get("t_end")) or date(9999, 12, 31)
    if start and anchor < start:
        return "future_law"
    if anchor <= end:
        return "active_at_anchor"
    return "expired_at_anchor"


def extract_anchor_from_text(text: str) -> tuple[str | None, str]:
    patterns = [
        r"\b(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        r"\b(\d{1,2})/(\d{1,2})/(19\d{2}|20\d{2})\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(19\d{2}|20\d{2})\b",
    ]
    month_names = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, text or "")
        if not match:
            continue
        if idx == 0:
            y, m, d = match.groups()
        elif idx == 1:
            m, d, y = match.groups()
        else:
            month_name, d, y = match.groups()
            m = month_names[month_name]
        try:
            return date(int(y), int(m), int(d)).isoformat(), "extracted_from_clause_text"
        except ValueError:
            continue
    return None, "missing"


def is_suspicious_anchor(row: dict, anchor_value: str | None) -> bool:
    anchor = parse_date(anchor_value)
    if not anchor:
        return False
    year = anchor.year
    if year < 1980 or year > CURRENT_YEAR:
        return True
    text = row.get("clause_text", "")
    if re.search(rf"\b{year}\s+(Act|Code|Rule|Regulation)\b", text, flags=re.IGNORECASE):
        return True
    return False


def normalize_anchor(row: dict, contract_anchor_cache: dict[str, str]) -> tuple[str, str]:
    current = parse_date(row.get("anchor_date"))
    suspicious = is_suspicious_anchor(row, row.get("anchor_date"))
    if current and not suspicious:
        return current.isoformat(), "existing_anchor_date"
    extracted, source = extract_anchor_from_text(row.get("clause_text", ""))
    if extracted and not is_suspicious_anchor({"clause_text": row.get("clause_text", ""), "anchor_date": extracted}, extracted):
        return extracted, source
    contract_id = row.get("contract_id")
    if contract_id and contract_id in contract_anchor_cache:
        return contract_anchor_cache[contract_id], "imputed_from_same_contract_after_suspicious_or_missing_anchor"
    return "2020-01-01", "default_imputed_no_reliable_date_found"


def build_contract_anchor_cache(rows: list[dict]) -> dict[str, str]:
    anchors: dict[str, list[str]] = {}
    for row in rows:
        contract_id = row.get("contract_id")
        anchor = iso(parse_date(row.get("anchor_date")))
        if contract_id and anchor and not is_suspicious_anchor(row, anchor):
            anchors.setdefault(contract_id, []).append(anchor)
    cache = {}
    for contract_id, values in anchors.items():
        cache[contract_id] = Counter(values).most_common(1)[0][0]
    return cache


def evidence_priority(law: dict) -> tuple[int, int, str]:
    source_rank = 0
    if law.get("source_type", "").startswith("free_legal_reference"):
        source_rank = 1
    return (source_rank, 0 if law.get("status") == "effective" else 1, law["law_id"])


def select_evidence(row: dict, kb_by_id: dict[str, dict], mapping: dict[str, list[str]]) -> tuple[list[str], list[dict]]:
    risk_type = row.get("risk_type") or ""
    candidates = [kb_by_id[x] for x in mapping.get(risk_type, []) if x in kb_by_id]
    anchor = row.get("anchor_date")
    active = [law for law in candidates if temporal_state(law, anchor) == "active_at_anchor"]
    selected = sorted(active, key=evidence_priority)[:3]
    if not selected and candidates:
        selected = sorted(candidates, key=evidence_priority)[:1]
    return [law["law_id"] for law in selected], candidates


def make_evidence_text(selected_ids: list[str], kb_by_id: dict[str, dict]) -> str:
    parts = []
    for law_id in selected_ids:
        law = kb_by_id[law_id]
        parts.append(f"{law['law_name']} {law['article_no']}: {law['article_summary']}")
    return " ".join(parts)


def make_review_steps(row: dict, selected_ids: list[str], candidates: list[dict], kb_by_id: dict[str, dict]) -> dict:
    label = int(row.get("label", 0))
    anchor = row.get("anchor_date")
    selected_states = [temporal_state(kb_by_id[law_id], anchor) for law_id in selected_ids]
    inactive = [
        law["law_id"]
        for law in candidates
        if law["law_id"] not in selected_ids and temporal_state(law, anchor) != "active_at_anchor"
    ][:5]
    judgement = (
        f"The clause is labeled as a risk clause for `{row.get('risk_type')}` because the wording may create overbroad, unilateral, unclear, or non-compliant obligations."
        if label
        else f"The clause is labeled as a non-risk clause for `{row.get('risk_type')}` because the wording is not treated as triggering the mapped legal risk under the current annotation."
    )
    return {
        "s1_clause_summary": re.sub(r"\s+", " ", row.get("clause_text", "")).strip()[:260],
        "s2_risk_type": row.get("risk_type"),
        "s3_selected_legal_evidence": selected_ids,
        "s4_temporal_alignment": {
            "anchor_date": anchor,
            "selected_evidence_states": selected_states,
            "inactive_or_future_candidates_excluded": inactive,
        },
        "s5_evidence_use": "The selected evidence is used to judge whether the contract clause is risky while respecting the law validity period at the contract time anchor.",
        "s6_gold_judgement": judgement,
    }


def remap_row(row: dict, kb_by_id: dict[str, dict], mapping: dict[str, list[str]], contract_anchor_cache: dict[str, str]) -> dict:
    row = dict(row)
    original_anchor = row.get("anchor_date")
    anchor, anchor_source = normalize_anchor(row, contract_anchor_cache)
    if original_anchor != anchor:
        row["original_anchor_date"] = original_anchor
    row["anchor_date"] = anchor
    row["anchor_date_source"] = anchor_source
    row["risk_category"] = row.get("risk_category") or RISK_CATEGORY_FALLBACK.get(row.get("risk_type"), "generic")

    selected_ids, candidates = select_evidence(row, kb_by_id, mapping)
    row["gold_evidence_ids"] = selected_ids
    row["candidate_evidence_ids"] = [law["law_id"] for law in candidates]
    row["evidence_language"] = "en"
    row["legal_kb"] = "CUADRiskEnglishLegalKB"
    row["evidence_text"] = make_evidence_text(selected_ids, kb_by_id)
    row["gold_evidence_meta"] = [
        {
            "law_id": law_id,
            "law_name": kb_by_id[law_id]["law_name"],
            "article_no": kb_by_id[law_id]["article_no"],
            "valid_from": kb_by_id[law_id]["valid_from"],
            "valid_to": kb_by_id[law_id]["valid_to"],
            "status": kb_by_id[law_id]["status"],
            "temporal_state": temporal_state(kb_by_id[law_id], anchor),
            "source_url": kb_by_id[law_id]["source_url"],
        }
        for law_id in selected_ids
    ]
    row["temporal_candidate_states"] = [
        {
            "law_id": law["law_id"],
            "valid_from": law["valid_from"],
            "valid_to": law["valid_to"],
            "status": law["status"],
            "temporal_state": temporal_state(law, anchor),
        }
        for law in candidates
    ]
    row["review_steps"] = make_review_steps(row, selected_ids, candidates, kb_by_id)
    row["gold_legal_analysis"] = (
        f"CUADRisk English legal evidence remap. Risk type: {row.get('risk_type')}. "
        f"Anchor date: {anchor}. Selected evidence: {', '.join(selected_ids)}. "
        f"Gold label: {row.get('label')} ({row.get('label_name')}). "
        f"{row['evidence_text']}"
    )
    return row


def backup_file(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)


def main() -> None:
    kb_rows = read_jsonl(KB_PATH)
    kb_by_id = {row["law_id"]: row for row in kb_rows}
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    all_rows = read_jsonl(DATA_DIR / "all.jsonl")
    contract_anchor_cache = build_contract_anchor_cache(all_rows)

    backup_dir = AUDIT_DIR / "backup_before_english_remap"
    for name in ["all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "audit.json"]:
        backup_file(DATA_DIR / name, backup_dir)
    backup_file(ANNOTATION_PATH, backup_dir)

    stats = {}
    for split in ["all", "train", "val", "test"]:
        path = DATA_DIR / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows = [remap_row(row, kb_by_id, mapping, contract_anchor_cache) for row in rows]
        write_jsonl(path, new_rows)
        stats[split] = {
            "total": len(new_rows),
            "positive": sum(1 for row in new_rows if int(row.get("label", 0)) == 1),
            "negative": sum(1 for row in new_rows if int(row.get("label", 0)) == 0),
            "missing_anchor_date": sum(1 for row in new_rows if not row.get("anchor_date")),
            "missing_gold_evidence_ids": sum(1 for row in new_rows if not row.get("gold_evidence_ids")),
            "non_english_evidence_ids": sum(
                1
                for row in new_rows
                for law_id in row.get("gold_evidence_ids", [])
                if law_id not in kb_by_id
            ),
            "anchor_date_sources": dict(Counter(row.get("anchor_date_source") for row in new_rows)),
            "selected_temporal_states": dict(
                Counter(
                    meta["temporal_state"]
                    for row in new_rows
                    for meta in row.get("gold_evidence_meta", [])
                )
            ),
        }

    final_all = read_jsonl(DATA_DIR / "all.jsonl")
    write_jsonl(ANNOTATION_PATH, final_all)

    audit = {
        "dataset": "CUADRisk",
        "operation": "remap_gold_evidence_to_cuadrisk_english_legal_kb",
        "legal_kb": str(KB_PATH),
        "risk_type_mapping": str(MAPPING_PATH),
        "backup_dir": str(backup_dir),
        "stats": stats,
        "fields_added_or_updated": [
            "gold_evidence_ids",
            "candidate_evidence_ids",
            "evidence_text",
            "evidence_language",
            "legal_kb",
            "gold_evidence_meta",
            "temporal_candidate_states",
            "anchor_date",
            "anchor_date_source",
            "review_steps",
            "gold_legal_analysis",
        ],
        "note": "Gold evidence now points to the English CUADRisk legal validity KB. Evidence is selected by risk_type and contract anchor-date validity.",
    }
    (DATA_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "remap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
