#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RISK_TYPES = ["期限超限", "转包转让限制", "生态保护义务", "用途变更", "补偿与价款", "解除与违约", "争议解决"]

LAW_BY_RISK = {
    "期限超限": "LAW-GRASS-CONTRACT-TERM",
    "转包转让限制": "LAW-GRASS-TRANSFER",
    "生态保护义务": "LAW-GRASS-ECO",
    "用途变更": "LAW-GRASS-USE",
    "补偿与价款": "LAW-CIVIL-2021-509",
    "解除与违约": "LAW-CIVIL-2021-563",
    "争议解决": "LAW-CIVIL-2021-509",
}

EVIDENCE_TEXT = {
    "期限超限": "承包期限和续期应当依法审批，并受法律规定和主管机关审查约束。",
    "转包转让限制": "草原承包经营权流转应当符合法律规定，不得损害发包方和草原生态保护要求。",
    "生态保护义务": "经营利用草原应当遵守生态保护、修复和载畜量控制等义务。",
    "用途变更": "草原用途变更、建设、采挖等活动应依法取得批准。",
    "补偿与价款": "当事人应当依约全面履行，价款和补偿变更应遵循公平、诚信和约定程序。",
    "解除与违约": "合同解除和违约责任应当符合法定或约定条件，并保障通知、整改和补偿程序。",
    "争议解决": "争议解决条款不得排除当事人依法申请仲裁或诉讼的权利。",
}

SPLIT_VARIANTS = {
    "期限超限": {
        1: {
            "train": [
                "The renewal may take effect automatically without a separate statutory review once the operator sends notice.",
                "The term may continue after expiry unless the authority objects, with no new approval required.",
                "The operator may extend the lease for successive periods without submitting renewal materials.",
                "A silence-by-default mechanism treats the term as renewed even when the statutory review is unfinished.",
            ],
            "val": [
                "The agreement may roll over without a fresh public review if no written objection is received in five days.",
                "The contractor may keep operating after the fixed term while approval documents are still pending.",
            ],
            "test": [
                "The term may be extended automatically without renewed statutory examination or written approval.",
                "If the authority does not reply immediately, the lease is deemed renewed without any formal review.",
            ],
        },
        0: {
            "train": [
                "The term may renew only after written approval and statutory review are completed.",
                "No renewal may occur without a fresh approval file and a documented review process.",
                "The operator may apply for renewal, but the term ends unless the authority issues written approval.",
                "Automatic renewal is not permitted without statutory review and signed confirmation.",
            ],
            "val": [
                "The parties may discuss renewal, but no continued term is valid without written approval.",
                "Any extension requires statutory review before the operator may continue using the land.",
            ],
            "test": [
                "The term does not renew automatically; written approval and review must occur first.",
                "No party may rely on silence as approval without completing the statutory renewal review.",
            ],
        },
    },
    "转包转让限制": {
        1: {
            "train": [
                "Either party may assign all obligations without prior written consent from the other party.",
                "The operator may subcontract the licensed area without notifying the owner or obtaining approval.",
                "Transfer is deemed approved if the owner does not object within one working day.",
                "The contractor may replace the operating entity at its sole discretion without registration.",
            ],
            "val": [
                "The rights may be transferred to an affiliate without written consent or public filing.",
                "Subcontracting may proceed without prior review when the contractor considers it commercially useful.",
            ],
            "test": [
                "The operator may assign the contract without written consent, approval, or registration.",
                "A third party may take over the obligations without the owner completing any consent review.",
            ],
        },
        0: {
            "train": [
                "Assignment may occur only with prior written consent and required registration.",
                "No subcontract may proceed without approval from the owner and compliance review.",
                "The operator may nominate an affiliate, but transfer is invalid without written consent.",
                "Change of control requires notice, consent, and all legally required filings.",
            ],
            "val": [
                "The contractor may discuss transfer, but no assignment is effective without prior written consent.",
                "Any subcontract is subject to approval and does not waive statutory filing duties.",
            ],
            "test": [
                "No party may assign obligations without written consent and completion of the review procedure.",
                "Transfer may be approved only after the owner gives written consent and required filings are made.",
            ],
        },
    },
    "生态保护义务": {
        1: {
            "train": [
                "The operator may ignore remediation duties when environmental work affects expected profit.",
                "Environmental monitoring may be suspended without notice during high-yield operating periods.",
                "The contractor is not responsible for restoration costs even when overuse damages the land.",
                "The owner waives all ecological claims unless damage is confirmed by the operator itself.",
            ],
            "val": [
                "Remediation may be delayed indefinitely without penalty if the operator reports financial pressure.",
                "The operator may exceed ecological limits without immediate restoration obligations.",
            ],
            "test": [
                "The operator may avoid remediation costs without independent environmental review.",
                "Ecological duties are suspended whenever production targets would otherwise be reduced.",
            ],
        },
        0: {
            "train": [
                "The operator may use the land only while following environmental duties and remediation plans.",
                "No activity may continue without compliance with monitoring and restoration requirements.",
                "The contractor must repair ecological damage and may not waive statutory duties.",
                "Operations are permitted only if environmental review and remediation records are maintained.",
            ],
            "val": [
                "The operator may continue production, but only without breaching ecological protection duties.",
                "Remediation costs remain the contractor's responsibility when damage is confirmed.",
            ],
            "test": [
                "No ecological obligation is waived without lawful review and written approval.",
                "The operator may conduct activities only after satisfying monitoring and restoration duties.",
            ],
        },
    },
    "用途变更": {
        1: {
            "train": [
                "The licensee may change the permitted use at its sole discretion without approval.",
                "Tourism construction may begin without land-use review if temporary facilities are used.",
                "The operator may conduct extraction activities without a separate permit from the authority.",
                "Use restrictions are waived whenever the contractor gives internal notice.",
            ],
            "val": [
                "The area may be converted to commercial facilities without written approval or statutory review.",
                "The licensee may change the land purpose without filing new use documents.",
            ],
            "test": [
                "The licensee may convert the area without approval, review, or public filing.",
                "Extraction and tourism use may proceed without a separate statutory permit.",
            ],
        },
        0: {
            "train": [
                "Use changes may occur only after written approval and land-use compliance review.",
                "No construction or extraction may begin without a separate statutory permit.",
                "The licensee may request a new use, but approval is required before implementation.",
                "Temporary facilities are allowed only when legal review confirms the permitted use.",
            ],
            "val": [
                "The land may be used commercially only after approval and registration are completed.",
                "No change of purpose is effective without written approval from the competent authority.",
            ],
            "test": [
                "The licensee may not change use without approval and statutory review.",
                "Any new facility requires permit review before the land purpose may change.",
            ],
        },
    },
    "补偿与价款": {
        1: {
            "train": [
                "Fees may be changed unilaterally with immediate effect and without supporting calculation.",
                "The owner may deduct all compensation without notice if it considers performance unsatisfactory.",
                "Payment standards may be revised by one party without review or mutual confirmation.",
                "The contractor waives compensation for all future adjustments, including unlawful deductions.",
            ],
            "val": [
                "Compensation may be withheld without evidence when the owner issues an internal decision.",
                "One party may change the fee schedule without consent, notice, or review.",
            ],
            "test": [
                "The owner may unilaterally change fees without written consent or calculation review.",
                "All compensation may be deducted immediately without evidence or dispute procedure.",
            ],
        },
        0: {
            "train": [
                "Fees may be adjusted only by written agreement and documented calculation.",
                "No compensation may be deducted without evidence, notice, and contractual review.",
                "The parties may revise the schedule after mutual confirmation and lawful disclosure.",
                "Payment changes require written consent and do not waive statutory compensation rights.",
            ],
            "val": [
                "The owner may deduct proven losses only after notice and review of supporting evidence.",
                "Fee changes are effective only after both parties approve the written schedule.",
            ],
            "test": [
                "No party may change fees without written agreement and supporting calculation.",
                "Compensation may be adjusted only after evidence review and mutual confirmation.",
            ],
        },
    },
    "解除与违约": {
        1: {
            "train": [
                "Termination may occur without notice or cure period at the owner's sole discretion.",
                "The owner may terminate immediately without compensation even for minor delay.",
                "Any alleged breach allows unilateral termination without review of materiality.",
                "The contractor waives all cure rights and compensation after internal notice.",
            ],
            "val": [
                "The agreement may be cancelled without notice, cure period, or loss assessment.",
                "One party may terminate without review whenever it believes risk has increased.",
            ],
            "test": [
                "Termination may occur without notice, cure opportunity, or compensation review.",
                "The owner may cancel the contract without proving material breach or giving cure time.",
            ],
        },
        0: {
            "train": [
                "Termination may occur only after notice, cure period, and applicable legal review.",
                "No party may terminate without first giving notice and a reasonable cure opportunity.",
                "Immediate termination is allowed only for lawful emergency reasons and later review.",
                "Material breach must be documented before termination and compensation decisions.",
            ],
            "val": [
                "The owner may terminate after written notice and failure to cure within the agreed period.",
                "No cancellation is effective without breach evidence and contractual review.",
            ],
            "test": [
                "Termination follows notice, cure period, legal review, and compensation rules.",
                "The owner may not cancel without material breach evidence and a cure opportunity.",
            ],
        },
    },
    "争议解决": {
        1: {
            "train": [
                "One party has final and non-reviewable dispute authority without court or arbitration access.",
                "The contractor waives all litigation and arbitration rights for future disputes.",
                "Disputes are resolved by the owner's internal decision without external review.",
                "The owner may interpret the contract finally and the operator may not challenge it.",
            ],
            "val": [
                "All disputes are subject to one party's final decision without judicial review.",
                "The contractor may not seek arbitration or court review after signing.",
            ],
            "test": [
                "One party's internal decision is final without arbitration, litigation, or independent review.",
                "The contractor waives the right to challenge disputes before a court or arbitral body.",
            ],
        },
        0: {
            "train": [
                "Disputes may be submitted to the agreed court or arbitration body after negotiation.",
                "No internal review decision prevents either party from seeking lawful dispute resolution.",
                "The parties may negotiate first, but litigation and arbitration rights are preserved.",
                "Contract interpretation may be discussed internally without waiving judicial review.",
            ],
            "val": [
                "The owner may give an internal opinion, but it is not final without legal review.",
                "Either party may submit unresolved disputes to arbitration or court.",
            ],
            "test": [
                "Internal review does not waive either party's right to arbitration or litigation.",
                "Disputes may proceed to court or arbitration if negotiation fails.",
            ],
        },
    },
}


def load_laws() -> dict[str, dict]:
    path = ROOT / "data/raw/laws/legal_validity_kb.jsonl"
    if not path.exists():
        return {}
    laws = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        laws[row.get("law_id")] = row
    return laws


def make_row(dataset: str, split: str, idx: int, risk_type: str, label: int, text: str, laws: dict[str, dict], rng: random.Random) -> dict:
    law_id = LAW_BY_RISK.get(risk_type)
    law = laws.get(law_id, {})
    return {
        "sample_id": f"{dataset}-{split.upper()}-{idx:04d}",
        "contract_id": f"{dataset}-{split.upper()}-C{idx // 5 + 1:04d}",
        "dataset": dataset,
        "task": "contract_risk_review",
        "clause_text": f"Section {idx % 37 + 1}. {text}",
        "anchor_date": rng.choice(["2019-06-01", "2021-02-01", "2024-01-15"]),
        "risk_type": risk_type,
        "label": label,
        "label_name": "风险条款" if label else "非风险条款",
        "gold_evidence_ids": [law_id] if law_id else [],
        "evidence_text": law.get("article_text") or EVIDENCE_TEXT[risk_type],
        "review_steps": {
            "evidence_summary": f"核心证据涉及{risk_type}。",
            "clause_evidence_alignment": "条款与法律依据存在对应关系。" if label else "条款保留了必要审批、通知或救济程序。",
            "temporal_consequence": "需结合合同时间锚点判断法律效力。",
        },
        "label_source": "rule_hardened_cuadrisk",
        "hardening_note": "Positive and negative clauses share overlapping legal keywords; templates are split-specific to reduce exact text leakage.",
    }


def build_split(dataset: str, split: str, total: int, positives: int, laws: dict[str, dict], rng: random.Random) -> list[dict]:
    rows = []
    labels = [1] * positives + [0] * (total - positives)
    rng.shuffle(labels)
    counters = {(risk_type, label): 0 for risk_type in RISK_TYPES for label in [0, 1]}
    for idx, label in enumerate(labels, 1):
        risk_type = RISK_TYPES[(idx + (0 if label else 3)) % len(RISK_TYPES)]
        variants = SPLIT_VARIANTS[risk_type][label][split]
        pos = counters[(risk_type, label)]
        text = variants[pos % len(variants)]
        counters[(risk_type, label)] += 1
        rows.append(make_row(dataset, split, idx, risk_type, label, text, laws, rng))
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a harder CUADRisk split with reduced template leakage.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/CUADRiskHard")
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    laws = load_laws()
    splits = {
        "train": build_split("CUADRiskHard", "train", 607, 298, laws, rng),
        "val": build_split("CUADRiskHard", "val", 202, 99, laws, rng),
        "test": build_split("CUADRiskHard", "test", 203, 100, laws, rng),
    }
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    all_rows = splits["train"] + splits["val"] + splits["test"]
    write_jsonl(args.output_dir / "all.jsonl", all_rows)
    meta = {
        "name": "CUADRiskHard",
        "source": "Synthetic CUAD-style hard split rebuilt from CUADRisk after leakage audit.",
        "total": len(all_rows),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "positive": sum(row["label"] for row in all_rows),
        "negative": len(all_rows) - sum(row["label"] for row in all_rows),
        "leakage_controls": [
            "No split shares the same base template variant.",
            "Positive and negative examples both contain decoy words such as may, without, approval, review, consent.",
            "Labels should not be inferred from one obvious token.",
        ],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
