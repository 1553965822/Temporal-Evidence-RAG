#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RISK_TYPES = [
    "承包期限",
    "流转限制",
    "生态保护",
    "用途变更",
    "补偿价款",
    "解除违约",
    "争议解决",
]

LAW_KEYWORDS = {
    "承包期限": ["承包期", "延包", "期限", "农村土地承包"],
    "流转限制": ["流转", "转包", "转让", "承包经营权"],
    "生态保护": ["草原", "生态", "修复", "载畜量"],
    "用途变更": ["草原", "改变用途", "建设", "采矿", "审批"],
    "补偿价款": ["价款", "补偿", "履行", "公平"],
    "解除违约": ["合同解除", "违约责任", "通知", "赔偿"],
    "争议解决": ["仲裁", "诉讼", "争议", "人民法院"],
}

EVIDENCE_TEXT = {
    "承包期限": "承包期限、续期和延包应当依据合同时间锚点适用现行有效法律，并完成必要审查。",
    "流转限制": "承包经营权流转、转包或转让应当符合法律规定和合同约定，不得跳过同意、备案或审查程序。",
    "生态保护": "草原经营利用应遵守生态保护、修复、禁牧休牧和载畜量控制义务。",
    "用途变更": "草原用途变更、建设、采矿和临时占用应依法取得批准，不得擅自改变用途。",
    "补偿价款": "价款、补偿和扣款调整应遵循公平、诚信、证据和书面确认程序。",
    "解除违约": "合同解除和违约责任应符合法定或约定条件，并保障通知、整改、证据审查和补偿程序。",
    "争议解决": "争议解决条款不得排除当事人依法申请仲裁、诉讼或其他法定救济的权利。",
}

TEMPLATES = {
    "承包期限": {
        1: {
            "train": [
                "第{section}条 承包期届满后，如发包方在{days}日内未提出异议，承包期限自动续期，无需重新审批或审查。",
                "第{section}条 乙方可在原期限届满后继续经营，续期手续可由乙方单方确认，主管机关审查不是生效条件。",
                "第{section}条 合同期限可连续顺延，双方确认无需另行进行法定延包审查。",
            ],
            "val": [
                "第{section}条 承包期限届满后默示续期，未取得书面批准不影响乙方继续使用草原。",
                "第{section}条 乙方提交续期申请即视为续期完成，审批和审查材料可事后补办。",
            ],
            "test": [
                "第{section}条 承包期满后自动延续，未完成法定审查或书面批准仍视为有效续期。",
                "第{section}条 发包方未即时回复的，承包期自动顺延，无需重新办理延包审批。",
            ],
        },
        0: {
            "train": [
                "第{section}条 承包期限届满后，续期须经书面申请、法定审查和主管机关批准后方可生效。",
                "第{section}条 双方可以协商延包，但未经书面批准和审查，不得继续占用草原。",
                "第{section}条 续期仅在完成审批、备案和合同补充协议后生效。",
            ],
            "val": [
                "第{section}条 承包期届满不当然续期，乙方继续经营须取得书面批准并完成审查。",
                "第{section}条 延包事项应依法办理审批和备案，沉默不构成续期同意。",
            ],
            "test": [
                "第{section}条 承包期不自动续期，继续经营须先完成法定审查和书面批准。",
                "第{section}条 双方可协商续期，但未取得批准前不得视为期限延长。",
            ],
        },
    },
    "流转限制": {
        1: {
            "train": [
                "第{section}条 乙方可将承包经营权转包或转让给第三方，无需发包方书面同意或备案。",
                "第{section}条 乙方关联公司可直接承接受让本合同权利义务，发包方不得拒绝且无需审查。",
                "第{section}条 转包事项由乙方自主决定，未通知发包方不影响流转效力。",
            ],
            "val": [
                "第{section}条 乙方认为经营需要时，可不经同意将草原经营权整体转让。",
                "第{section}条 第三方接管经营无需登记，发包方收到通知即视为同意。",
            ],
            "test": [
                "第{section}条 乙方可不经书面同意、审批或备案，将承包经营权转让给第三方。",
                "第{section}条 第三方可直接接管合同义务，发包方无需完成任何同意审查。",
            ],
        },
        0: {
            "train": [
                "第{section}条 承包经营权流转须经发包方书面同意，并依法完成备案或登记。",
                "第{section}条 乙方拟转包的，应提前提交方案，经审查同意后方可实施。",
                "第{section}条 任何转让不得损害草原用途和生态保护义务，且须履行书面程序。",
            ],
            "val": [
                "第{section}条 乙方可申请转包，但未经书面同意和备案不得生效。",
                "第{section}条 流转事项应保留审批、备案和生态保护承诺。",
            ],
            "test": [
                "第{section}条 未经书面同意和法定备案，任何一方不得转让合同权利义务。",
                "第{section}条 承包经营权流转须经过发包方审查同意并完成登记手续。",
            ],
        },
    },
    "生态保护": {
        1: {
            "train": [
                "第{section}条 因收益目标需要，乙方可暂缓履行草原修复、监测和载畜量控制义务。",
                "第{section}条 发生草原退化时，修复费用由发包方承担，乙方无需承担生态损害责任。",
                "第{section}条 乙方可在旺季暂停生态监测，相关恢复义务不影响继续经营。",
            ],
            "val": [
                "第{section}条 草原生态修复可无限期顺延，乙方仅在自愿时承担费用。",
                "第{section}条 超载放牧造成损害的，乙方无需立即整改或修复。",
            ],
            "test": [
                "第{section}条 未经第三方生态评估，乙方仍可免除草原修复费用和监测义务。",
                "第{section}条 生产指标优先于生态保护要求，乙方可暂停载畜量控制。",
            ],
        },
        0: {
            "train": [
                "第{section}条 乙方应遵守草原生态保护、修复、监测和载畜量控制要求。",
                "第{section}条 发生生态损害的，乙方应按审查意见及时整改并承担修复费用。",
                "第{section}条 经营活动不得突破禁牧休牧、修复和监测要求。",
            ],
            "val": [
                "第{section}条 乙方可以继续经营，但必须履行生态监测和草原修复义务。",
                "第{section}条 草原损害经确认后，乙方应承担整改和修复责任。",
            ],
            "test": [
                "第{section}条 草原生态义务不得免除，经营活动应符合监测和修复要求。",
                "第{section}条 乙方应在确认损害后及时整改，并继续遵守载畜量控制。",
            ],
        },
    },
    "用途变更": {
        1: {
            "train": [
                "第{section}条 乙方可自行将草原改作旅游、仓储或建设用地，无需主管机关批准。",
                "第{section}条 临时设施可长期使用，乙方无需另行办理用途变更审批。",
                "第{section}条 采挖、取土或建设活动可由乙方内部审批后实施。",
            ],
            "val": [
                "第{section}条 草原可直接转为商业设施用地，审批和公示程序可免除。",
                "第{section}条 乙方改变草原用途时，仅需向发包方口头告知。",
            ],
            "test": [
                "第{section}条 乙方可未经审批、审查或备案，将草原用途改为旅游建设。",
                "第{section}条 采挖和建设活动可直接开展，无需取得单独法定许可。",
            ],
        },
        0: {
            "train": [
                "第{section}条 草原用途变更须依法取得批准，并完成土地用途和生态影响审查。",
                "第{section}条 未经许可，乙方不得建设、采挖、取土或改变草原用途。",
                "第{section}条 临时设施应符合法定用途，不得规避审批程序。",
            ],
            "val": [
                "第{section}条 商业利用须取得书面批准和用途变更备案后实施。",
                "第{section}条 乙方不得擅自改变草原用途，确需变更的应依法审批。",
            ],
            "test": [
                "第{section}条 未经批准和用途审查，乙方不得改变草原用途。",
                "第{section}条 新建设施应先取得许可，确认不违反草原用途管制。",
            ],
        },
    },
    "补偿价款": {
        1: {
            "train": [
                "第{section}条 发包方可单方调整承包费和补偿标准，无需提供测算依据或书面确认。",
                "第{section}条 发包方认为履约不充分时，可直接扣除全部补偿，乙方不得异议。",
                "第{section}条 价款调整由发包方内部决定，乙方放弃未来补偿请求。",
            ],
            "val": [
                "第{section}条 发包方可凭内部意见扣减补偿，无需证据、通知或复核。",
                "第{section}条 承包费可即时调整，不需要双方确认或计算说明。",
            ],
            "test": [
                "第{section}条 发包方可未经书面同意和计算复核，单方提高或扣减承包费。",
                "第{section}条 全部补偿可立即扣除，无需损失证据或争议处理程序。",
            ],
        },
        0: {
            "train": [
                "第{section}条 承包费和补偿调整须经双方书面确认，并附计算依据。",
                "第{section}条 扣款应以实际损失证据为基础，并履行通知和复核程序。",
                "第{section}条 价款变更不得免除依法享有的补偿权利。",
            ],
            "val": [
                "第{section}条 发包方仅可在证据充分并通知乙方后扣除实际损失。",
                "第{section}条 费用调整应经双方书面确认并保留测算材料。",
            ],
            "test": [
                "第{section}条 未经书面协议和计算依据，任何一方不得变更承包费。",
                "第{section}条 补偿调整应经过证据审查和双方确认后实施。",
            ],
        },
    },
    "解除违约": {
        1: {
            "train": [
                "第{section}条 发包方可随时解除合同，无需通知、整改期限或损失评估。",
                "第{section}条 乙方轻微迟延即构成重大违约，发包方可立即解除且不予补偿。",
                "第{section}条 发包方认定风险增加即可单方解除，乙方放弃申辩和整改权利。",
            ],
            "val": [
                "第{section}条 合同可不经通知和整改程序直接终止，补偿由发包方决定。",
                "第{section}条 任一违约线索均允许发包方立即解除，无需审查重大性。",
            ],
            "test": [
                "第{section}条 发包方可未经通知、整改机会或补偿审查直接解除合同。",
                "第{section}条 未证明重大违约时，发包方仍可取消合同并拒绝补偿。",
            ],
        },
        0: {
            "train": [
                "第{section}条 合同解除应以违约证据为基础，并给予通知和合理整改期限。",
                "第{section}条 解除合同前应审查违约程度、损失和补偿安排。",
                "第{section}条 立即解除仅限法定紧急情形，并应事后完成审查。",
            ],
            "val": [
                "第{section}条 发包方解除合同前应书面通知乙方并给予整改机会。",
                "第{section}条 无违约证据和合同审查，不得直接终止承包关系。",
            ],
            "test": [
                "第{section}条 合同解除应经过通知、整改期限、法律审查和补偿处理。",
                "第{section}条 发包方不得在未证明重大违约时取消合同。",
            ],
        },
    },
    "争议解决": {
        1: {
            "train": [
                "第{section}条 发包方内部决定为最终结论，乙方不得申请仲裁或诉讼。",
                "第{section}条 双方争议由发包方单方解释，乙方放弃司法和仲裁救济。",
                "第{section}条 乙方签约后不得向法院或仲裁机构提出任何合同争议。",
            ],
            "val": [
                "第{section}条 一切争议以发包方最终意见为准，不接受外部审查。",
                "第{section}条 乙方不得就本合同申请仲裁、诉讼或其他法定救济。",
            ],
            "test": [
                "第{section}条 发包方内部处理意见为最终决定，乙方不得仲裁、诉讼或申请独立审查。",
                "第{section}条 乙方放弃向法院或仲裁机构挑战争议事项的权利。",
            ],
        },
        0: {
            "train": [
                "第{section}条 双方争议可先协商，协商不成的可依法申请仲裁或诉讼。",
                "第{section}条 内部复核不排除任何一方依法寻求司法或仲裁救济。",
                "第{section}条 合同解释争议应保留协商、仲裁和诉讼路径。",
            ],
            "val": [
                "第{section}条 争议经协商未解决的，任一方可提交约定仲裁机构或法院。",
                "第{section}条 发包方内部意见不得限制乙方依法申请外部救济。",
            ],
            "test": [
                "第{section}条 争议可协商解决，协商不成的保留仲裁和诉讼权利。",
                "第{section}条 任何内部审查结论均不排除法定救济途径。",
            ],
        },
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_evidence_ids(laws: list[dict], risk_type: str, top_k: int = 3) -> list[str]:
    keywords = LAW_KEYWORDS[risk_type]
    scored = []
    for row in laws:
        text = f"{row.get('law_name', '')} {row.get('article_no', '')} {row.get('article_text', '')}"
        score = sum(1 for kw in keywords if kw in text)
        if score:
            active_bonus = 1 if not row.get("valid_to") else 0
            scored.append((score, active_bonus, row.get("law_id", "")))
    scored.sort(reverse=True)
    return [law_id for _, _, law_id in scored[:top_k] if law_id]


def make_rows(split: str, count: int, laws: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    anchors = ["2012-06-01", "2017-05-01", "2019-06-01", "2021-02-01", "2024-01-15"]
    pairs = [(risk, label) for risk in RISK_TYPES for label in [0, 1]]
    for idx in range(count):
        risk_type, label = pairs[idx % len(pairs)]
        template = rng.choice(TEMPLATES[risk_type][label][split])
        anchor_date = anchors[(idx + rng.randrange(len(anchors))) % len(anchors)]
        section = rng.randint(2, 28)
        days = rng.choice([3, 5, 7, 10, 15])
        clause = template.format(section=section, days=days)
        clause = f"{clause} 本条签约时间锚点为{anchor_date}，适用签约时有效法律。"
        if idx % 11 == 0:
            clause += " 双方均应保留审批、备案、证据和通知材料。"
        if idx % 17 == 0:
            clause += " 本条同时提及协商、书面同意和复核程序，以避免单一关键词决定标签。"
        rows.append(
            {
                "sample_id": f"CUADRiskTemporalCN-{split.upper()}-{idx + 1:04d}",
                "contract_id": f"CUADRiskTemporalCN-{split.upper()}-C{idx // 5 + 1:04d}",
                "dataset": "CUADRiskTemporalCN",
                "task": "contract_risk_review",
                "clause_text": clause,
                "anchor_date": anchor_date,
                "risk_type": risk_type,
                "label": label,
                "label_name": "风险条款" if label else "非风险条款",
                "gold_evidence_ids": find_evidence_ids(laws, risk_type),
                "evidence_text": EVIDENCE_TEXT[risk_type],
                "review_steps": {
                    "evidence_summary": f"核心证据涉及{risk_type}。",
                    "clause_evidence_alignment": "检查条款是否保留审批、通知、证据、整改、备案或救济程序。",
                    "temporal_consequence": "以合同时间锚点筛选当时有效的法律版本，避免引用已失效或尚未生效规则。",
                },
                "label_source": "temporal_cn_rule_generated",
                "hardening_note": "中文条款与中文真实法规同语种；正负例共享审批、备案、同意、审查等词，以降低关键词泄漏。",
            }
        )
    rng.shuffle(rows)
    return rows


def audit(rows_by_split: dict[str, list[dict]]) -> dict:
    audit_data = {}
    for split, rows in rows_by_split.items():
        labels = [int(row["label"]) for row in rows]
        audit_data[split] = {
            "count": len(rows),
            "positive": sum(labels),
            "negative": len(labels) - sum(labels),
            "anchors": sorted({row["anchor_date"] for row in rows}),
        }
    train_texts = {row["clause_text"] for row in rows_by_split["train"]}
    for split in ["val", "test"]:
        audit_data[f"exact_overlap_train_{split}"] = len(train_texts & {row["clause_text"] for row in rows_by_split[split]})
    return audit_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="CUADRiskTemporalCN")
    parser.add_argument("--train", type=int, default=607)
    parser.add_argument("--val", type=int, default=202)
    parser.add_argument("--test", type=int, default=203)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    laws = load_jsonl(ROOT / "data/raw/laws/legal_validity_kb.jsonl")
    out_dir = ROOT / "data/processed" / args.output
    rows_by_split = {
        "train": make_rows("train", args.train, laws, args.seed + 1),
        "val": make_rows("val", args.val, laws, args.seed + 2),
        "test": make_rows("test", args.test, laws, args.seed + 3),
    }
    for split, rows in rows_by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    audit_data = audit(rows_by_split)
    (out_dir / "audit.json").write_text(json.dumps(audit_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "audit": audit_data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
