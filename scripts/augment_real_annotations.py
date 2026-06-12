#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RISK_TARGET_TOTAL = 312
NEGATIVE_TARGET_TOTAL = 308
SEED = 20260521


RISK_TEMPLATES = [
    {
        "risk_type": "旧法依据适用风险",
        "temporal_label": 1,
        "temporal_error_type": "expired_law_reference",
        "keywords": ["施行", "有效", "民法典", "草原法", "承包"],
        "clause": "本合同签订及履行仍以已失效或者已被新法调整的旧版草原承包规定为唯一依据，双方不得援引现行有效法律提出异议。",
        "analysis": "条款把旧版法律规范作为唯一审查依据，排除了合同时间锚点对应的现行有效规则，属于论文实验中需要识别的法律时效错配风险。",
    },
    {
        "risk_type": "任意解除风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["解除", "合同", "违约", "承包"],
        "clause": "甲方可根据经营管理需要随时单方解除合同，乙方不得要求说明理由、继续履行或者请求赔偿。",
        "analysis": "条款赋予发包方任意解除权并排除承包方救济，明显突破依法解除和违约责任边界，存在合同解除风险。",
    },
    {
        "risk_type": "违法收回和程序缺失风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["收回", "承包", "程序", "补偿"],
        "clause": "甲方认为需要调整草原用途时，可以直接收回全部承包草原，无需履行通知、协商、审批或者补偿程序。",
        "analysis": "条款允许发包方绕过法定或者约定程序收回承包草原，并排除补偿安排，损害承包方稳定经营权益。",
    },
    {
        "risk_type": "承包期限超期风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["期限", "承包", "三十年", "五十年"],
        "clause": "承包期限为八十年，自合同生效之日起计算；期间不因法律政策调整而变更。",
        "analysis": "草地承包期限通常应落在法定合理区间内，八十年约定明显偏离期限规则，存在期限无效或者调整风险。",
    },
    {
        "risk_type": "流转期限超期风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["流转", "期限", "转包", "承包"],
        "clause": "乙方可将草原经营权一次性转包三十五年，即使超过原承包合同剩余期限，甲方也不得干预。",
        "analysis": "流转期限不得超过原承包关系剩余期限，该条款直接允许超期流转，存在流转效力风险。",
    },
    {
        "risk_type": "用途变更风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["用途", "建设", "草原", "保护"],
        "clause": "乙方可自行将承包草原改作仓储、旅游设施、采砂取土等非畜牧经营用途，无需另行审批。",
        "analysis": "条款允许擅自改变草原用途并规避审批，可能违反草原保护和土地用途管制要求。",
    },
    {
        "risk_type": "开垦草原风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["开垦", "草原", "保护", "生态"],
        "clause": "为提高收益，乙方可以自行开垦天然草原、翻耕种植经济作物，相关后果由乙方自行承担。",
        "analysis": "条款许可开垦天然草原，可能违反草原保护禁止性规范，属于生态保护高风险条款。",
    },
    {
        "risk_type": "补偿权益排除风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["征收", "征用", "补偿", "承包"],
        "clause": "因征收、征用或者政策调整产生的全部补偿款归甲方所有，乙方不得主张地上附着物、投入或者经营损失补偿。",
        "analysis": "条款概括排除承包方依法可能享有的补偿权益，容易造成补偿分配和投入返还争议。",
    },
    {
        "risk_type": "违约责任过重风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["违约", "赔偿", "损失", "责任"],
        "clause": "乙方迟延支付任何一期费用的，应按承包总价款五倍向甲方支付违约金，甲方无需证明实际损失。",
        "analysis": "违约金与可能损失明显不成比例，且免除损失证明要求，存在违约责任畸重和被调减风险。",
    },
    {
        "risk_type": "争议解决限制风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["争议", "仲裁", "诉讼", "调解"],
        "clause": "合同履行发生争议时，乙方只能接受甲方所在地村委会作出的最终处理意见，不得申请仲裁或者向人民法院起诉。",
        "analysis": "条款限制当事人依法申请调解、仲裁或者诉讼的救济渠道，存在争议解决条款无效风险。",
    },
    {
        "risk_type": "任意再流转风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["流转", "再流转", "同意", "登记"],
        "clause": "乙方取得经营权后可自行再转包、出租或者入股，无需征得甲方同意，也无需办理备案登记手续。",
        "analysis": "条款排除发包方同意、备案登记和用途审查要求，容易产生再流转效力和监管缺失风险。",
    },
    {
        "risk_type": "永久建设和用途变更风险",
        "temporal_label": 0,
        "temporal_error_type": "valid_or_not_applicable",
        "keywords": ["建设", "永久", "用途", "草原"],
        "clause": "乙方可在草原上建设永久性建筑物并长期占用，合同期满后建筑物归乙方所有且无需恢复草原原状。",
        "analysis": "条款允许永久建设并排除恢复义务，可能违反草原用途管制、生态修复和承包期满返还要求。",
    },
]


SAFE_TEMPLATES = [
    {
        "risk_type": "承包标的约定",
        "keywords": ["承包", "草原", "面积", "权利"],
        "clause": "双方确认承包草原的位置、四至、面积和权利性质以附件及登记资料为准，乙方依法享有合同约定的承包经营权益。",
        "analysis": "条款明确标的范围和权利边界，未排除承包方依法享有的经营权利，风险较低。",
    },
    {
        "risk_type": "承包期限约定",
        "keywords": ["期限", "承包", "三十年", "五十年"],
        "clause": "承包期限在依法允许的期限范围内确定，届满后的续包、调整或者终止按照届时有效法律法规和双方协商结果办理。",
        "analysis": "条款把期限安排限定在有效法律允许范围内，并保留依法续包或调整机制，符合时效对齐要求。",
    },
    {
        "risk_type": "用途和保护义务",
        "keywords": ["用途", "草原", "保护", "生态"],
        "clause": "乙方应按照合同约定和草原保护要求使用草原，不得擅自改变用途、开垦草原或者建设未经批准的设施。",
        "analysis": "条款明确用途限制和生态保护义务，与草原保护类证据能够形成一致支撑。",
    },
    {
        "risk_type": "价款支付约定",
        "keywords": ["价款", "支付", "承包", "合同"],
        "clause": "承包费用、支付期限和支付方式由双方确认；因客观政策调整需要变更的，双方应协商并形成书面补充协议。",
        "analysis": "条款明确费用履行方式，并对变更设置协商和书面确认程序，风险较低。",
    },
    {
        "risk_type": "发包方同意和登记",
        "keywords": ["流转", "登记", "同意", "备案"],
        "clause": "经营权流转、再流转或者用途调整应依法取得必要同意并办理备案、登记或者审批手续。",
        "analysis": "条款保留同意、备案和审批要求，有利于证据链与法定程序对齐。",
    },
    {
        "risk_type": "依法解除条款",
        "keywords": ["解除", "通知", "违约", "补偿"],
        "clause": "任何一方解除合同应具有法定或者约定事由，并提前书面通知对方；造成损失的，按照过错和实际损失依法处理。",
        "analysis": "条款没有设置任意解除权，保留通知、理由和损失处理机制，合同解除风险较低。",
    },
    {
        "risk_type": "征收补偿分配",
        "keywords": ["征收", "补偿", "投入", "经营"],
        "clause": "因征收、征用或者政策调整产生补偿的，双方按照法律规定、实际投入和权益归属协商分配。",
        "analysis": "条款未概括排除承包方补偿权益，保留依法分配和协商路径。",
    },
    {
        "risk_type": "争议解决条款",
        "keywords": ["争议", "调解", "仲裁", "诉讼"],
        "clause": "合同争议可先行协商或者调解；协商不成的，任何一方均可依法申请仲裁或者向有管辖权的人民法院起诉。",
        "analysis": "条款保留多元争议解决渠道，没有排除当事人法定救济权利。",
    },
    {
        "risk_type": "违约责任约定",
        "keywords": ["违约", "损失", "责任", "赔偿"],
        "clause": "违约责任以实际损失、过错程度和合同履行情况为基础确定；约定违约金明显过高或者过低的，可依法调整。",
        "analysis": "条款允许依据实际损失和过错调整违约责任，避免畸重责任。",
    },
    {
        "risk_type": "现行有效规则表达",
        "keywords": ["有效", "法律", "施行", "合同"],
        "clause": "本合同的解释、履行和争议处理适用合同时间锚点对应的现行有效法律法规；法律修订后按新旧规则衔接处理。",
        "analysis": "条款显式要求按时间锚点适用有效法律，符合论文中的法律时效对齐逻辑。",
    },
]


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


def is_active(law: dict, anchor: date) -> bool:
    start = parse_date(law.get("valid_from") or law.get("t_start"))
    end_value = law.get("valid_to") or law.get("t_end")
    end = parse_date(end_value) if end_value else date(9999, 12, 31)
    return start <= anchor <= end


def pick_evidence(kb: list[dict], anchor_value: str | None, keywords: list[str], limit: int = 3) -> list[str]:
    anchor = parse_date(anchor_value)
    scored: list[tuple[int, str]] = []
    for law in kb:
        if not is_active(law, anchor):
            continue
        text = f"{law.get('law_name', '')} {law.get('article_text', '')}"
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scored.append((score, law["law_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [law_id for _, law_id in scored[:limit]]


def original_as_augmented(row: dict) -> dict:
    item = copy.deepcopy(row)
    item["dataset"] = "GrassRiskAugmented"
    item["label_source"] = "human_real"
    item["augmentation_strategy"] = "original_real_annotation"
    item["base_sample_id"] = row.get("sample_id")
    return item


def build_augmented_row(base: dict, template: dict, idx: int, label: int, strategy: str, kb: list[dict]) -> dict:
    row = copy.deepcopy(base)
    anchor = row.get("anchor_date") or row.get("time_anchor") or row.get("contract_effective_date")
    if template.get("temporal_label") == 1 and parse_date(anchor) < date(2021, 1, 1):
        anchor = "2021-01-01"
        row["contract_sign_date"] = "2020-12-20"
        row["contract_effective_date"] = "2021-01-01"
        row["anchor_date"] = anchor
        row["time_anchor"] = anchor

    evidence = pick_evidence(kb, anchor, template["keywords"])
    if not evidence:
        evidence = row.get("gold_basis_ids") or row.get("gold_evidence_ids") or []

    prefix = f"【增强样本{idx:03d}】"
    row.update(
        {
            "sample_id": f"grassrisk_aug_{idx:04d}",
            "clause_id": f"{base.get('contract_id', 'contract')}_aug_clause_{idx:04d}",
            "dataset": "GrassRiskAugmented",
            "task": "contract_risk_review",
            "clause_no": f"增强第{idx}条",
            "clause_text": f"{prefix}{template['clause']}",
            "risk_type": template["risk_type"],
            "label": label,
            "risk_label": label,
            "label_name": "风险条款" if label else "非风险条款",
            "temporal_label": int(template.get("temporal_label", 0)) if label else 0,
            "temporal_error_type": template.get("temporal_error_type", "valid_or_not_applicable"),
            "gold_evidence_ids": evidence,
            "gold_basis_ids": evidence,
            "gold_risk_judgment": "有风险" if label else "无风险",
            "gold_legal_analysis": template["analysis"],
            "gold_temporal_explanation": (
                f"合同时间锚点为{anchor}，应检索该时点有效的法律版本；"
                + ("该样本设置了旧法/新法错配，需要识别失效依据。" if template.get("temporal_label") else "该样本未设置时效错配。")
            ),
            "gold_consistency_label": 1,
            "review_steps": {
                "evidence_summary": template["analysis"],
                "clause_evidence_alignment": "条款文本、风险标签和证据关键词由规则模板同步生成，用于扩充训练，不作为新增人工真值。",
                "temporal_consequence": (
                    "需要根据合同时间锚点过滤有效法律版本。"
                    if template.get("temporal_label")
                    else "按合同时间锚点检索有效法律即可。"
                ),
            },
            "source_type": "rule_augmented_from_real_train_annotation",
            "label_source": "rule_augmented",
            "augmentation_strategy": strategy,
            "base_sample_id": base.get("sample_id"),
            "base_contract_id": base.get("contract_id"),
        }
    )
    return row


def split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    result = {}
    for name in ["train", "val", "test"]:
        part = [r for r in rows if r["_split"] == name]
        result[name] = {
            "total": len(part),
            "positive": sum(1 for r in part if r["label"] == 1),
            "negative": sum(1 for r in part if r["label"] == 0),
            "temporal_positive": sum(1 for r in part if r.get("temporal_label") == 1),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-scale augmented annotations from real GrassRisk annotations.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed/GrassRiskReal")
    parser.add_argument("--law-kb", type=Path, default=ROOT / "data/raw/laws/legal_validity_kb.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/GrassRiskAugmented")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    kb = load_jsonl(args.law_kb)
    splits = {name: load_jsonl(args.input_dir / f"{name}.jsonl") for name in ["train", "val", "test"]}

    rows: list[dict] = []
    for split_name, split_rows in splits.items():
        for row in split_rows:
            item = original_as_augmented(row)
            item["_split"] = split_name
            rows.append(item)

    current_positive = sum(1 for row in rows if row["label"] == 1)
    current_negative = sum(1 for row in rows if row["label"] == 0)
    need_positive = RISK_TARGET_TOTAL - current_positive
    need_negative = NEGATIVE_TARGET_TOTAL - current_negative
    if need_positive < 0 or need_negative < 0:
        raise ValueError("Current real labels exceed the configured paper-scale target.")

    train_positive = [row for row in splits["train"] if row["label"] == 1]
    train_negative = [row for row in splits["train"] if row["label"] == 0]
    train_all = splits["train"]
    if not train_positive:
        train_positive = train_all
    if not train_negative:
        train_negative = train_all

    next_idx = 1
    for _ in range(need_positive):
        base_pool = train_positive if rng.random() < 0.65 else train_all
        base = rng.choice(base_pool)
        template = RISK_TEMPLATES[(next_idx - 1) % len(RISK_TEMPLATES)]
        row = build_augmented_row(base, template, next_idx, 1, "paper_logic_risk_counterfactual", kb)
        row["_split"] = "train"
        rows.append(row)
        next_idx += 1

    for _ in range(need_negative):
        base = rng.choice(train_negative)
        template = SAFE_TEMPLATES[(next_idx - 1) % len(SAFE_TEMPLATES)]
        row = build_augmented_row(base, template, next_idx, 0, "paper_logic_compliant_variant", kb)
        row["_split"] = "train"
        rows.append(row)
        next_idx += 1

    rows.sort(key=lambda r: ({"train": 0, "val": 1, "test": 2}[r["_split"]], r["sample_id"]))
    out_splits = {name: [] for name in ["train", "val", "test"]}
    all_rows = []
    for row in rows:
        split_name = row.pop("_split")
        out_splits[split_name].append(row)
        all_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all.jsonl", all_rows)
    for split_name, split_rows in out_splits.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", split_rows)

    meta = {
        "dataset": "GrassRiskAugmented",
        "description": "Paper-scale training augmentation built from GrassRiskReal. Validation and test splits keep only original human annotations; synthetic rows are placed in train only to avoid leakage.",
        "total": len(all_rows),
        "positive": sum(1 for row in all_rows if row["label"] == 1),
        "negative": sum(1 for row in all_rows if row["label"] == 0),
        "human_real": sum(1 for row in all_rows if row.get("label_source") == "human_real"),
        "rule_augmented": sum(1 for row in all_rows if row.get("label_source") == "rule_augmented"),
        "temporal_positive": sum(1 for row in all_rows if row.get("temporal_label") == 1),
        "split_counts": split_counts([{**row, "_split": split_name} for split_name, split_rows in out_splits.items() for row in split_rows]),
        "risk_type_counts": dict(Counter(row["risk_type"] for row in all_rows)),
        "label_policy": "Do not report augmented labels as newly collected manual truth. Use GrassRiskReal for final real-world evaluation; use this dataset for data-scarce training and ablation.",
        "targets": {"total": 620, "positive": RISK_TARGET_TOTAL, "negative": NEGATIVE_TARGET_TOTAL},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = ROOT / "outputs/grassrisk_augmented_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
