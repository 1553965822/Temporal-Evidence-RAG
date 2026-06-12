from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from .config import ensure_dir, load_yaml, project_path


@dataclass(frozen=True)
class LawItem:
    law_id: str
    law_key: str
    law_name: str
    article_no: str
    article_text: str
    valid_from: str
    valid_to: str | None
    status: str
    risk_tags: list[str]

    def as_dict(self) -> dict:
        return {
            "law_id": self.law_id,
            "law_key": self.law_key,
            "law_name": self.law_name,
            "article_no": self.article_no,
            "article_text": self.article_text,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "status": self.status,
            "risk_tags": self.risk_tags,
        }


RISK_TYPES = [
    "期限超限",
    "转包转让限制",
    "生态保护义务",
    "用途变更",
    "补偿与价款",
    "解除与违约",
    "争议解决",
]


def make_laws() -> list[dict]:
    laws = [
        LawItem(
            "LAW-GRASS-2002-13",
            "grassland_law_article_13",
            "中华人民共和国草原法",
            "第十三条",
            "草原承包经营应当遵守草原保护、建设、利用规划，不得破坏草原植被和生态环境。",
            "2002-12-28",
            "2021-04-30",
            "expired",
            ["生态保护义务", "用途变更"],
        ),
        LawItem(
            "LAW-GRASS-2021-13",
            "grassland_law_article_13",
            "中华人民共和国草原法",
            "第十三条",
            "草原承包经营和流转应当符合草原保护制度，严禁擅自开垦、改变用途或者造成草原退化。",
            "2021-05-01",
            None,
            "effective",
            ["生态保护义务", "用途变更"],
        ),
        LawItem(
            "LAW-GRASS-2002-46",
            "grassland_law_article_46",
            "中华人民共和国草原法",
            "第四十六条",
            "违反草原保护规定造成草原资源损害的，应当依法承担恢复治理和赔偿责任。",
            "2002-12-28",
            "2021-04-30",
            "expired",
            ["生态保护义务", "解除与违约"],
        ),
        LawItem(
            "LAW-GRASS-2021-46",
            "grassland_law_article_46",
            "中华人民共和国草原法",
            "第四十六条",
            "破坏草原生态或者擅自改变草原用途的，应承担停止侵害、恢复原状、赔偿损失等责任。",
            "2021-05-01",
            None,
            "effective",
            ["生态保护义务", "用途变更", "解除与违约"],
        ),
        LawItem(
            "LAW-RURAL-2018-38",
            "rural_land_contract_law_article_38",
            "中华人民共和国农村土地承包法",
            "第三十八条",
            "土地经营权流转应当依法、自愿、有偿，任何组织和个人不得强迫或者阻碍承包方流转。",
            "2019-01-01",
            None,
            "effective",
            ["转包转让限制", "补偿与价款"],
        ),
        LawItem(
            "LAW-CIVIL-2021-509",
            "civil_code_article_509",
            "中华人民共和国民法典",
            "第五百零九条",
            "当事人应当按照约定全面履行自己的义务，并遵循诚信原则。",
            "2021-01-01",
            None,
            "effective",
            ["解除与违约", "争议解决"],
        ),
        LawItem(
            "LAW-CIVIL-2021-563",
            "civil_code_article_563",
            "中华人民共和国民法典",
            "第五百六十三条",
            "当事人一方迟延履行主要债务，经催告后在合理期限内仍未履行的，可以解除合同。",
            "2021-01-01",
            None,
            "effective",
            ["解除与违约"],
        ),
        LawItem(
            "LAW-LOCAL-2015-22",
            "local_grassland_rule_article_22",
            "内蒙古自治区草原管理条例",
            "第二十二条",
            "承包经营草原不得超载放牧，不得擅自采挖、开垦或者建设与草原保护无关的设施。",
            "2015-07-01",
            "2023-12-31",
            "expired",
            ["生态保护义务", "用途变更"],
        ),
        LawItem(
            "LAW-LOCAL-2024-22",
            "local_grassland_rule_article_22",
            "内蒙古自治区草原管理条例",
            "第二十二条",
            "承包经营草原应当落实载畜量控制、休牧轮牧和生态修复要求，未经批准不得改变用途。",
            "2024-01-01",
            None,
            "effective",
            ["生态保护义务", "用途变更"],
        ),
        LawItem(
            "LAW-CONTRACT-1999-94",
            "contract_law_article_94",
            "中华人民共和国合同法",
            "第九十四条",
            "当事人一方迟延履行债务或者有其他违约行为致使不能实现合同目的的，当事人可以解除合同。",
            "1999-10-01",
            "2020-12-31",
            "expired",
            ["解除与违约"],
        ),
        LawItem(
            "LAW-CIVIL-2021-577",
            "civil_code_article_577",
            "中华人民共和国民法典",
            "第五百七十七条",
            "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任。",
            "2021-01-01",
            None,
            "effective",
            ["解除与违约"],
        ),
    ]
    return [law.as_dict() for law in laws]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def valid_at(law: dict, anchor: str) -> bool:
    current = date.fromisoformat(anchor)
    start = date.fromisoformat(law["valid_from"])
    end = date.fromisoformat(law["valid_to"]) if law.get("valid_to") else None
    return start <= current and (end is None or current <= end)


def choose_law_pair(rng: random.Random, invalid: bool) -> tuple[dict, dict, str]:
    laws = make_laws()
    law_keys = sorted({law["law_key"] for law in laws})
    key = rng.choice(law_keys)
    versions = [law for law in laws if law["law_key"] == key]
    if len(versions) == 1:
        gold = versions[0]
        anchor = rng.choice(["2022-06-01", "2024-04-12", "2019-05-20"])
        return gold, gold, anchor

    old = min(versions, key=lambda x: x["valid_from"])
    new = max(versions, key=lambda x: x["valid_from"])
    if invalid:
        if rng.random() < 0.5:
            return new, old, "2018-06-01"
        return old, new, "2024-05-01"
    if rng.random() < 0.5:
        return old, old, "2018-06-01"
    return new, new, "2024-05-01"


def build_gltrd(rng: random.Random, total: int, positives: int) -> list[dict]:
    records = []
    for i in range(total):
        invalid = i < positives
        cited, gold, anchor = choose_law_pair(rng, invalid)
        risk = rng.choice(cited["risk_tags"])
        clause = (
            f"本合同于{anchor}确定草场承包事项，承包方在{risk}方面应按"
            f"{cited['law_name']}{cited['article_no']}执行。"
        )
        records.append(
            {
                "sample_id": f"GLTRD-{i + 1:04d}",
                "contract_id": f"GLTRD-C{i // 4 + 1:04d}",
                "dataset": "GLTRD",
                "task": "legal_temporal_alignment",
                "clause_text": clause,
                "anchor_date": anchor,
                "cited_law_id": cited["law_id"],
                "gold_law_id": gold["law_id"],
                "law_key": gold["law_key"],
                "risk_type": risk,
                "label": 1 if invalid else 0,
                "label_name": "失效引用" if invalid else "有效引用",
                "evidence_text": gold["article_text"],
            }
        )
    rng.shuffle(records)
    return records


def grass_clause(rng: random.Random, risk_type: str, risk: bool, idx: int) -> str:
    date_text = rng.choice(["2018年6月1日", "2021年5月1日", "2024年4月12日", "2020-09-01"])
    compliant = {
        "期限超限": "承包期限按批准文件执行，期满后依法重新协商。",
        "转包转让限制": "未经发包方书面同意不得转包、转让或改变经营主体。",
        "生态保护义务": "承包方应执行休牧轮牧、载畜量控制和生态修复要求。",
        "用途变更": "草场仅用于依法放牧和生态保护，不得擅自开垦建设。",
        "补偿与价款": "承包价款、补偿标准和支付期限均以书面清单为准。",
        "解除与违约": "违约责任、整改期限和解除条件依照现行法律执行。",
        "争议解决": "双方争议先协商，协商不成提交合同履行地人民法院处理。",
    }
    risky = {
        "期限超限": "承包期限固定为八十年，期满无需重新审批即可自动续期。",
        "转包转让限制": "承包方可自行转包给第三人，发包方不得提出异议。",
        "生态保护义务": "承包方可根据收益需要超载放牧，生态恢复费用由发包方承担。",
        "用途变更": "承包方可将草场改作旅游设施和临时采挖场地，无需另行批准。",
        "补偿与价款": "发包方可单方调整承包费并即时扣除保证金。",
        "解除与违约": "发包方可不经通知直接解除合同且不承担任何补偿责任。",
        "争议解决": "任何争议均由发包方最终解释，承包方不得诉讼或仲裁。",
    }
    body = risky[risk_type] if risk else compliant[risk_type]
    return f"第{idx % 18 + 1}条 本合同签订日期为{date_text}。{body}"


def cuad_clause(rng: random.Random, risk_type: str, risk: bool, idx: int) -> str:
    english_terms = {
        "期限超限": ("The term renews automatically without any statutory review.", "The term renews only after written approval."),
        "转包转让限制": ("Either party may assign all obligations without consent.", "Assignment requires prior written consent."),
        "生态保护义务": ("The operator may ignore environmental remediation duties.", "The operator must follow applicable environmental duties."),
        "用途变更": ("The licensee may change the permitted use at its sole discretion.", "Use changes require written approval and legal compliance."),
        "补偿与价款": ("Fees may be changed unilaterally with immediate effect.", "Fees and compensation follow the agreed schedule."),
        "解除与违约": ("Termination may occur without notice or cure period.", "Termination follows notice, cure period, and applicable law."),
        "争议解决": ("One party has final and non-reviewable dispute authority.", "Disputes are submitted to the agreed court or arbitration body."),
    }
    risky, safe = english_terms[risk_type]
    return f"Clause {idx % 25 + 1}. {risky if risk else safe}"


def build_risk_dataset(
    rng: random.Random,
    dataset_name: str,
    total: int,
    positives: int,
) -> list[dict]:
    records = []
    for i in range(total):
        risk = i < positives
        risk_type = RISK_TYPES[i % len(RISK_TYPES)]
        if dataset_name == "GrassRisk":
            clause = grass_clause(rng, risk_type, risk, i)
            anchor = rng.choice(["2018-06-01", "2021-05-01", "2024-04-12"])
        else:
            clause = cuad_clause(rng, risk_type, risk, i)
            anchor = rng.choice(["2019-06-01", "2021-02-01", "2024-01-15"])
        law = rng.choice([law for law in make_laws() if risk_type in law["risk_tags"]] or make_laws())
        records.append(
            {
                "sample_id": f"{dataset_name}-{i + 1:04d}",
                "contract_id": f"{dataset_name}-C{i // 5 + 1:04d}",
                "dataset": dataset_name,
                "task": "contract_risk_review",
                "clause_text": clause,
                "anchor_date": anchor,
                "risk_type": risk_type,
                "label": 1 if risk else 0,
                "label_name": "风险条款" if risk else "非风险条款",
                "gold_evidence_ids": [law["law_id"]],
                "evidence_text": law["article_text"],
                "review_steps": {
                    "evidence_summary": f"核心证据涉及{risk_type}。",
                    "clause_evidence_alignment": "条款与法律依据存在对应关系。" if risk else "条款与法律依据基本一致。",
                    "temporal_consequence": "需结合合同时间锚点判断法律效力。",
                },
            }
        )
    rng.shuffle(records)
    return records


def stratified_split(records: list[dict], ratios: list[float], rng: random.Random) -> dict[str, list[dict]]:
    buckets: dict[int, list[dict]] = {0: [], 1: []}
    for record in records:
        buckets[int(record["label"])].append(record)
    splits = {"train": [], "val": [], "test": []}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        splits["train"].extend(bucket[:n_train])
        splits["val"].extend(bucket[n_train : n_train + n_val])
        splits["test"].extend(bucket[n_train + n_val :])
    for part in splits.values():
        rng.shuffle(part)
    return splits


def write_dataset(name: str, records: list[dict], ratios: list[float], rng: random.Random) -> None:
    out_dir = ensure_dir(project_path("data", "processed", name))
    write_jsonl(out_dir / "all.jsonl", records)
    splits = stratified_split(records, ratios, rng)
    for split_name, split_records in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", split_records)
    meta = {
        "dataset": name,
        "total": len(records),
        "positive": sum(1 for r in records if r["label"] == 1),
        "negative": sum(1 for r in records if r["label"] == 0),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "paper_aligned": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_all(config_path: str | Path = "configs/experiment.yaml", force: bool = False) -> dict[str, int]:
    cfg = load_yaml(config_path)
    rng = random.Random(cfg["project"]["seed"])
    raw_dir = ensure_dir(project_path("data", "raw"))
    laws = make_laws()
    synthetic_laws_path = raw_dir / "legal_validity_kb.synthetic.jsonl"
    if force or not synthetic_laws_path.exists():
        write_jsonl(synthetic_laws_path, laws)
    canonical_laws_path = raw_dir / "legal_validity_kb.jsonl"
    if not canonical_laws_path.exists():
        write_jsonl(canonical_laws_path, laws)

    built = {}
    for name, spec in cfg["datasets"].items():
        out_dir = project_path("data", "processed", name)
        if out_dir.exists() and (out_dir / "all.jsonl").exists() and not force:
            built[name] = len(read_jsonl(out_dir / "all.jsonl"))
            continue
        total = int(spec["total"])
        positives = int(spec["positive_count"])
        if name == "GLTRD":
            records = build_gltrd(rng, total, positives)
        else:
            records = build_risk_dataset(rng, name, total, positives)
        write_dataset(name, records, list(spec["split"]), rng)
        built[name] = len(records)
    return built


def load_dataset(name: str, split: str = "all") -> list[dict]:
    return read_jsonl(project_path("data", "processed", name, f"{split}.jsonl"))
