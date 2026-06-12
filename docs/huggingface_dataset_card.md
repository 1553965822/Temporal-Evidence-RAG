---
pretty_name: Paper-Aligned Contract Review RAG Datasets
language:
  - zh
  - en
task_categories:
  - text-classification
  - text-generation
  - question-answering
tags:
  - legal
  - contract-review
  - rag
  - temporal-retrieval
  - evidence-aware-generation
---

# Paper-Aligned Contract Review RAG Datasets

This dataset package supports the project "基于法律时效对齐与证据利用的合同审查RAG框架".

## Dataset Components

- `GLTRD`: Grassland Legal Timeliness Review Dataset for legal temporal validity alignment.
- `GrassRisk`: grassland contract risk review dataset.
- `CUADRisk`: reconstructed general contract risk review dataset based on CUAD-style contract risk review.
- `legal_validity_kb`: legal temporal validity knowledge base used by Temporal-RAG.

## Expected File Layout

```text
raw/legal_validity_kb.jsonl
raw/laws/legal_validity_kb.jsonl
processed/GLTRD/train.jsonl
processed/GLTRD/val.jsonl
processed/GLTRD/test.jsonl
processed/GLTRD/all.jsonl
processed/GrassRisk/train.jsonl
processed/GrassRisk/val.jsonl
processed/GrassRisk/test.jsonl
processed/GrassRisk/all.jsonl
processed/CUADRisk/train.jsonl
processed/CUADRisk/val.jsonl
processed/CUADRisk/test.jsonl
processed/CUADRisk/all.jsonl
```

## Fields

Common fields include:

- `sample_id`
- `contract_id`
- `dataset`
- `task`
- `clause_text`
- `anchor_date`
- `risk_type`
- `label`
- `label_name`
- `evidence_text`

Risk-review datasets may also include:

- `gold_evidence_ids`
- `review_steps`

## Use

```bash
python scripts/download_datasets.py \
  --repo-id p1553965822/paper-aligned-contract-rag-datasets \
  --local-dir data
```

## Notes

Do not upload private contracts, personal information, API keys, model weights, or local run outputs to this dataset repository.
