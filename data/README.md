# Data Placeholder

This repository intentionally does not include dataset files.

Download the datasets from Hugging Face:

```bash
python scripts/download_datasets.py \
  --repo-id p1553965822/paper-aligned-contract-rag-datasets \
  --local-dir data
```

Expected local layout after download:

```text
data/raw/legal_validity_kb.jsonl
data/raw/laws/legal_validity_kb.jsonl
data/processed/GLTRD/{train,val,test,all}.jsonl
data/processed/GrassRisk/{train,val,test,all}.jsonl
data/processed/CUADRisk/{train,val,test,all}.jsonl
```

If you do not have the released dataset yet, generate synthetic data for a quick smoke run:

```bash
python scripts/build_datasets.py --force
```
