# Publishing Checklist

This file records the exact release steps for publishing the cleaned project.

## GitHub Code Repository

Recommended repository:

```text
https://github.com/1553965822/Temporal-Evidence-RAG
```

Run from the project root:

```bash
git init -b main
git config user.email "1553965822@qq.com"
git config user.name "1553965822"
git add .
git commit -m "Initial open-source release"
git remote add origin https://github.com/1553965822/Temporal-Evidence-RAG.git
git push -u origin main
```

Before pushing, verify that the staged files do not include real datasets or model weights:

```bash
git status --short
git check-ignore -v data/processed/GrassRisk/train.jsonl \
  data/raw/legal_validity_kb.jsonl \
  models/some_weight.safetensors \
  outputs/run.log \
  .env
```

Expected: data files, model weights, outputs, and `.env` are ignored.

## Hugging Face Dataset Repository

Recommended dataset repository:

```text
p1553965822/paper-aligned-contract-rag-datasets
```

Install dependencies and log in:

```bash
python -m pip install -r requirements.txt
hf auth login
```

Create the dataset repository:

```bash
hf repos create p1553965822/paper-aligned-contract-rag-datasets --type dataset --private
```

Upload:

```bash
python scripts/upload_datasets_to_hf.py \
  --repo-id p1553965822/paper-aligned-contract-rag-datasets \
  --source-dir data \
  --private
```

Use `--private` unless you have verified that every dataset sample is safe to publish.

## Download Test

After uploading, test a fresh dataset download:

```bash
python scripts/download_datasets.py \
  --repo-id p1553965822/paper-aligned-contract-rag-datasets \
  --local-dir data
```

Then run:

```bash
python scripts/build_datasets.py
python scripts/check_model_connections.py
```
