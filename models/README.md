# Model Placeholder

Model weights are intentionally not included in the open-source repository.

Download local models or adapters into this directory, or set absolute paths in `.env`:

- `MINICPM_2_4B_MODEL_PATH`
- `MINICPM_SFT_ADAPTER_PATH`
- `QWEN3_8B_MODEL_PATH`
- `INTERNLM_LAW_7B_MODEL_PATH`
- `ROBERTA_MODEL_PATH`
- `EXPERT_13B_MODEL_PATH`

You can download supported public baselines with:

```bash
python scripts/download_models.py --models minicpm roberta qwen3
```

Never commit downloaded weights, LoRA adapters, merged checkpoints, API keys, or private checkpoints.
