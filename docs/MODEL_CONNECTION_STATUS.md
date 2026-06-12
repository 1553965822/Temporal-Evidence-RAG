# Model Connection Status

Model weights are not bundled in this open-source copy. Configure local paths in `.env`.

| Model | Environment variable | Purpose |
|---|---|---|
| MiniCPM-2.4B | `MINICPM_2_4B_MODEL_PATH` | Lightweight generation model and RAG generator |
| MiniCPM-SFT | `MINICPM_SFT_ADAPTER_PATH` | Local LoRA/SFT adapter |
| Qwen3-8B | `QWEN3_8B_MODEL_PATH` | Larger baseline or manual connection check |
| RoBERTa | `ROBERTA_MODEL_PATH` | Encoder baseline |
| InternLM-Law-7B or substitute | `INTERNLM_LAW_7B_MODEL_PATH` | Legal LLM baseline |
| Expert-only 13B | `EXPERT_13B_MODEL_PATH` | Efficiency benchmark expert model |

If a model path is missing, cannot be loaded, or exceeds available memory, the corresponding method should be omitted from real result tables instead of being filled with placeholder values.

Main entry points:

- `scripts/run_minicpm_rag_evaluation.py`
- `scripts/train_minicpm_sft.py`
- `scripts/run_component_experiments.py`
- `scripts/run_efficiency_benchmark.py`
