# 免费/本地优先模型来源核验

本工程默认不调用远程 API，不自动使用任何可能扣费的接口。API Key 只能写入 `.env`，不能写入代码、README 或日志。

## Qwen3-8B

- 候选模型：`Qwen/Qwen3-8B`
- 来源：Hugging Face / ModelScope
- 策略：本地权重优先，路径由 `.env` 的 `QWEN3_8B_MODEL_PATH` 提供。
- 默认状态：不作为默认远程 API 调用。

## InternLM-Law-7B

- 候选模型：`internlm/internlm2-law-7b`
- 来源：InternLM/InternLM-Law、Hugging Face 或官方发布位置。
- 策略：仅在本地权重路径可用时参与实验，路径由 `.env` 的 `INTERNLM_LAW_7B_MODEL_PATH` 提供。
- 默认状态：找不到可加载权重时不写入结果表。

## MiniCPM-2.4B

- 候选模型：`openbmb/MiniCPM-2B-sft-bf16`
- 来源：Hugging Face / ModelScope
- 策略：本地权重优先，路径由 `.env` 的 `MINICPM_2_4B_MODEL_PATH` 提供。

## MiniCPM-SFT

- 外部 API：不使用。
- 策略：基于本地 MiniCPM-2.4B 进行 LoRA/SFT 后保存 adapter，路径由 `.env` 的 `MINICPM_SFT_ADAPTER_PATH` 提供。
- 默认状态：adapter 不存在时不写入结果表。

## Expert-only

- 用途：表13“计算效率与部署收益”中的 13B 专家模型配置。
- 策略：只读取 `.env` 中的 `EXPERT_13B_MODEL_PATH` 或命令行 `--expert-model-path`。
- 默认状态：路径不存在、权重加载失败或显存不足时不写入结果表。
