# 论文实验到真实工程入口的映射

| 论文实验 | 工程位置 |
|---|---|
| 数据集统计 | `scripts/run_paper_experiment.py --mode measured` |
| GLTRD | `data/processed/GLTRD/*.jsonl` |
| GrassRisk | `data/processed/GrassRisk/*.jsonl` |
| CUADRisk | `data/processed/CUADRisk/*.jsonl` |
| 法律效力周期知识库 | `data/raw/laws/legal_validity_kb.jsonl` |
| CUADRisk 英文法律知识库 | `data/raw/laws/en/cuadrisk_legal_validity_kb.en.jsonl` |
| GrassRisk 整体合同风险审查性能对比 | `scripts/run_minicpm_rag_evaluation.py --dataset GrassRisk --calibration-objective f1` |
| CUADRisk 整体合同风险审查性能对比 | `scripts/run_minicpm_rag_evaluation.py --dataset CUADRisk --calibration-objective f1` |
| 法律时效对齐整体性能 | `scripts/run_component_experiments.py --mode gltrd` |
| 时间锚点提取 | `scripts/run_component_experiments.py --mode gltrd` |
| 检索证据命中 | `scripts/run_component_experiments.py --mode gltrd` |
| Temporal-RAG 关键环节贡献 | `scripts/run_component_experiments.py --mode gltrd` |
| 低资源证据利用 | `scripts/run_component_experiments.py --mode evidence` |
| 关键步骤对齐训练策略 | `scripts/run_component_experiments.py --mode evidence` |
| 计算效率与部署收益 | `scripts/run_efficiency_benchmark.py` |

所有入口均重新读取当前数据集并重新计算输出，不读取旧结果表作为新实验结果。
