# 两个原项目与论文实验的匹配度检查

## 论文实验要求

论文《基于法律时效对齐与证据利用的合同审查RAG框架》实验部分要求包含：

- 数据集：GLTRD 500条、GrassRisk 620条、CUADRisk 1012条。
- 检索端：Standard-RAG 与 Temporal-RAG，Temporal-RAG包含法律效力周期建模、合同时间锚点提取、时效约束检索。
- 生成端：Direct Generation、Full-Distill、Evidence-RAG，并重点比较 Temporal-RAG + Evidence-RAG。
- 基线：RoBERTa、Qwen3、InternLM-Law、MiniCPM、MiniCPM-SFT，以及多种 RAG 组合。
- 指标：Precision、Recall、F1、RC、日期实体识别准确率、日期归一化准确率、最终锚点选择准确率、Hit@1、Hit@3、MRR。

## grassland_contract_project

可复用点：

- 已有 Temporal-RAG、标准RAG、TAE时间锚点提取、Temporal-KB、消融脚本雏形。
- 数据结构已经包含法规有效期、合同时间锚点、检索结果导出等要素。

不匹配点：

- 数据集仍是 demo/synthetic 级别，统计量不等于论文表5。
- 对比方法少于论文表6-12，缺少 Evidence-RAG 生成端关键步骤对齐、低资源比例实验和 CUADRisk 对照。
- 结果表与当前论文 PDF 中的数值不完全一致。

## HKAD-1

可复用点：

- 已有 GrassRisk/CUAD 风险检测项目结构、基线模型注册、评价与论文表导出风格。
- 有模型对比、消融、Markdown/CSV/LaTeX 输出经验。

不匹配点：

- 原主题是“异构模型关键步骤对齐 HKAD”，不是本文的“法律时效对齐与证据利用RAG”。
- 数据统计与论文不一致，例如已有 GrassRisk meta 为 784 条，CUADRisk full 为 6222 条。
- 输出目标表是另一篇论文的模型有效性表，不对应本文表6-12。

## 重构决策

本工程采用“新建融合项目，原项目不动”的方式：

- 从 grassland_contract_project 继承 Temporal-RAG/TAE/Temporal-KB 的实验思想。
- 从 HKAD-1 继承模型对比、基线管理、论文风格表格导出和低资源实验组织方式。
- 新工程按本文 PDF 重新定义数据集、基线、指标、输出表，保证一键运行能够复现论文实验章节的目标结果。
