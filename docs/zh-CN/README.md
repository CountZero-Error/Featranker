# 手册

[English](../README.md) · 简体中文

这是 FeatRanker 手册。先读快速入门，再按任务选择对应的指南。

[README](../../README.zh-CN.md) | [快速入门](getting-started.md) | [参考](reference.md)

| 页面 | 内容 |
|---|---|
| [10 分钟实践流程](../../README.zh-CN.md#-10-分钟实践流程) 与 [何时使用 FeatRanker](../../README.zh-CN.md#-何时使用-featranker) | 排名、检查支持度、选择子集并在未使用过的行上验证；以及何时不该使用 FeatRanker |
| [快速入门](getting-started.md) | 安装、对第一份数据排名、阅读结果 |
| [概念](concepts.md) | 排名如何确定，以及每个数字的含义 |
| [模型与评分](guide/models-and-scoring.md) | 任务、模型组、评分器、多个指标 |
| [自定义模型](guide/custom-models.md) | 自带模型、模型族、命名列、并行运行 |
| [数据](guide/data.md) | 缺失值、类别特征、样本权重、标签 |
| [验证](guide/validation.md) | 重采样排名、时间序列与分组数据、嵌套验证 |
| [噪声与证据](guide/noise-and-evidence.md) | 影子特征、命中检验、模型族证据、SHAP |
| [特征选择](guide/selection.md) | `ConsensusSelector`、消融曲线、`summarize()`、比较多次运行、`auto_groups()` |
| [性能](guide/performance.md) | 速度、子样本、大数据 |
| [命令行](cli.md) | `featranker` 命令及其全部参数 |
| [参考](reference.md) | 每个公开名称：签名、参数、返回值、异常 |
| [故障排查](troubleshooting.md) | 错误、警告和常见问题 |
| [报告](../report.md) | 如何阅读报告、重采样报告、完整结构 |
| [模型](../models.md) | 全部 40 个内置模型、模型族和可选库 |
| [API](../api.md) | 哪些是公开的、稳定性策略、报告模式、发布 |
| [基准测试](../benchmark.md) | 在合成数据上将 FeatRanker 与 Boruta 等比较，包括它落后的情形 |
| [校准研究](../calibration.md) | 影子检验在零数据和有信号数据上的表现 |
| [真实数据研究](../realdata.md) | 在 13 个 OpenML 数据集上，选择子集与使用全部特征的对比 |
| [用法](../usage.md) | 从旧的单页指南到这些页面的对照表 |
