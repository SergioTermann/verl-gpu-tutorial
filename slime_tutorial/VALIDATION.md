# slime 系列验证记录

参考环境：WSL Ubuntu 24.04、Python 3.12、`/opt/verl-venv`、RTX 4060 Laptop 8GB。上游源码来自 `THUDM/slime` commit `79aa41a884ab0bf65222c94418f6357d3c31192b`。本次仅验证源码、接口和 CPU 夹具；没有启动完整 Megatron、SGLang 或 Ray 训练。

## 已执行

- S01 读取固定版本的 `train.py`、`Sample` 类型、奖励模块和官方调试文档，确认训练循环包含初始权重同步、rollout、异步训练、训练完成通知、checkpoint 和下一次权重同步入口。
- S02 从固定版本导入真实 `slime.utils.types.Sample`，测试严格 yes/no 奖励、错误标签拒绝、批量奖励顺序、response_length、loss_mask 和 rollout_log_probs 对齐。错误 mask 和重复 segment 统计均被断言拦截。
- S03 检查官方文档中的 `--debug-rollout-only`、`--debug-train-only`、保存/加载 rollout 数据和保存训练数据参数，生成未填写硬件与镜像字段的运行计划。它明确标记为不可直接启动的计划。

## 当前环境限制

本机 verl 环境没有安装 SGLang，硬件也不是官方 GPU CI 主要覆盖的 H100/H200。课程因此没有伪造端到端训练结果。要继续，需要按上游 Quick Start 使用匹配的 Docker 或 conda 环境，并记录镜像 digest、驱动、CUDA、Megatron、SGLang、Ray 与模型 revision。

课程执行会检查上游 checkout 的 commit。Windows checkout 在 WSL 中可能显示换行差异，代码只忽略 line-ending 差异，仍会拒绝有实际内容的本地修改。第三方源码不复制进本仓库。
