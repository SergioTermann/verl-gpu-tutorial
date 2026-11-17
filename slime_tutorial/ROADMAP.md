# slime 学习路线

## 已完成

S01 建立架构地图并核对训练主循环。S02 用真实 `Sample` 类测试严格奖励、批量奖励和 response mask。S03 建立 rollout-only、train-only、完整两轮和 checkpoint resume 的验收顺序。

当前验证只到接口和调试计划；不要将这三课的 PASS 写成 slime 端到端训练结果。

## 阶段一：官方最小配方

在兼容 Docker 或 conda 环境中完成官方 quick start，不修改模型、并行和奖励参数。提交镜像 digest、模型转换命令、首轮日志、显存峰值、rollout 样本和 checkpoint 目录。先用 `--debug-rollout-only`，再用 `--debug-train-only`，最后跑至少两轮完整闭环。

## 阶段二：可验证奖励

把 S02 的 strict yes/no 奖励接到真实数据源，按模板或任务族划分 train/validation/test。记录合法输出率、奖励分布、有效组比例、准确率、梯度范数和参数版本。单独报告生成失败、截断和任务错误。

## 阶段三：定制生成与工具轨迹

使用 `--custom-generate-function-path` 加入多轮工具或检索步骤；只有需要替换整个 rollout 编排时才使用 `--rollout-function-path`。保留完整轨迹、工具失败和 segment 的 rollout_id/group_index，核对共享前缀只计一次 loss。

## 阶段四：异步、恢复和扩展

比较同步与 fully async rollout 的样本滞后和吞吐。人为中断后恢复模型、优化器、数据进度和权重版本。再测多卡并行、offload、straw 持久化传输和长尾 agent 任务。所有性能比较固定请求分布和测量窗口。

## 阶段五：研究级证据

至少三个训练种子，预注册主指标和失败判据，保留最终留出集。比较 GRPO、DAPO、DPO 或自定义 loss 时，记录生成 token、训练 token、GPU 时间、峰值显存和质量。不要只用 reward 曲线或单次 checkpoint 选择结论。

## 研究生毕业标准

你能从一条错误样本追到模板、token、mask、reward、advantage、梯度、权重同步和下一轮生成；能在不扩大模型或预算的情况下提出最小可验证修复；能把一个成功或失败结论绑定到具体 commit、配置、数据哈希和日志。
