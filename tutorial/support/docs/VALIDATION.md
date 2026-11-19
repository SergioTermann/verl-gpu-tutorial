# 新增课程验证记录

本记录对应 2026-11-17 的参考运行。环境：RTX 4060 Laptop GPU 8GB、WSL Ubuntu 24.04、Python 3.12.3、PyTorch 2.8.0+cu128。Notebook 在独立内核中按 13、14、15、16 顺序完整执行；公开文件清空输出，下面的 JSON/CSV 保存实际测量。

## 执行结果与证据

| 检查 | 结果 | 证据 |
|---|---|---|
| 13：标签错位、策略梯度、裁剪、mask、KL | 断言全部通过 | [checks.json](../experiment13_output/checks.json) |
| 14：固定 200 步 SFT | 测试合法率/准确率从 0%/0% 到 100%/100%，共 80 题 | [summary.csv](../experiment14_output/summary.csv)、[逐题预测](../experiment14_output/predictions.csv) |
| 14：追加真实 verl GRPO | 10 步完成，1 步非零梯度；测试准确率仍为 100% | [报告](../experiment14_output/grpo_report.json)、[逐步指标](../experiment14_output/grpo_metrics.csv)、[逐题预测](../experiment14_output/grpo_predictions.csv) |
| 15：配对评估 | 80 题均改善；bootstrap 百分位区间退化为 [1,1] | [evaluation_audit.json](../experiment15_output/evaluation_audit.json) |
| 16：CUDA 微基准、BF16 数值检查 | 三种 batch 的重复计时完成，BF16 输出有限 | [microbenchmark.csv](../experiment16_output/microbenchmark.csv)、[环境记录](../experiment16_output/environment.json) |

第 14 课的数据、种子与超参数见 [manifest.json](../experiment14_output/manifest.json)，验证曲线原始点见 [history.csv](../experiment14_output/history.csv)。SFT 与 GRPO 均使用固定最后一步，不按测试成绩选检查点。

## 怎样解释这些结果

SFT 实验支持“这套监督数据和预算能解决本次协议与三位数奇偶任务”。它同时改变了第 12 课的词表、数字范围和训练方式，因此不隔离某一个因素的贡献，也不证明通用推理能力。

GRPO 测试说明预热模型可以通过导出的 tokenizer、Parquet 和奖励函数完成真实训练。SFT 已在这批测试上达到满分，追加训练没有带来可见准确率提升。训练中出现少量非零梯度不等于测试分数必须改变。本次未运行继续 SFT 对照，也未执行多种子，因此不比较两种方法的普遍优劣或成本收益。

第 15 课的 [1,1] 是本次观测差值全部为 1 时，非参数重采样必然退化的结果。它不表示总体提升必定是 100 个百分点。第 15 课同时提供单模型 Wilson 区间并讨论假设；没有将退化区间当作普遍性能保证。

第 16 课只测 MLP 推理，不包含 tokenization、自回归解码、请求排队或分布式通信。其吞吐单位为样本/秒，不能当作大模型服务的 token/秒。报告中的显存仅是当前 PyTorch 进程统计。

## 复现方式

按 README 配置环境，运行四本 Notebook。第 14 课导出模型后，在终端执行其打印的 `VERL_ADV14_ROOT=... bash tutorial/support/advanced14/run_grpo.sh`；成功后将第 5 节的 `READ_GRPO=True` 再运行，即可从检查点重新评估并保存指标。读取只适用于自己生成的可信检查点。

大型权重、训练原始日志与包含本机路径的运行指针未提交。代码会为新实验创建独立运行目录；再次执行报告保存格会覆盖工作区内同名小型报告。需要比较多个种子时，先为报告建立独立目录并保留每轮实验卡。

## 第 17 课：预训练 Qwen LoRA

2026-11-17 使用 Qwen2.5-0.5B-Instruct 固定 revision、PEFT 0.17.1、transformers 4.57.1，在同一参考 GPU 上完整执行。正式训练为 60 次更新，LoRA 可训练参数 540,672，总参数 494,573,440。小批量诊断损失下降后，适配器和优化器已重置再开始正式训练。

- 冻结基础模型：12/24 正确，50%；LoRA：20/24 正确，83.3%。合法格式比例均为 100%。
- 测试集只有 6 个语义模板组，每组四种编号变化。24 条记录不能视为 24 个独立语义样本，结果不证明真实工单泛化能力。
- LoRA 仍把 `The receipt shows a different price from checkout.` 归到物流，四条编号变体均错误。错误完整保留。
- 重新加载 adapter 后，固定验证输入的 logits 最大绝对差为 0。
- 第一次 BF16 直接合并未通过原先的 0.02 相对 L2 检查线，误差为 0.04242。未放宽阈值；改在 FP32 下检查合并，误差为 0.000003168，通过预设 1e-4 阈值。
- 导出 FP32 合并产物，在测试集上与 BF16 LoRA 的答案无分歧。这个答案一致性不代表所有输入或数值逐位一致。

证据：[manifest](../experiment17_output/manifest.json)、[合并与重载检查](../experiment17_output/artifact_checks.json)、[逐题预测](../experiment17_output/predictions.csv)、[合并模型预测](../experiment17_output/merged_predictions.csv)、[数据划分](../experiment17_output/dataset.json)。

## 第 18 课：本地 vLLM 流式服务

服务显式将 FP32 产物加载为 BF16，vLLM 0.11.0，仅监听 127.0.0.1。216 个计分请求加 1 个 warm-up 请求全部完成，计分请求无 HTTP/流式协议失败，与 transformers 合并模型答案无分歧。服务准确率为 83.3%，反映第 17 课同一组错误，没有把“服务成功”等同于“任务全对”。服务退出后已确认端口不再监听。

| 并发 | 计分请求 | 输出 token/s | TTFT p95（ms） | 端到端 p95（ms） |
|---|---:|---:|---:|---:|
| 1 | 72 | 106.7 | 22.5 | 45.1 |
| 2 | 72 | 190.4 | 26.6 | 48.7 |
| 4 | 72 | 299.2 | 39.4 | 60.8 |

这是短输出、关闭前缀缓存、eager 模式、闭环固定并发下的客户端测量。请求开始前在线程池内的等待不计入逐请求延迟；总吞吐窗口包含整个批次。不能将这些数字当作长输出、固定外部到达率或生产 SLA。启动耗时约 30 秒，单独记录，未计入稳态吞吐。

证据：[逐请求指标](../experiment18_output/requests.csv)、[分轮窗口](../experiment18_output/windows.csv)、[汇总](../experiment18_output/summary.csv)、[协议](../experiment18_output/protocol.json)、[答案回归](../experiment18_output/regression.json)。
