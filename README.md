# verl GPU 中文交互教程

**简体中文** · [English](README.en.md)

通过 Jupyter Notebook 逐格运行代码，学习 GPU 计算、监督微调、奖励学习、评估、复现和真实 verl 单卡训练。

**[从课程首页开始](tutorial/00_课程导航.ipynb)** · [详细目录](tutorial/support/docs/README.md) · [高级课程](tutorial/support/docs/ADVANCED.md)

在 GitHub 上可以阅读教程源代码和说明；要执行代码、显示实时曲线和使用交互按钮，需要在本地启动 JupyterLab。

## 学习路线

| 课程 | 内容 |
|---|---|
| [01：GPU 与奖励学习入门](tutorial/01_从GPU到verl的第一步.ipynb) | GPU 检查、CPU/GPU 对比与策略梯度 |
| [02：微型语言模型训练](tutorial/02_训练微型语言模型.ipynb) | 分词、监督训练、生成与测试 |
| [03：组内奖励学习](tutorial/03_用组内奖励微调语言模型.ipynb) | GRPO 风格教学循环、优势、裁剪与 KL |
| [04：三组训练对照](tutorial/04_对比冻结模型、监督训练与奖励微调.ipynb) | 冻结基线、继续 SFT 与奖励微调 |
| [05：交互评估台](tutorial/05_用交互评估台检验模型.ipynb) | 输入题目、比较答案与泛化边界 |
| [06：扩大数据覆盖](tutorial/06_用数据覆盖改善泛化.ipynb) | 数据划分、格式增强与独立评估 |
| [07：逐位推理](tutorial/07_逐位推理与长整数加法.ipynb) | 局部规则与人工竖式算法的组合 |
| [08：采样与奖励漏洞](tutorial/08_采样温度与奖励漏洞.ipynb) | 温度、多样性与奖励函数检查 |
| [09：保存与恢复](tutorial/09_保存检查点与恢复训练.ipynb) | 优化器、随机状态与可复现续训 |
| [10：连接真实 verl](tutorial/10_将数据和模型接入verl.ipynb) | tokenizer、Parquet、奖励接口与结业检查 |
| [11：真实单步 GRPO](tutorial/11_运行单步GRPO训练.ipynb) | Ray、vLLM、FSDP 更新与检查点 |
| [12：稀疏奖励诊断](tutorial/12_诊断稀疏奖励.ipynb) | 真实任务的冷启动失败与指标分析 |
| [13：目标函数与梯度](tutorial/13_从目标函数推导策略梯度.ipynb) | SFT 对齐、策略梯度、裁剪、掩码与 KL |
| [14：SFT 预热与 verl](tutorial/14_用监督微调预热GRPO.ipynb) | 修复冷启动、固定评估与真实训练接口 |
| [15：评估不确定性](tutorial/15_估计评估结果的不确定性.ipynb) | 配对比较、bootstrap、多种子与实验设计 |
| [16：显存与吞吐](tutorial/16_测量显存与训练吞吐.ipynb) | 资源预算、CUDA 计时、混合精度与瓶颈 |
| [17：预训练模型 LoRA](tutorial/17_用LoRA微调和导出预训练模型.ipynb) | 聊天模板、回答掩码、适配器与合并验证 |
| [18：本地流式服务](tutorial/18_测试流式推理服务与并发性能.ipynb) | vLLM、HTTP 流式响应、并发与答案回归 |

希望独立开展大模型工程实践，请结合[工程学习路线与考核](tutorial/support/docs/ENGINEERING_ROADMAP.md)。17/18 已补入预训练模型 LoRA 和本地服务实验；多卡恢复、更大模型及生产负载仍需单独验证。

第 17 课首次运行下载约 1GB 的固定版本 Qwen2.5-0.5B-Instruct，需 PEFT；基础模型与合并模型占用额外磁盘，权重在仓库外保存，遵循模型自身许可。第 18 课只启动临时本机服务，测试结束后自动关闭，不开放公网访问。

另有独立的 [slime 学习系列](slime_tutorial/README.md)，讲解 Megatron 训练、SGLang rollout、Ray 编排、Sample/奖励契约和分离调试。该系列使用固定版本的上游 slime checkout，与 verl 课程使用不同环境。

首次学习建议按顺序执行，各课之间有模型和数据依赖。Notebook 已清空运行输出，模型权重和检查点需要在本地生成；仓库中的小型 JSON/CSV 保留历史实测结果供参考。

## 本地运行

本仓库是教程资源包，不包含 verl 框架源码。先按照 [verl v0.7.1 官方说明](https://github.com/volcengine/verl/tree/v0.7.1) 配置 Linux 或 WSL2 CUDA 环境。

1. 克隆官方 verl v0.7.1 和本仓库。
2. 将本仓库的 `tutorial/` 和 `local_smoke/` 复制到官方 verl 项目根目录。
3. 在已经安装 verl 的 Python 环境中安装 Notebook 组件并注册内核：

   ```bash
   python -m pip install jupyterlab ipykernel ipywidgets matplotlib nbclient
   python -m ipykernel install --user --name verl-gpu --display-name "Python (verl GPU)"
   ```

4. 在 verl 项目根目录启动：

   ```bash
   jupyter lab --notebook-dir=tutorial
   ```

5. 打开终端打印的本机链接，进入 `00_课程导航.ipynb`，选择 **Python (verl GPU)** 内核，从上到下按 **Shift + Enter** 执行。

交互按钮只有在相关代码格执行后才能使用。完成一课后，可通过 **Kernel → Shut Down Kernel** 释放资源；仅关闭网页标签不一定会关闭内核。

高级脚本默认使用 WSL 环境 `/opt/verl-venv` 和 `/opt` 输出目录，请按自己的机器调整。`/path/to/verl` 是路径占位符。第 11、12 课默认读取本地已完成训练，首次使用应先在 verl 根目录运行对应脚本：

```bash
bash tutorial/support/advanced11/run.sh
# 学习第 12 课时再执行：
bash tutorial/support/advanced12/run.sh
```

## 环境与结果边界

参考环境为 RTX 4060 Laptop 8GB、WSL2 Ubuntu 24.04、Python 3.12、PyTorch 2.8.0/CUDA 12.8、verl 0.7.1、vLLM 0.11.0、transformers 4.57.1。请遵循官方依赖兼容要求；同样显存容量不保证任意模型和配置都能运行。

- 第 03/04 课是教学版 PyTorch GRPO 风格循环，不等同于完整 verl trainer。
- 第 07 课由程序提供竖式组合规则，不代表语言模型自行发现了长加法算法。
- 第 11 课已实测真实参数更新和检查点保存，合成奖励只验证工程流程。
- 第 12 课保留稀疏奖励冷启动失败的真实结果，不宣称任务准确率提高。
- 小数据集、单次运行不能证明统计显著性或通用能力。历史结果也不是不同设备上的性能保证。

## 许可与来源

本教程及配套脚本采用 [MIT 协议](LICENSE)。verl 框架需单独安装，其许可证保持不变。本项目为独立教学项目，并非官方教程或上游背书。
