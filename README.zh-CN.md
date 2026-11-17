# verl GPU 中文交互教程

[English](README.md) | **简体中文**

欢迎使用中文学习入口。课程正文为中文，通过 Jupyter Notebook 逐格运行代码，学习 GPU 计算、监督微调、奖励学习、评估、复现和真实 verl 单卡训练。

**[从课程首页开始](tutorial/00_START_HERE.ipynb)** · [详细目录](tutorial/README.md) · [高级课程](tutorial/ADVANCED.md)

在 GitHub 上可以阅读教程源代码和说明；要执行代码、显示实时曲线和使用交互按钮，需要在本地启动 JupyterLab。

## 学习路线

| 课程 | 内容 |
|---|---|
| [01：GPU 与奖励学习入门](tutorial/verl_gpu_beginner.ipynb) | GPU 检查、CPU/GPU 对比与策略梯度 |
| [02：微型语言模型训练](tutorial/02_tiny_language_model.ipynb) | 分词、监督训练、生成与测试 |
| [03：组内奖励学习](tutorial/03_group_reward_learning.ipynb) | GRPO 风格教学循环、优势、裁剪与 KL |
| [04：三组训练对照](tutorial/04_visible_learning_comparison.ipynb) | 冻结基线、继续 SFT 与奖励微调 |
| [05：交互评估台](tutorial/05_model_playground.ipynb) | 输入题目、比较答案与泛化边界 |
| [06：扩大数据覆盖](tutorial/06_data_coverage.ipynb) | 数据划分、格式增强与独立评估 |
| [07：逐位推理](tutorial/07_column_reasoning.ipynb) | 局部规则与人工竖式算法的组合 |
| [08：采样与奖励漏洞](tutorial/08_sampling_and_rewards.ipynb) | 温度、多样性与奖励函数检查 |
| [09：保存与恢复](tutorial/09_checkpoints_and_reproducibility.ipynb) | 优化器、随机状态与可复现续训 |
| [10：连接真实 verl](tutorial/10_verl_bridge_and_graduation.ipynb) | tokenizer、Parquet、奖励接口与结业检查 |
| [11：真实单步 GRPO](tutorial/11_real_verl_training.ipynb) | Ray、vLLM、FSDP 更新与检查点 |
| [12：稀疏奖励诊断](tutorial/12_sparse_reward_diagnosis.ipynb) | 真实任务的冷启动失败与指标分析 |

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

5. 打开终端打印的本机链接，进入 `00_START_HERE.ipynb`，选择 **Python (verl GPU)** 内核，从上到下按 **Shift + Enter** 执行。

交互按钮只有在相关代码格执行后才能使用。完成一课后，可通过 **Kernel → Shut Down Kernel** 释放资源；仅关闭网页标签不一定会关闭内核。

高级脚本默认使用 WSL 环境 `/opt/verl-venv` 和 `/opt` 输出目录，请按自己的机器调整。`/path/to/verl` 是路径占位符。第 11、12 课默认读取本地已完成训练，首次使用应先在 verl 根目录运行对应脚本：

```bash
bash tutorial/advanced11/run.sh
# 学习第 12 课时再执行：
bash tutorial/advanced12/run.sh
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
