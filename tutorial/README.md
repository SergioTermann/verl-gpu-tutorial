# verl GPU 中文交互教程

## 高级部分

- `12_sparse_reward_diagnosis.ipynb`：有明确含义的奇偶判断奖励，10 步真实 verl 训练及固定测试评估。随机冷启动的稀疏奖励失败案例，包含合法输出率、梯度和训练前后预测诊断；脚本在 `advanced12/`，报告在 `experiment12_output/`。

- `11_real_verl_training.ipynb`：真实 Ray/vLLM/FSDP 单步 GRPO，已检查梯度、参数变化、优化器状态和检查点。随机微型模型与字节和合成奖励仅验证工程链路，不衡量任务质量。默认展示最近实测，设 `RUN_TRAINING=True` 才会新训练。
- 配套脚本在 `advanced11/`；每轮数据与权重保存在 WSL `/opt/verl-advanced11/runs/`，最近运行路径见 `advanced11/latest_run.txt`，报告副本见 `experiment11_output/verification.json`。
- 本机配置将奖励 worker 从默认 8 降为 1，解决首次试跑的主机内存压力。后续每个独立 run 目录均保存完整日志。

基础课程共 10 课，统一入口：`00_START_HERE.ipynb`。启动脚本默认打开课程首页，首次按顺序学习；已经完成 01–07 时可直接运行 08–10。

## 实验目录

- `verl_gpu_beginner.ipynb`：第一课，GPU 验证与策略梯度原理。
- `02_tiny_language_model.ipynb`：第二课，从零训练微型 Qwen2 生成加法答案，包含实时曲线、独立测试集、错误分析与模型保存。无需下载模型；这是监督训练实验，不是完整 verl RL 训练。
- `03_group_reward_learning.ipynb`：第三课，加载第二课权重，运行教学版 GRPO 风格奖励微调；展示组采样、优势、策略裁剪、参考策略 KL、冻结基线与改善/退化题目。必须先完成第二课保存格。它是 PyTorch 教学实现，不是完整 verl 训练器运行。
- `04_visible_learning_comparison.ipynb`：第四课，从只预热 200 步的相同模型出发，对比冻结基线、继续 SFT、奖励 RL；固定测试集，显示三组准确率和全部测试题答案。输出写入 `experiment04_output/`，保留前三课结果。这是单种子教学演示，改变了第三课的初始条件。
- `05_model_playground.ipynb`：第五课，加载第四课的三个模型，输入数字并排比较真实 GPU 生成；评估原测试题、前导零、新数字范围和三位答案，保存全部预测与权重哈希到 `experiment05_output/`。不训练、不修改权重。运行第 5 节代码格后可使用交互按钮。
- `06_data_coverage.ipynb`：第六课，用 0–69 数字范围和前导零格式继续 SFT；从未在先前训练或评估中出现的数对构建新验证/最终测试集，另测 70–99 外推。训练只监测验证集，固定步数后做最终评估。模型、划分和完整预测保存到 `experiment06_output/`。
- `07_column_reasoning.ipynb`：第七课，学习全部 200 个单列数字/进位状态，再用人工编写的竖式循环组合；测试长数字与连续进位，并提供逐列交互面板。明确区分学习的局部规则与人为算法，不声称 Qwen 自己发现了加法。结果保存到 `experiment07_output/`。
- `08_sampling_and_rewards.ipynb`：第八课，真实模型温度采样对比、人工奖励漏洞样例和严格奖励测试。
- `09_checkpoints_and_reproducibility.ipynb`：第九课，连续训练与完整状态恢复比较，展示只恢复权重为何不能准确续训。
- `10_verl_bridge_and_graduation.ipynb`：第十课，构建真实 tokenizer/model 包、Parquet 和奖励文件，通过 verl 数据加载器及配置检查，完成课程产物检查与理解题。不会启动完整 Ray/vLLM 训练。

基础教学在第十课结束；完整大模型 RL、LoRA、分布式与部署属于进阶部分。`build_foundations_final.py` 会覆盖首页和 08–10 原始教程，日常学习不要运行生成脚本。

第三课结果保存到 `experiment03_output/`，不覆盖第二课。重跑完整实验应从第一格重新加载起点；`build_experiment03.py` 只用于重新生成教程，会覆盖 Notebook 修改和输出。

第二课结果和权重保存在 `experiment02_output/`；重跑保存格会覆盖该目录中的对应文件。`build_experiment02.py` 是教程生成脚本，日常学习无需运行，否则会覆盖 Notebook 修改和输出。

在项目目录的 PowerShell 中运行 ` .\start-notebook.ps1 `，打开终端输出的带 token 的本机链接。

打开 `verl_gpu_beginner.ipynb`，选择 **Python (verl GPU)** 内核，从上到下按 **Shift + Enter** 执行。保持启动终端运行，结束时按 Ctrl+C。

教程包括 GPU/FlashAttention 检查、CPU/GPU 对比、verl DataProto 和奖励、实时 GPU 策略梯度曲线、真实 verl 配置加载。小训练是 REINFORCE 原理演示，不是完整语言模型 PPO/GRPO 训练。

运行环境：WSL Ubuntu-24.04 `/opt/verl-venv`。JupyterLab 4.6.6；安装时保留原有训练依赖版本，`uv pip check` 通过。

Notebook 保存了实测输出，重新运行即可更新。`build_notebook.py` 用于从 `lessons.py` 重新生成初始教程，会覆盖 Notebook 及其输出；日常学习直接编辑 Notebook 即可，不要运行生成脚本。
