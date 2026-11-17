"""Build the notebook with visible, editable code rather than hidden runners."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT / 'lessons.py').read_text(encoding='utf-8'))
functions = {n.name: ast.unparse(ast.Module(body=n.body, type_ignores=[])) for n in tree.body if isinstance(n, ast.FunctionDef)}
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None})


md('''# 从 GPU 到 verl：动手入门

本课使用 verl 0.7.1，参考 GPU 为 RTX 4060 8GB，无需下载模型。

先检查 GPU 是否可用，再比较 CPU 与 GPU 的计算时间，最后训练一个只有四个选项的小策略，观察奖励如何改变它的选择。预计 15 至 25 分钟；实际计算通常只需几分钟。

这是 JupyterLab 网页中的 Notebook。点击代码格，再按 **Shift + Enter**：运行当前格并前进。**Ctrl + Enter**：运行并停留。左侧 `[*]` 表示运行中，数字表示已执行。上方 ■ 可中断当前计算。

请从上到下运行。修改代码后可以重跑；要清空内存状态，选择 **Kernel → Restart Kernel and Run All Cells**。保存用 **Ctrl + S**。公开版已清空运行输出。执行代码后，页面会显示本次运行的结果；保存后再次打开时，请注意区分历史输出与新结果。

右上角内核请选择 **Python (verl GPU)**。GPU 在本机 WSL 中计算，网页负责显示结果。

## 0 · 准备工具

下一格只导入工具。Notebook 会保留变量，因此后面的格子可以使用这里的导入。''')
code('''import sys, time, statistics
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def emit(**values):
    print(values)

print('Python:', sys.executable)
assert sys.executable.startswith('/opt/verl-venv/'), '请选择 Python (verl GPU) 内核'
print('准备完成。接下来查看真实显卡。')''')
md('''## 1 · 确认代码真的在 GPU 上运行

CUDA 是 PyTorch 使用 NVIDIA GPU 的计算接口。`device="cuda"` 才会把张量放到 GPU 上；安装好 GPU 版本不意味着所有代码会自动加速。

**观察：** 核对输出中的 GPU 型号和 CUDA 版本，并查看 FlashAttention 检查是否显示 PASS。GPU 型号应与你的设备一致。FlashAttention 是注意力运算的优化实现。''')
code(functions['environment'])
md('''## 2 · 测一次 CPU / GPU 加速

矩阵乘法是神经网络的重要基础操作。这里让 CPU 和 GPU 计算同一组 FP32 矩阵，先预热，再测 10 次，比较中位数。

GPU 运算通常异步提交，因此计时前后需要同步。这个实验**不含 CPU↔GPU 数据传输**，也不是完整训练的速度测试。后台负载、温度和电源模式会影响结果。

**动手：** 运行后把矩阵尺寸从 2048 改成 512 再试。小任务中 GPU 的启动开销可能抵消优势。''')
code(functions['benchmark'])
code('''fig, ax = plt.subplots(figsize=(6, 2.8))
ax.barh(['CPU', 'GPU'], [times['cpu'], times['cuda']], color=['#94a3b8', '#10b981'])
ax.set_xlabel('Median matrix multiplication time (ms); lower is better')
ax.set_title('Your measured result')
plt.tight_layout()
plt.show()''')
md('''## 3 · verl 负责什么？先理解奖励

大语言模型强化学习的基本循环是：**输入问题 → 模型生成回答（rollout）→ 给回答打分（reward）→ 更新策略（actor）→ 再生成**。

verl 负责组织这个训练流程；vLLM 常用于生成回答，PyTorch 用于计算梯度和更新参数，Ray 用于协调工作进程。

GRPO 会为同一个问题生成一组回答，用组内奖励比较来估计优势。下面先创建真实的 `verl.DataProto` 数据容器，再用四个回答展示简化的组内优势。它不是完整 GRPO 实现。

**观察：** 高于组内平均奖励的回答有正优势。如果所有回答奖励相同，组内比较就无法区分优劣。''')
code(functions['reward'])
md('''## 4 · 看策略在 GPU 上一点点学会选择

现在执行一个小型 **REINFORCE 策略梯度实验**。对固定问题 `2+2=?`，策略从 `[3,4,5,6]` 采样；答对奖励为 1，否则为 0。用 `loss = -log_prob × reward` 计算损失、反向传播并更新参数。

这里只训练四个可学习分数，**没有训练语言模型，也没有调用 verl 训练器**。它用于理解奖励如何影响参数，且无需下载模型。与完整 PPO/GRPO 相比，它省略了策略裁剪、KL 约束、序列和分布式训练。

下方曲线会随计算实时更新。绿色表示答对的策略概率，灰色表示当前采样批次的实际奖励；采样奖励会波动。''')
code('''history = []
chart = display('', display_id=True)

def emit(**values):
    history.append(values)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot([v['step'] for v in history], [v['probability'] for v in history], color='#059669', label='P(correct)')
    ax.plot([v['step'] for v in history], [v['reward'] for v in history], color='#94a3b8', alpha=.7, label='Sample reward')
    ax.set(xlabel='Training step', ylabel='Probability / reward', ylim=(0, 1.05), title='Live GPU policy learning')
    ax.legend(loc='lower right')
    ax.grid(alpha=.15)
    fig.tight_layout()
    chart.update(fig)
    plt.close(fig)

''' + functions['train'])
md('''### 小练习

1. 把学习率 `lr=0.08` 改成 `0.01`，重新运行第 4 节的代码格，观察相同步数内学习速度的变化。
2. 把 `rewards = (actions == 1).float()` 改成奖励第 0 个选项，会发生什么？同时修改显示标签和最终断言，让它们检查新的目标。
3. 想一想：如果奖励函数把错误答案评为高分，策略会学到什么？

每次重跑该格都会重新初始化策略。曲线变好只表示学会了这个固定任务，不能证明泛化能力。

## 5 · 加载真正的 verl 配置

下一格调用真实的 `verl.trainer.main_ppo`，使用 `--cfg job` **仅打印配置**，不会启动完整训练。

| 配置 | 作用 |
|---|---|
| `algorithm.adv_estimator=grpo` | 选择 GRPO 优势估计 |
| `actor_rollout_ref.model.path` | 初始模型目录或模型名 |
| `data.train_files` | 训练 Parquet 文件 |
| `actor_rollout_ref.rollout.n` | 每个提示生成的回答数；组内比较通常需要大于 1 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 每张 GPU 单次处理的小批量大小 |
| `trainer.n_gpus_per_node=1` | 每台机器使用一张 GPU |

**观察：** 配置打印成功代表训练入口可用，不代表某个具体模型已经完成训练。''')
code(functions['config'])
md('''## 6 · 从这个实验走向真实训练

真实训练还需要：一个预训练语言模型、符合格式的数据、能表达任务目标的奖励函数，以及适配显存的训练参数。本机仓库的 `local_smoke/` 有一个随机小 Qwen2 的试运行脚本；随机模型只适合检查流程，不适合作为问答能力示例。本教程不把它标为已通过的完整训练。

本教程所用版本的官方 GSM8K 快速入门要求 **至少 24GB GPU 显存**。使用 8GB 显卡时，需要调整批量和序列长度；需要另外验证短序列、小批量、LoRA 或卸载等具体配置。小模型也不能仅凭参数量保证整个训练流程放得下。

参考资料（网页可能随版本更新，本教程依据本机 verl 0.7.1）：

- [verl 官方文档](https://verl.readthedocs.io/en/latest/)
- [官方 GSM8K 快速入门](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [JupyterLab Notebook 操作](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html)
- 官方 verl 源码中的 `docs/start/quickstart.rst`，以及本教程提供的 `local_smoke/run.sh`

### 常见问题

- **找不到 torch / verl**：切换到 `Python (verl GPU)` 内核。
- **长时间 `[*]`**：首次导入可能要十几秒；查看下方输出，需要时用 ■ 中断。
- **显存不足**：先重启内核释放本教程占用，关闭其他 GPU 计算，再降低实验规模。
- **网页断开**：保持启动 Notebook 的终端运行，重新打开终端打印的本机链接。
- **曲线是旧的**：保存后的文件可能显示上次结果；重跑第 4 节即可生成本次曲线。

完成后，你应该能解释：代码怎样进入 GPU、奖励怎样改变策略，以及 verl 如何把这些步骤组织成语言模型训练流程。''')
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python (verl GPU)', 'language': 'python', 'name': 'verl-gpu'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'lesson-{i:02d}'
(ROOT / 'verl_gpu_beginner.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
