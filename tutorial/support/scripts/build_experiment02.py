"""Generate the second standalone executable tutorial."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append(dict(cell_type='markdown', metadata={}, source=source))


def code(source):
    cells.append(dict(cell_type='code', metadata={}, source=source, outputs=[], execution_count=None))


md('''# 实验 02 · 训练一个会生成答案的小语言模型

上一课的策略只需在四个选项中选择。本课使用 Qwen2 因果 Transformer，从随机参数开始训练，让模型逐个字符生成加法答案。例如输入 `12+7=`，输出 `19`。

**目标：** 理解分词、训练/测试划分、答案位置的交叉熵损失、反向传播与自回归生成。无需下载模型权重，已在参考环境的 8GB GPU 上运行。预计阅读 20 分钟，训练约 1 至 3 分钟，实际以机器状态为准。

这是一个小语言模型的监督训练（SFT 原理），**没有运行 verl 的 PPO/GRPO 训练器**。它是理解后续“生成答案 → 奖励 → RL 更新”的准备实验。模型从零学习有限范围的数据，不保证学会通用加法。

选择 **Python (verl GPU)** 内核，从上到下按 **Shift + Enter**。运行后会显示当前实验的输出；重复执行会更新页面上的结果。训练中可以点击 ■ 中断。不要与上一课同时运行 GPU 训练。

## 1 · 固定实验条件

先设置随机种子。重复运行整本教程会重新初始化模型；CUDA 不同版本仍可能有细小数值差异。''')
code('''import sys, time, random, json
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display
from transformers import Qwen2Config, Qwen2ForCausalLM

assert sys.executable.startswith('/opt/verl-venv/'), '请选择 Python (verl GPU)'
assert torch.cuda.is_available(), '需要 CUDA GPU'
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
device = 'cuda'
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__)''')
md('''## 2 · 字符分词与独立测试集

token 是模型处理文本的基本单位。本课每个字符对应一个 token：词表包含数字、`+`、`=`、结束符 `#` 和填充符 `_`。例如，`12+7=19#` 会按字符拆分并编码。

在 0 至 19 的整数范围内共有 400 个有序题目。按无序数对分组划分 80% 训练、20% 测试：如果 `3+7` 在测试集，`7+3` 也必须在测试集，防止交换顺序造成泄漏。测试题不会参与梯度更新。

训练标签在问题和填充位置设为 `-100`，仅在答案和结束符位置计算损失。模型内部会把 logits 与标签错开一位，以“前面的字符预测下一个字符”。''')
code('''vocab = '_0123456789+=#'
stoi = {c: i for i, c in enumerate(vocab)}
PAD, EOS = stoi['_'], stoi['#']
MAX_LEN = 10
def encode(text):
    return [stoi[c] for c in text]

groups = [(a, b) for a in range(20) for b in range(a, 20)]
random.Random(SEED).shuffle(groups)
heldout = set(groups[:42])
train_pairs, test_pairs = [], []
for a in range(20):
    for b in range(20):
        (test_pairs if tuple(sorted((a, b))) in heldout else train_pairs).append((a, b))
assert not {tuple(sorted(p)) for p in train_pairs} & {tuple(sorted(p)) for p in test_pairs}

def pack(pairs):
    ids = torch.full((len(pairs), MAX_LEN), PAD, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (a, b) in enumerate(pairs):
        prompt = f'{a}+{b}='
        sequence = encode(prompt + str(a+b) + '#')
        ids[i, :len(sequence)] = torch.tensor(sequence)
        labels[i, len(prompt):len(sequence)] = ids[i, len(prompt):len(sequence)]
        mask[i, :len(sequence)] = 1
    return ids.to(device), labels.to(device), mask.to(device)

train_ids, train_labels, train_mask = pack(train_pairs)
print('训练题数:', len(train_pairs), '| 独立测试题数:', len(test_pairs))
print('字符:', list('12+7=19#'), '\\ntoken:', encode('12+7=19#'))
print('示例 labels:', train_labels[0].tolist())''')
md('''## 3 · 创建微型 Qwen2，并记录训练前基线

使用两层、隐藏维度 128 的随机 Qwen2。它与预训练 Qwen 聊天模型不是同一组权重，不具备语言理解能力。

下面的生成函数只接收题目，不接收正确答案。按相同提示长度分组，逐步选择概率最大的下一个字符，最多生成 3 个 token（两位答案 + 结束符）。评估采用完整答案字符串精确匹配。

这里使用 PyTorch SDPA 注意力，便于把重点放在训练逻辑。''')
code('''config = Qwen2Config(vocab_size=len(vocab), hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=32, pad_token_id=PAD, eos_token_id=EOS,
    tie_word_embeddings=True, attention_dropout=0.0)
model = Qwen2ForCausalLM(config).to(device)
model.config.use_cache = False
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

@torch.no_grad()
def predict(pairs):
    model.eval()
    results = [''] * len(pairs)
    by_length = {}
    for i, (a, b) in enumerate(pairs):
        prompt = f'{a}+{b}='
        by_length.setdefault(len(prompt), []).append((i, prompt))
    for length, rows in by_length.items():
        x = torch.tensor([encode(p) for _, p in rows], device=device)
        for _ in range(3):
            nxt = model(input_ids=x).logits[:, -1].argmax(-1, keepdim=True)
            x = torch.cat((x, nxt), dim=1)
        for (i, _), tokens in zip(rows, x[:, length:].tolist()):
            results[i] = ''.join(vocab[t] for t in tokens).split('#')[0]
    return results

def accuracy(pairs):
    answers = predict(pairs)
    return sum(s == str(a+b) for s, (a,b) in zip(answers, pairs)) / len(pairs)

before_train = accuracy(train_pairs)
before_test = accuracy(test_pairs)
print(f'训练前：训练集 {before_train:.1%} / 测试集 {before_test:.1%}')''')
md('''## 4 · 开始训练：观察损失与准确率

每一步从训练集抽取 64 道题，用交叉熵训练答案 token。`loss.backward()` 计算梯度，`optimizer.step()` 修改模型参数。

每 100 步评估一次，曲线实时刷新。训练损失是最近 100 步的均值；准确率是整个训练/测试集的贪心生成结果。测试集只做评估；本实验固定跑完 800 步，不按测试分数挑选检查点。

**观察重点：** 训练准确率与测试准确率的差距。训练分数上升而测试停滞，说明可能只是在记忆见过的题目。''')
code('''STEPS = 800
BATCH = 64
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
history = [dict(step=0, loss=None, train=before_train, test=before_test)]
live = display('', display_id=True)
start = time.perf_counter()
losses = []
torch.cuda.reset_peak_memory_stats()
for step in range(1, STEPS + 1):
    model.train()
    idx = torch.randint(len(train_pairs), (BATCH,), device=device)
    loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]).loss
    assert torch.isfinite(loss), '损失出现 NaN/Inf'
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
    if step % 100 == 0:
        row = dict(step=step, loss=sum(losses)/len(losses), train=accuracy(train_pairs), test=accuracy(test_pairs))
        history.append(row)
        losses = []
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        axes[0].plot([r['step'] for r in history[1:]], [r['loss'] for r in history[1:]], color='#2563eb')
        axes[0].set(title='Answer-token cross entropy', xlabel='Step', ylabel='Loss')
        for key, color in [('train', '#059669'), ('test', '#ea580c')]:
            axes[1].plot([r['step'] for r in history], [r[key] for r in history], label=key, color=color)
        axes[1].set(title='Exact-match generation accuracy', xlabel='Step', ylim=(0,1.05))
        axes[1].legend()
        fig.tight_layout()
        live.update(fig)
        plt.close(fig)
        print(f"step {step:3d} | loss {row['loss']:.4f} | train {row['train']:.1%} | test {row['test']:.1%}", flush=True)
torch.cuda.synchronize()
elapsed = time.perf_counter()-start
print(f'用时 {elapsed:.1f} 秒；PyTorch 峰值已分配显存 {torch.cuda.max_memory_allocated()/2**20:.1f} MiB（非整卡占用）')''')
md('''## 5 · 检查生成答案与错误样例

先固定展示测试集前 12 道题，再列出错误例子。示例选择不依赖答案好坏；错误表可帮助判断模型是否只是学到了输出格式。

你还可以编辑最后一行的问题。超过 0 至 19 的输入属于分布外测试，即使算对一题也不能说明普遍泛化。''')
code('''test_answers = predict(test_pairs)
print('测试题        生成答案   正确答案   正确?')
for (a,b), answer in list(zip(test_pairs, test_answers))[:12]:
    print(f'{a:2d}+{b:<2d}=        {answer!r:8s}   {a+b:<3d}      {answer == str(a+b)}')
failures = [(a,b,s) for (a,b),s in zip(test_pairs,test_answers) if s != str(a+b)]
print(f'测试错误：{len(failures)}/{len(test_pairs)}')
for a,b,s in failures[:10]:
    print(f'  {a}+{b} → {s!r}，正确为 {a+b}')
my_questions = [(3, 7), (12, 7), (19, 19), (20, 5)]
for (a,b),s in zip(my_questions,predict(my_questions)):
    split = '训练集' if (a,b) in train_pairs else '测试集' if (a,b) in test_pairs else '分布外'
    print(f'[{split}] {a}+{b}= → {s!r}，正确答案 {a+b}')''')
md('''## 6 · 保存权重与可复查的实验记录

保存目录为 Notebook 所在目录下的 `support/experiment02_output/`。重复运行本格会覆盖该实验的上次保存；与第一课和 verl 的原始训练脚本无关。

记录包括训练/测试划分、超参数、每次评估的结果及生成答案。下一次实验可以加载这个模型，再研究奖励函数与策略更新。''')
code('''out = Path('support/experiment02_output')
out.mkdir(exist_ok=True)
model.save_pretrained(out / 'model')
record = dict(seed=SEED, steps=STEPS, batch_size=BATCH, learning_rate=0.002,
    vocab=vocab, train_pairs=train_pairs, test_pairs=test_pairs,
    test_answers=test_answers, history=history, elapsed_seconds=elapsed,
    torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0))
(out/'results.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print('已保存:', out.resolve())
print(f"最终训练准确率 {history[-1]['train']:.1%}；独立测试准确率 {history[-1]['test']:.1%}")''')
md('''## 7 · 复盘与下一步

- **token 和数据：** 提示和答案都编码成 token，但本实验只对答案计算损失。
- **SFT 与 RL：** 本次直接给出标准答案，用交叉熵学习；RL 则对生成结果打分，再用奖励更新策略。
- **泛化边界：** 这是一组有限加法题、一个随机种子和固定切分。不能据此推断大模型训练效果；多次按测试成绩调参后，还需要新的最终测试集。
- **verl 的对应关系：** 模型可对应 actor，题目对应 prompt，生成过程对应 rollout，精确匹配可作为 reward。完整 verl 训练还需要模型/tokenizer 接口、数据格式和训练配置的集成，当前自定义字符词表不能直接当成完整 Hugging Face tokenizer 使用。

### 练习

先保存当前 `results.json` 的副本，再改 `STEPS=200`，从第 1 格开始重跑，比较欠训练时的结果。之后可尝试学习率 `0.001`，保持数据划分不变。每次都记录实际结果，不要只保留最好的一次。

参考：[PyTorch 训练入门](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[Qwen2 模型接口](https://huggingface.co/docs/transformers/model_doc/qwen2)。''')
for i, cell in enumerate(cells):
    cell['id'] = f'exp02-{i:02d}'
notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
    metadata=dict(kernelspec=dict(display_name='Python (verl GPU)', language='python', name='verl-gpu'),
                  language_info=dict(name='python', version='3.12')))
(Path(__file__).resolve().parents[2] / '02_tiny_language_model.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
