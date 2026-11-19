"""Rebuild lessons 13-16; generated notebooks intentionally contain no outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip()+'\n'))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.strip()+'\n',
                      execution_count=None, outputs=[]))


def save(name):
    md('''---

<a href="https://github.com/SergioTermann"><img src="support/assets/logo.png" alt="SergioTermann" width="44" height="44" align="left" style="margin: 0 12px 0 0;"></a>

**更多有趣的 AI 课程**  
[← 访问 SergioTermann 的 GitHub 主页](https://github.com/SergioTermann)''')
    for index, cell in enumerate(cells):
        cell['id'] = f'{name[:2]}-{index:02d}'
        cell['source'] = cell['source'].splitlines(keepends=True)
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(kernelspec=dict(name='verl-gpu', display_name='Python (verl GPU)', language='python'),
                            language_info=dict(name='python', version='3.12')))
    (ROOT/name).write_text(json.dumps(nb, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    cells.clear()


md(r'''# 实验 13 · 从目标函数推导策略梯度

完成前 12 课后，你已经见过参数更新。本课要求你能解释每个张量为何存在，并通过反例发现实现错误。使用 CPU 即可，不依赖前课权重。

先回答：`loss` 数值相同，梯度一定相同吗？提示 token 是否该计算 SFT 损失？GRPO 的 old policy 和 reference policy 是否相同？先写下判断，运行后再修改。

## 1 · SFT 的标签为何要错开一位

因果语言模型计算 $p_\theta(x_t\mid x_{<t})$。给定提示 $q$ 和回答 $a$，本课使用

$$L_{SFT}=-\frac{1}{N_a}\sum_{t\in\text{answer+EOS}}\log p_\theta(x_t\mid x_{<t}).$$

logits 形状是 `[batch, sequence, vocabulary]`。位置 t 的 logits 预测位置 t+1 的标签；提示与 padding 的标签设成 `-100`，EOS 仍参与学习。Hugging Face 因果模型内部通常已经做 shift，外部再 shift 一次会错位。

下面用独立张量检验对齐，之后才把同一规则交给模型。''')
code('''import json, math
from pathlib import Path
import torch
import torch.nn.functional as F
torch.manual_seed(13)
ids = torch.tensor([[1, 2, 3, 4, 0]])  # 两个提示 token、答案、EOS、padding
labels = torch.tensor([[-100, -100, 3, 4, -100]])
logits = torch.randn(1, 5, 6, requires_grad=True)
# 位置 t 的输出预测位置 t+1，所以预测去掉末位、标签去掉首位。标签为 -100 的提示和填充位置不计入损失。
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
manual = -(logits.log_softmax(-1)[0, 1, 3] + logits.log_softmax(-1)[0, 2, 4])/2
assert torch.allclose(loss, manual)
loss.backward()
assert logits.grad[0, [0, 3, 4]].abs().sum() == 0
print('PASS: 只有预测答案与 EOS 的两个位置收到梯度。', float(loss.detach()))''')
md(r'''## 2 · 策略梯度：奖励不是可微标签

策略目标 $J(\theta)=\mathbb E_{a\sim\pi_\theta}[r(a)]$ 的梯度为
$\mathbb E[r(a)\nabla_\theta\log\pi_\theta(a)]$，奖励函数本身无需可微。用采样到的回答近似这个期望，基线可以降低方差。

本格枚举四个动作，比较精确期望梯度与一个具有相同梯度的 surrogate loss。概率权重必须 `detach()`：采样发生时的分布不应在这条估计式里被再次微分。这不是“所有概率都要 detach”，而是要区分采样权重和被优化的 log probability。''')
code('''scores = torch.tensor([.2, -.1, .4, -.2], dtype=torch.float64, requires_grad=True)
rewards = torch.tensor([0., 1., 0., 0.], dtype=torch.float64)
probs = scores.softmax(-1)
exact = -(probs * rewards).sum()
g_exact, = torch.autograd.grad(exact, scores, retain_graph=True)
# detach 把采样概率当作固定权重，仅让 log 概率贡献梯度。下面用 g_exact 作对照，可以直接看到漏掉 detach 的后果。
surrogate = -(probs.detach() * rewards * probs.log()).sum()
g_surrogate, = torch.autograd.grad(surrogate, scores, retain_graph=True)
wrong = -(probs * rewards * probs.log()).sum()
g_wrong, = torch.autograd.grad(wrong, scores)
assert torch.allclose(g_exact, g_surrogate)
assert not torch.allclose(g_exact, g_wrong)
print('精确梯度:', g_exact.tolist())
print('遗漏 detach 的梯度:', g_wrong.tolist())''')
md(r'''## 3 · GRPO：组内比较与裁剪

本教学定义 $A_i=(r_i-\bar r)/(\sigma_r+10^{-6})$，标准差使用总体版本 `unbiased=False`。不同库的标准差、序列/批次归一化和优势尺度可能不同，迁移代码时必须核对实现。

设 $\rho_t=\exp(\log\pi_\theta-\log\pi_{old})$，策略损失为

$$L=-\operatorname{mean}_{\text{有效回答 token}}\min(\rho_t A,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A).$$

old policy 是采样该批回答的策略快照；reference policy 是提供 KL 约束的参考模型。它们用途不同。old log-prob 与优势应固定，new log-prob 要保留梯度。对负优势，裁剪的有效方向与正优势不同，不能直接把 ratio 裁剪后相乘代替 `min`。

下面测试全对组、全错组、混合组、正负优势和 padding。它是张量单元实验，不是新的 verl trainer。''')
code('''reward_groups = torch.tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 1., 0., 1.]])
advantages = (reward_groups-reward_groups.mean(-1, keepdim=True))/(reward_groups.std(-1, keepdim=True, unbiased=False)+1e-6)
assert advantages[:2].abs().sum() == 0
assert advantages[2, 1] > 0 and advantages[2, 0] < 0
ratios = torch.tensor([[1.4, 1.4, 1.0], [.6, .6, 1.0]], requires_grad=True)
adv = torch.tensor([[1.], [-1.]])
mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]])
# 对正、负优势分别看裁剪方向：本例 1.4 已高于正优势上界，0.6 已低于负优势下界；继续往该方向移动不会增加目标值。
terms = torch.minimum(ratios*adv, ratios.clamp(.8, 1.2)*adv)
clipped_loss = -(terms*mask).sum()/mask.sum()
clipped_loss.backward()
assert ratios.grad.abs().sum() == 0  # 两行都越过了有利方向的裁剪边界
print('优势:', advantages)
print('PASS: 全同奖励无相对信号；有利方向超限后梯度被裁剪。')
# 不能把 seq-mean 与 token-mean 混为同一个目标。
values = torch.tensor([[1., 0., 0.], [3., 3., 3.]])
m = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
token_mean = (values*m).sum()/m.sum()
# 先按每行有效 token 数归一化，再平均各行，每条回答权重相同。对照上一行，算算一长一短两条回答各占多少。
sequence_mean = ((values*m).sum(-1)/m.sum(-1)).mean()
print('token mean:', float(token_mean), 'sequence mean:', float(sequence_mean))
assert token_mean != sequence_mean''')
md(r'''## 4 · KL 估计与代码审查

记 $d=\log\pi_{ref}-\log\pi_\theta$。常用采样项 $e^d-d-1$ 在数值上非负。当样本确实来自当前策略且支持集满足条件时，其期望对应 $D_{KL}(\pi_\theta\|\pi_{ref})$；用旧策略样本或有限批次时，不能直接当作精确 KL。

此处验证离散分布上的期望恒等式。实际序列训练还需要回答掩码、EOS 处理与相应的采样/更新约定。''')
code('''p = torch.tensor([.1, .2, .7], dtype=torch.float64)
q = torch.tensor([.3, .3, .4], dtype=torch.float64)
d = q.log()-p.log()
# 这里 d=log(q/p)，所以 exp(d) 就是 q/p。接下来按 p 加权求和，检验这个估计式与离散分布的精确 KL 是否一致。
k3 = d.exp()-d-1
exact_kl = (p*(p.log()-q.log())).sum()
assert torch.allclose((p*k3).sum(), exact_kl)
out = Path('support/experiment13_output'); out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(dict(shift=True, policy_gradient=True,
    clipping=True, masks=True, kl_identity=True, exact_kl=float(exact_kl)), indent=2))
print('PASS: KL 期望一致。请继续完成下面的代码审查题。')''')
md('''## 5 · 验收：你能否发现“跑得动”的错误

1. 把第 1 格的 label mask 全部取消。哪些 token 改变了训练目标？
2. 把混合组换成一个正确、七个错误。比较组大小变化后的优势尺度。
3. 在裁剪格中把 ratio 改为正优势 0.6、负优势 1.4。先预测梯度是否为零，再运行。
4. 从第 03 课找到 old log-prob、reference log-prob 和 EOS mask。用具体变量解释它们的生命周期。

交付一页审查记录：写出张量形状、被优化的目标、三个可能静默出错的位置。仅通过断言不足以证明完整算法正确。

参考：[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[PPO](https://arxiv.org/abs/1707.06347)、[PyTorch autograd](https://pytorch.org/docs/stable/autograd.html)。下一课把理论用于修复冷启动。''')
save('13_从目标函数推导策略梯度.ipynb')

md('''# 实验 14 · 用监督微调预热 GRPO

第 12 课有 3 步出现非零梯度，但固定测试的合法输出率与准确率均为零。可能的问题包括输出协议未学会、输入含 UNK、奖励过于稀疏，以及训练预算不足。本课先验证一个可检验的假设：监督预热能否让模型稳定输出合法答案？

我们显式改变协议：补上词表中的 `or`，使用三位数，固定新的 240/80/80 训练、验证、测试划分，各组奇偶平衡。三位数未用于第 12 课，但任务的个位规则相同。这是新实验，不把新旧分数差直接解释为单一改动的因果效果。

需要 CUDA、transformers、tokenizers、pandas、pyarrow。无需下载模型。先关闭其他训练内核；每次完整执行创建独立目录。可在 [配套源码](support/advanced14/experiment.py) 查看所有实现。

历史验证记录：参考环境、种子 14014、固定 200 步 SFT 后，80 道测试题的合法率与准确率从冷启动的 0% 提高到 100%。再追加真实 verl 的 10 步 GRPO，准确率仍为 100%，其中 1 步有非零梯度。完整记录见 [验证说明](support/docs/VALIDATION.md)；这是单次实测，不是预设的运行断言。

## 1 · 检查协议与划分

在训练前核对：提示没有 UNK；标准答案不出现在提示尾部；train/validation/test 数值不交叉。问题中的 yes/no 是候选协议，不是正确标签。三位数测试仍是同分布组合评估，不是通用数学能力。''')
code('''import gc, importlib.util, json, os, time
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
spec = importlib.util.spec_from_file_location('lesson14', Path('support/advanced14/experiment.py'))
lesson = importlib.util.module_from_spec(spec); spec.loader.exec_module(lesson)
assert torch.cuda.is_available(), '请选择支持 CUDA 的内核'
SEED = 14014
tokenizer, initial, splits = lesson.prepare(SEED)
initial = initial.cuda()
assert not(set(splits['train']) & set(splits['test']))
assert not(set(splits['validation']) & (set(splits['train']) | set(splits['test'])))
for numbers in splits.values(): lesson.batch(tokenizer, numbers, 'cuda')
print({k: len(v) for k, v in splits.items()})
example = splits['train'][0]
print('提示:', lesson.prompt(example))
print('token:', tokenizer.convert_ids_to_tokens(lesson.encode_prompt(tokenizer, example)))
print('参数量:', sum(p.numel() for p in initial.parameters()))
base = Path(os.environ.get('VERL_ADV14_BASE', str(Path.home()/'verl-advanced14/runs')))
run = base/f'{time.strftime("%Y%m%d-%H%M%S")}-{time.time_ns()}'
run.mkdir(parents=True)
print('本次目录:', run)''')
md('''## 2 · 固定预算预热，只监测验证集

从同一随机起点保留冻结基线，另一个副本做 200 步 SFT。每步 32 题、学习率 0.001，只计算答案与 EOS 的损失，每 20 步展示验证结果。固定最后一步用于评估，不按最终测试挑模型。

合法输出率衡量格式，准确率衡量任务，EOS 率衡量停止行为。三者要分开看。监督预热同时学习了格式和任务，不能据此归因于“只修复格式”。''')
code('''def live(history):
    clear_output(wait=True)
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(frame.step, frame.loss); axes[0].set(title='SFT loss', xlabel='Step')
    axes[1].plot(frame.step, frame.valid_rate, label='Valid')
    axes[1].plot(frame.step, frame.accuracy, label='Accuracy')
    axes[1].set(ylim=(-.05, 1.05), title='Validation only', xlabel='Step'); axes[1].legend()
    plt.tight_layout(); plt.show(); plt.close(fig)
    print(frame.tail(1).to_string(index=False))
# 使用带标准答案的样本先做 SFT。history 记录训练过程，返回的 model 随后导出，作为真实 GRPO 的起点。
model, history = lesson.warmup(tokenizer, initial, splits, steps=200, seed=SEED, callback=live)
manifest = lesson.export(run, tokenizer, model, initial, splits, history, seed=SEED)
print('模型、数据与配置已导出:', run)''')
md('''## 3 · 预先固定的测试与完整记录

现在才对测试集运行冻结基线和 SFT 模型。多数类别基线为 50%，由预先平衡的数据设计决定。固定猜 yes 也能得到 100% 合法率，因此合法率改善本身不足以证明任务学会。

如果 SFT 已达到很高准确率，后续 GRPO 可能只有很少混合奖励组。此时应研究有效组比例与资源成本，不应为了制造提升而事后挑一个较差起点。若要比较预热步数，另做验证集消融并预留新的最终测试。''')
code('''rows = []
for stage, obj in [('cold', initial), ('sft', model)]:
    rows += lesson.evaluate(obj, tokenizer, splits['test'], stage, 'test')
predictions = pd.DataFrame(rows)
summary = predictions.groupby('stage')[['valid','correct','eos']].mean()
display(summary)
display(predictions[predictions.stage=='sft'].head(10))
out = Path('support/experiment14_output'); out.mkdir(exist_ok=True)
predictions.to_csv(out/'predictions.csv', index=False)
summary.to_csv(out/'summary.csv')
pd.DataFrame(history).to_csv(out/'history.csv', index=False)
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
(out/'latest_run.txt').write_text(str(run))
print('多数类别基线: 0.5；完整预测已保存。')''')
md('''## 4 · 接回真实 verl：可选的完整训练

导出的 `model/`、Parquet 和奖励函数可以交给 verl。下格只打印命令并释放当前模型显存，训练需要在终端执行，避免 Notebook 持有的模型与 vLLM 竞争显存。脚本沿用第 12 课的单卡配置，固定 10 步，KL 关闭；本课不会自动启动训练。

这里的对照是 SFT 冻结模型与 SFT 后再 GRPO，尚未包含等预算继续 SFT，因此只能回答“追加 GRPO 后发生了什么”。吞吐、梯度及结果都以本轮日志为准。训练成功后才写 `latest_successful_grpo.txt`，失败目录保留用于诊断。''')
code('''import shlex
del model, initial
gc.collect(); torch.cuda.empty_cache()
print('在已安装 verl 的项目根目录执行：')
print('VERL_ADV14_ROOT=' + shlex.quote(str(run)) + ' bash tutorial/support/advanced14/run_grpo.sh')
print('完成后在下一格启用读取；未运行时不生成 GRPO 成绩。')''')
md('''## 5 · 读取真实检查点，或明确标记尚未运行

仅加载自己生成的可信检查点。FSDP 文件可能包含 pickle 对象。默认跳过读取，避免误把旧目录当作新训练。启用后对同一测试题作配对比较，第 15 课介绍如何量化不确定性。''')
code('''# 打开开关前先运行上一节打印的训练命令；这个分支负责读取真实 GRPO 产物并评估。
READ_GRPO = False
if READ_GRPO:
    completed = Path((run/'latest_successful_grpo.txt').read_text().strip())
    display(lesson.evaluate_grpo(completed, out))
else:
    print('GRPO 尚未在本 Notebook 中评估；当前结果仅包含冷启动与 SFT。')''')
md('''## 6 · 工程验收与下一次实验

提交运行 manifest、全部测试预测、学习曲线，以及一个能被证据否定的结论。例如：“固定 200 步 SFT 是否提高了协议合法率？”

进一步研究请预先注册三个分支：冻结 SFT、继续 SFT、SFT 后 GRPO。记录每个分支的 GPU 时间、生成 token 数、监督标签使用量与至少三个随机种子，不将更新次数相同称为算力相同。

下一项消融应把“补齐词表”“仅格式预热”“任务预热”分别控制，使用新的评估划分。每次只改变一个因素，保留失败运行。课后参考：[verl GRPO 示例](https://verl.readthedocs.io/en/latest/examples/grpo_example.html)。''')
save('14_用监督微调预热GRPO.ipynb')

md('''# 实验 15 · 估计评估结果的不确定性

博士训练中的统计判断要延伸到模型评估：同一测试集上的模型差异是配对数据；同一模型多次采样不是多个独立训练种子；反复调参会消耗测试集的独立性。

本课用第 14 课实际保存的逐题预测计算配对 bootstrap 区间，并用第 12 课“0/20 正确”说明零观察值的含义。需要先运行第 14 课的 SFT 评估。使用 CPU 即可。

## 1 · 先检查数据单位与评估协议

这里的单位是数值相同的一道题。同题的前导零、交换顺序、多个采样回答不能直接当作独立题目。复杂数据应按用户、文档或题目族分组重采样。

本实验 test 是固定的小型三位数集合。下面的区间仅反映在近似独立、代表性抽样假设下的题目抽样不确定性；不包含训练种子、设备差异或真实业务分布变化。''')
code('''import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
path = Path('support/experiment14_output/predictions.csv')
assert path.exists(), '请先完成第 14 课第 3 节'
df = pd.read_csv(path, keep_default_na=False)
df = df[df.split=='test'].copy()
assert not df.duplicated(['stage','number']).any()
def bool_values(series):
    result = series.astype(str).str.lower().map({'true':1, 'false':0, '1':1, '0':0})
    assert result.notna().all(), '未知布尔值'
    return result.astype(int)
df['score'] = bool_values(df.correct)
# 把同一题的冷启动和 SFT 结果对齐在一行。相减得到逐题变化，后续抽样才能保留配对关系。
paired = df.pivot(index='number', columns='stage', values='score')
assert {'cold','sft'} <= set(paired.columns) and not paired[['cold','sft']].isna().any().any()
display(paired.head())
print('测试题数:', len(paired))''')
md(r'''## 2 · 配对差异与 bootstrap

先对每题计算 $d_i=score_{SFT,i}-score_{cold,i}$，再重采样题目索引。两个模型必须使用同一组重采样索引；分别采样会丢失题目难度带来的相关性。

本格使用 10,000 次重采样和固定种子，报告百分位区间。小样本或所有差值都相同时，bootstrap 可能退化为很窄甚至零宽区间，这不意味着未来样本没有不确定性。不要用一个区间替代实验设计。''')
code('''delta = (paired.sft-paired.cold).to_numpy(dtype=float)
rng = np.random.default_rng(15015)
indices = rng.integers(0, len(delta), size=(10000,len(delta)))
boot = delta[indices].mean(axis=1)
# 从重采样得到的准确率差中取两个分位点。这里估计的是题目抽样的不确定性；训练种子之间的波动需要多次训练来估计。
lo, hi = np.quantile(boot, [.025,.975])
improved = int((delta>0).sum()); worsened = int((delta<0).sum())
print(f'准确率差: {delta.mean():.1%}; 配对 bootstrap 95% 区间: [{lo:.1%}, {hi:.1%}]')
print('改善题数:', improved, '退化题数:', worsened, '不变:', int((delta==0).sum()))
# 明确的边界检查：相同模型在每道题上差值均为零。
same = (paired.sft-paired.sft).to_numpy()
assert same[indices].mean() == 0''')
md(r'''## 3 · 0/20 正确并不等于总体成功概率严格为零

假定题目是独立同分布的 Bernoulli 试验，观察 n 次全失败后，成功概率的单侧 95% 精确上界为 $1-0.05^{1/n}$。对第 12 课 n=20 的结果，这个界说明小样本的分辨率有限，而不是替失败结果开脱。

Wilson 区间可用于单个模型的二项比例描述，不能把两个模型的 Wilson 区间是否重叠当成配对差异检验。''')
code('''def wilson(successes, total, z=1.959963984540054):
    assert 0 <= successes <= total and total > 0
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return max(0, center-radius), min(1, center+radius)
upper = 1-.05**(1/20)
print(f'0/20 的单侧 95% 精确上界: {upper:.2%}')
for stage in ['cold','sft']:
    successes = int(paired[stage].sum())
    print(stage, successes, '/', len(paired), 'Wilson:', wilson(successes,len(paired)))
assert wilson(0,20)[0] < 1e-12
assert wilson(20,20)[1] > 1-1e-12
out=Path('support/experiment15_output'); out.mkdir(exist_ok=True)
report=dict(n=len(delta), delta=float(delta.mean()), paired_bootstrap_95=[float(lo),float(hi)],
    improved=improved, worsened=worsened, zero_of_20_one_sided_95_upper=upper,
    scope='Item uncertainty only; one training seed; approximate iid item assumption.')
(out/'evaluation_audit.json').write_text(json.dumps(report,indent=2))''')
md('''## 4 · 从单次演示到可发表、可上线的证据

在运行前写好实验卡：问题、主指标、数据分组、基线、预算、停止条件和失败判据。验证集用于选方案，最终测试只回答预先指定的问题。公开教程里的测试集一旦被反复查看，就应视为开发基准。

| 问题 | 应记录什么 | 常见误判 |
|---|---|---|
| 训练是否稳定 | 每个训练种子的结果、均值与离散程度 | 把多个采样回答当成多个种子 |
| 哪些用户受损 | 按题型/难度/长度分层的指标与样本数 | 只报总体均值 |
| RL 是否值得成本 | 同起点冻结、继续 SFT、RL，GPU 时间与 token 预算 | 只比较更新步数 |
| 线上是否改善 | 预先定义的业务指标、延迟、成本与回滚阈值 | 离线分数直接等于用户收益 |

毕业作业：至少三个训练种子，对同一最终题集保存预测；分别报告种子变异与题目不确定性。若要合并估计，说明层级重采样结构，不把所有 seed×题目行当成独立观测。保留每次运行，而不是挑最好的种子。''')
save('15_估计评估结果的不确定性.ipynb')

md(r'''# 实验 16 · 测量显存与训练吞吐

“GPU 可用”与“GPU 被有效利用”是两件事。本课练习建立资源模型、正确计时，并识别微基准的适用范围。需要 CUDA；无需前课权重。8GB 卡从小尺寸开始，先关闭其他训练内核。

## 1 · 先做显存预算，再启动训练

普通混合精度 AdamW 的一种常见估算是：BF16 参数 2P 字节、梯度 2P、FP32 master 参数 4P、两份 FP32 优化器状态 8P，合计约 16P。具体框架可能使用 FP32 梯度、不同 optimizer state dtype 或没有独立 master copy，必须检查实际实现。

这还不含激活、临时工作区、通信缓冲、碎片、rollout KV cache 和额外模型副本。KV cache 的简化估算为

$$2\times L\times N_{seq}\times T\times H_{kv}\times d_{head}\times bytes.$$

其中 2 表示 K 与 V。GQA 使用 KV head 数，不是 query head 数。vLLM 的分页、块对齐与并行切分会进一步影响实际占用。''')
code('''import gc, json, statistics, time
from pathlib import Path
import pandas as pd
import torch
from IPython.display import display
assert torch.cuda.is_available(), '本课需要 CUDA'
GiB = 1024**3
def budget(parameters, layers, sequences, tokens, kv_heads, head_dim, bytes_per_element=2):
    return dict(parameters=parameters, adamw_example_gib=16*parameters/GiB,
        kv_cache_example_gib=2*layers*sequences*tokens*kv_heads*head_dim*bytes_per_element/GiB)
display(pd.DataFrame([budget(p,24,8,2048,8,64) for p in [1250000,500000000,7000000000]]))
print('上表是模型示例预算，未包含激活与进程开销，不是可运行配置保证。')
print(torch.cuda.get_device_name(0))''')
md('''## 2 · CUDA 计时与批量大小

CPU 提交 CUDA 操作后可能立即返回。用 `torch.cuda.Event` 记录设备时间，并在读取前同步；另记同步后的墙钟时间。先预热，再重复测量，报告中位数。

本格是固定权重 MLP 的推理微基准，用 batch=1/32/128 比较样本吞吐与批次延迟。它不包含 tokenizer、数据加载、网络和自回归解码，不能用来宣称真实服务 token/s。关闭 TF32 以明确 FP32 计算设置，结束时恢复原设置。''')
code('''torch.manual_seed(16016)
original_tf32 = torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32 = False
net = torch.nn.Sequential(torch.nn.Linear(512,2048),torch.nn.GELU(),torch.nn.Linear(2048,512)).cuda().eval()
results=[]
try:
    with torch.inference_mode():
        for batch_size in [1,32,128]:
            x=torch.randn(batch_size,512,device='cuda')
            for _ in range(10): net(x)
            # CUDA 调用可以提前把控制权还给 Python；同步会等已提交的 GPU 工作完成，再读时钟才有意义。
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples=[]; walls=[]
            for _ in range(7):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter(); start.record()
                for _ in range(50): net(x)
                # 结束事件进入 GPU 队列，等待它完成再读取时间。对照 GPU 事件计时和主机墙钟时间，留意两者覆盖的开销。
                end.record(); end.synchronize()
                walls.append((time.perf_counter()-t)*1000/50)
                samples.append(start.elapsed_time(end)/50)
            latency=statistics.median(samples)
            results.append(dict(batch=batch_size,device_batch_ms=latency,
                wall_batch_ms=statistics.median(walls),samples_per_second=batch_size/(latency/1000),
                process_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                process_reserved_mib=torch.cuda.memory_reserved()/1024**2))
finally:
    torch.backends.cuda.matmul.allow_tf32=original_tf32
display(pd.DataFrame(results))''')
md('''## 3 · 混合精度的收益要与误差一起测

BF16 的指数范围接近 FP32，但尾数精度较低。FP16 的范围更窄，训练时常需要 loss scaling；具体设置取决于模型与框架。AMP autocast 只选择运算 dtype，不自动把所有模型状态都变小。

本格只比较一次 BF16 autocast 推理的数值误差，不据此断言端到端训练精度或速度。训练验证还应检查 loss、梯度范数、NaN/Inf 与固定验证集。''')
code('''if torch.cuda.is_bf16_supported():
    with torch.inference_mode():
        fp32=net(x)
        with torch.autocast('cuda',dtype=torch.bfloat16): mixed=net(x)
    relative_l2=float((mixed.float()-fp32).norm()/fp32.norm().clamp_min(1e-12))
    assert torch.isfinite(mixed).all()
    print('BF16 相对 L2 误差:', relative_l2)
else:
    relative_l2=None
    print('该 GPU 不支持 BF16，跳过本格比较。')
out=Path('support/experiment16_output');out.mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(out/'microbenchmark.csv',index=False)
(out/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),bf16_relative_l2=relative_l2,
    scope='MLP inference microbenchmark; not serving latency or training throughput.'),indent=2))
del net,x
if relative_l2 is not None: del fp32,mixed
gc.collect();torch.cuda.empty_cache()''')
md(r'''## 4 · 从日志定位问题，再选优化手段

| 现象 | 先确认的证据 | 可能的下一步 |
|---|---|---|
| CUDA OOM | 哪个进程、阶段与张量分配失败，allocated/reserved | 减少序列或 micro batch，再评估 checkpointing/offload |
| Ray worker 被杀 | 主机 RAM、系统日志、worker 数 | 降低并发 worker，减少重复加载 |
| GPU 利用率低 | 数据等待、CPU tokenizer、同步点与小 kernel | 用 profiler 区分 CPU/CUDA 时间，再改批量或流水线 |
| rollout 耗时占比高 | 输入/输出长度、每题采样数、prefill/decode 时间 | 调整采样预算或生成批次，同时复核评估协议 |
| 更新不稳定 | 梯度、裁剪比例、KL、奖励分布、有效组比例 | 先查 mask/log-prob/数据，再改学习率 |

进程的 `max_memory_allocated` 不是整张卡的总占用；`empty_cache()` 也不能释放仍被变量引用的张量。端到端测量要记录模型加载与 warm-up，并说明是否包含它们。

## 5 · LoRA、分布式与服务：下一阶段如何验收

LoRA 学习低秩增量 $\Delta W=BA$，减少可训练参数、梯度及优化器状态，但基础权重、激活和 KV cache 仍占资源。量化 LoRA 还引入量化误差与算子兼容问题。不要把“可训练参数少”直接换算成“整个作业显存同倍下降”。

数据并行复制模型处理不同样本；FSDP/ZeRO 切分参数或训练状态；张量并行切分层内计算；流水线并行切分层。它们交换的是不同张量，需要评估通信开销、负载均衡和失败恢复。单卡 NO_SHARD 不能证明多卡配置正确。

服务侧分别测首 token 延迟（TTFT）、后续 token 间隔（TPOT）、端到端延迟、并发下 p50/p95 与吞吐，并固定输入/输出长度和硬件。单请求快不意味着高并发稳定。

课后交付：为第 14 课画出进程与显存归属图，记录一次真实训练的时间分解。选择一个瓶颈，只改一个参数，比较质量、耗时和峰值内存。完整的 LoRA、多卡和服务实验尚未包含在本课程的实测范围内，下一阶段验收要求见 [工程学习路线](support/docs/ENGINEERING_ROADMAP.md)。

参考：[PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[AMP](https://pytorch.org/docs/stable/amp.html)、[LoRA](https://arxiv.org/abs/2106.09685)、[verl 性能调优](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)。''')
save('16_测量显存与训练吞吐.ipynb')
