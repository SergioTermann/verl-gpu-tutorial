"""Create the executable group-relative policy optimization lesson."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s))


def code(s):
    cells.append(dict(cell_type='code', metadata={}, source=s, outputs=[], execution_count=None))


md('''# 实验 03 · 用奖励微调小语言模型

第二课使用标准答案做监督训练。本课加载它保存的微型 Qwen2，让模型自己采样答案，按答对与否打分，再用组内相对优势更新策略。

**这是 PyTorch 编写的教学版 GRPO 风格训练循环，不是 verl 训练器的端到端运行。** 本课包含组采样、标准化优势、PPO 比率裁剪和冻结参考策略 KL 约束；为便于阅读，省略分布式调度、价值模型和生产训练中的各种修正。下一步可以把这些概念映射到 verl。

需要先完成实验 02 的保存格。选择 **Python (verl GPU)** 内核，从上到下 Shift+Enter。无需下载模型，固定训练 200 次迭代，不根据测试结果挑选最优检查点。预期几十秒至数分钟，以本机实测为准。

训练奖励上升不保证测试准确率上升；这里保留所有实际曲线和错误。不要同时运行其他实验的 GPU 训练。

## 1 · 加载第二课权重，建立冻结对照

actor 是正在学习的策略；reference 是第二课模型的冻结副本。两者初始参数相同，reference 全程不接受梯度。延用第二课的数据划分，测试题不进入奖励更新。''')
code('''import json, time, random, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display
from transformers import Qwen2ForCausalLM

assert torch.cuda.is_available(), '请选择 Python (verl GPU) 内核'
root = Path('experiment02_output')
assert (root/'model/model.safetensors').exists(), '请先运行实验 02，包括最后的保存格'
source = json.loads((root/'results.json').read_text(encoding='utf-8'))
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
vocab = source['vocab']
stoi = {c:i for i,c in enumerate(vocab)}
EOS = stoi['#']
train_pairs = [tuple(p) for p in source['train_pairs']]
test_pairs = [tuple(p) for p in source['test_pairs']]
assert not {tuple(sorted(p)) for p in train_pairs} & {tuple(sorted(p)) for p in test_pairs}
actor = Qwen2ForCausalLM.from_pretrained(root/'model').cuda().eval()
reference = Qwen2ForCausalLM.from_pretrained(root/'model').cuda().eval()
reference.requires_grad_(False)
actor.config.use_cache = reference.config.use_cache = False
initial_state = {k:v.detach().cpu().clone() for k,v in reference.state_dict().items()}
source_hash = hashlib.sha256((root/'model/model.safetensors').read_bytes()).hexdigest()
print('GPU:', torch.cuda.get_device_name(0))
print('训练 / 测试:', len(train_pairs), '/', len(test_pairs))
print('已加载相同初始模型；reference 已冻结。')''')
md('''## 2 · 采样回答与规则奖励

每道题采样 8 个答案。正确答案字符串奖励为 1，错误为 0。最多生成 3 个 token（两位数和结束符），结束符后的 token 不参与损失。

为了避免填充影响，这里将相同提示长度的题目放入同一批。温度固定为 1，采样和更新使用同一概率分布。训练采用随机采样，评估采用贪心解码，因此“采样平均奖励”和“贪心准确率”是不同指标。''')
code('''def encode(text):
    return [stoi[c] for c in text]

def decode(tokens):
    return ''.join(vocab[t] for t in tokens).split('#')[0]

@torch.no_grad()
def rollout(model, pairs, group=1, sample=True):
    prompts = [encode(f'{a}+{b}=') for a,b in pairs]
    assert len({len(p) for p in prompts}) == 1
    x = torch.tensor(prompts, device='cuda').repeat_interleave(group, dim=0)
    prompt_len = x.shape[1]
    active = torch.ones(x.shape[0], dtype=torch.bool, device='cuda')
    masks, old_logps = [], []
    for _ in range(3):
        logits = model(input_ids=x).logits[:, -1]
        distribution = torch.distributions.Categorical(logits=logits)
        nxt = distribution.sample() if sample else logits.argmax(-1)
        masks.append(active.clone())
        old_logps.append(distribution.log_prob(nxt))
        active = active & (nxt != EOS)
        x = torch.cat([x,nxt[:,None]],dim=1)
    answers = [decode(row) for row in x[:,prompt_len:].tolist()]
    targets = [str(a+b) for a,b in pairs for _ in range(group)]
    rewards = torch.tensor([float(a==b) for a,b in zip(answers,targets)],device='cuda')
    return x, torch.stack(masks,1), torch.stack(old_logps,1), rewards, answers, prompt_len

by_length = {}
for p in train_pairs:
    by_length.setdefault(len(f'{p[0]}+{p[1]}='), []).append(p)

def evaluate(model, pairs):
    buckets = {}
    answers = ['']*len(pairs)
    for i,p in enumerate(pairs):
        buckets.setdefault(len(f'{p[0]}+{p[1]}='), []).append((i,p))
    for rows in buckets.values():
        result = rollout(model,[p for _,p in rows],sample=False)[4]
        for (i,_),answer in zip(rows,result):
            answers[i] = answer
    acc = sum(s==str(a+b) for s,(a,b) in zip(answers,pairs))/len(pairs)
    return acc, answers

demo_pairs = by_length[min(by_length)][:2]
demo = rollout(actor,demo_pairs,group=8)
for i,p in enumerate(demo_pairs):
    print(f'{p[0]}+{p[1]}=：', list(zip(demo[4][i*8:(i+1)*8],demo[3][i*8:(i+1)*8].tolist())))
baseline_train,_ = evaluate(reference,train_pairs)
baseline_test,baseline_answers = evaluate(reference,test_pairs)
print(f'冻结基线：训练 {baseline_train:.2%} / 测试 {baseline_test:.2%}')''')
md('''## 3 · 组内优势与策略更新

对同一题的 G 个奖励，计算 `A = (r - mean(r)) / (std(r) + 1e-6)`。组内有好有坏时，高于均值的回答有正优势；如果全对或全错，优势为零，奖励项不给出更新信号。

下面会显示“有效组比例”（组内奖励有差异的比例）。已经训练得很好的模型容易采样到全对组，这也是 RL 信号稀疏的一个原因。

策略损失使用 token 概率比率 `exp(new_logp - old_logp)`，并在 `[0.8,1.2]` 裁剪。每次采样做 2 次更新，因此第二次更新可以体现概率比率变化。KL 项使用采样估计 `exp(ref_logp-new_logp) - (ref_logp-new_logp) - 1`，抑制远离参考模型；有限样本上的估计不是精确全词表 KL。

先对优势和 EOS 掩码做小检查，避免用错误实现解释训练结果。''')
code('''def group_advantages(rewards):
    return (rewards-rewards.mean(-1,keepdim=True))/(rewards.std(-1,keepdim=True,unbiased=False)+1e-6)

toy = torch.tensor([[1.,0.,1.,0.],[1.,1.,1.,1.]])
toy_adv = group_advantages(toy)
assert torch.allclose(toy_adv.mean(-1), torch.zeros(2), atol=1e-6)
assert torch.all(toy_adv[1]==0)
assert toy_adv[0,0]>0 and toy_adv[0,1]<0
print('优势示例:',toy_adv.tolist())
assert demo[1][:,0].all()
generated = demo[0][:,demo[5]:]
for i in range(1,3):
    assert torch.equal(demo[1][:,i], (generated[:,:i]!=EOS).all(1))
print('组内优势与结束符掩码检查通过。')

def response_logps(model,x,prompt_len):
    logits = model(input_ids=x).logits[:,prompt_len-1:-1]
    return F.log_softmax(logits.float(),dim=-1).gather(-1,x[:,prompt_len:,None]).squeeze(-1)

check = response_logps(actor,demo[0],demo[5]).detach()
torch.testing.assert_close(check,demo[2],atol=2e-5,rtol=2e-5)
print('采样与训练前向的 token 对数概率一致。')''')
md('''## 4 · GPU 上进行奖励微调

学习率固定为 2e-5，KL 系数 0.02，16 个问题/批，每题 8 次采样。每 20 步展示曲线和独立测试集结果。选择提示长度时按各桶题数加权，再均匀选题，让每道训练题有相同被抽中的机会。

评估只观察，不用于回传梯度，也不决定何时停止。`actor.eval()` 关闭 dropout，但并不关闭梯度；这样 rollout 与更新使用一致的策略。''')
code('''STEPS, GROUP, PROMPTS, UPDATES = 200, 8, 16, 2
LR, BETA, CLIP = 2e-5, 0.02, 0.2
optimizer = torch.optim.AdamW(actor.parameters(),lr=LR,weight_decay=0.0)
history = [dict(step=0,train=baseline_train,test=baseline_test,reward=None,active=None,kl=None)]
live = display('',display_id=True)
window = []
start = time.perf_counter()
for step in range(1,STEPS+1):
    length = random.choices(list(by_length),weights=[len(v) for v in by_length.values()])[0]
    pairs = random.choices(by_length[length],k=PROMPTS)
    x,mask,old_logp,rewards,answers,prompt_len = rollout(actor,pairs,group=GROUP)
    grouped = rewards.reshape(PROMPTS,GROUP)
    adv = group_advantages(grouped).reshape(-1,1)
    active_fraction = (grouped.std(-1,unbiased=False)>0).float().mean().item()
    with torch.no_grad():
        ref_logp = response_logps(reference,x,prompt_len)
    for _ in range(UPDATES):
        logp = response_logps(actor,x,prompt_len)
        ratio = (logp-old_logp).exp()
        policy = -torch.minimum(ratio*adv,ratio.clamp(1-CLIP,1+CLIP)*adv)
        delta = ref_logp-logp
        kl = delta.exp()-delta-1
        loss = (((policy+BETA*kl)*mask).sum(1)/mask.sum(1)).mean()
        assert torch.isfinite(loss), '损失出现非有限值'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0)
        optimizer.step()
    window.append((rewards.mean().item(),active_fraction,((kl.detach()*mask).sum()/mask.sum()).item()))
    if step%20==0:
        train_acc,_ = evaluate(actor,train_pairs)
        test_acc,_ = evaluate(actor,test_pairs)
        row = dict(step=step,train=train_acc,test=test_acc,
                   reward=sum(v[0] for v in window)/len(window),
                   active=sum(v[1] for v in window)/len(window),
                   kl=sum(v[2] for v in window)/len(window))
        history.append(row)
        window=[]
        fig,axes=plt.subplots(1,3,figsize=(12,3.2))
        steps=[r['step'] for r in history]
        axes[0].plot(steps,[r['train'] for r in history],label='Train')
        axes[0].plot(steps,[r['test'] for r in history],label='Test')
        axes[0].axhline(baseline_test,color='gray',ls='--',label='Frozen test baseline')
        axes[0].set(title='Greedy exact-match accuracy',ylim=(0,1.05))
        axes[0].legend(fontsize=8)
        axes[1].plot(steps[1:],[r['reward'] for r in history[1:]],label='Sample reward')
        axes[1].plot(steps[1:],[r['active'] for r in history[1:]],label='Active groups')
        axes[1].set(title='Sampling signal',ylim=(0,1.05))
        axes[1].legend(fontsize=8)
        axes[2].plot(steps[1:],[r['kl'] for r in history[1:]])
        axes[2].set(title='Sampled KL estimate')
        for ax in axes: ax.set_xlabel('Iteration')
        fig.tight_layout()
        live.update(fig)
        plt.close(fig)
        print(f"step {step:3d} | reward {row['reward']:.3f} | active {row['active']:.1%} | test {test_acc:.2%} | KL {row['kl']:.4f}",flush=True)
torch.cuda.synchronize()
elapsed=time.perf_counter()-start
print(f'训练完成，用时 {elapsed:.1f} 秒。')''')
md('''## 5 · 与冻结模型比较：哪些题改善，哪些退化？

两种模型使用相同测试题、相同贪心解码，不挑选好看的样本。列出从错误变正确、从正确变错误的题目，以及最终准确率差（百分点）。一次运行不能证明稳定提升；泛化效果需要新测试集和多个随机种子验证。''')
code('''final_test,final_answers=evaluate(actor,test_pairs)
final_train,_=evaluate(actor,train_pairs)
improved,regressed=[],[]
for pair,before,after in zip(test_pairs,baseline_answers,final_answers):
    correct=str(sum(pair))
    if before!=correct and after==correct: improved.append((pair,before,after))
    if before==correct and after!=correct: regressed.append((pair,before,after))
print(f'冻结基线：{baseline_test:.2%}；RL 后：{final_test:.2%}；变化 {(final_test-baseline_test)*100:+.2f} 个百分点')
print(f'改善 {len(improved)} 题；退化 {len(regressed)} 题；测试集共 {len(test_pairs)} 题')
for name,rows in [('改善',improved),('退化',regressed)]:
    for (a,b),before,after in rows[:10]:
        print(f'{name}: {a}+{b}，{before!r} → {after!r}，正确答案 {a+b}')
assert all(torch.equal(initial_state[k],v.detach().cpu()) for k,v in reference.state_dict().items())
assert any(not torch.equal(initial_state[k],v.detach().cpu()) for k,v in actor.state_dict().items())
assert all(p.grad is None for p in reference.parameters())
print('检查通过：actor 确实更新，reference 参数保持不变且没有梯度。')''')
md('''## 6 · 保存结果与 verl 概念对照

保存到新的 `experiment03_output/`，不会覆盖第二课的初始模型。重复运行保存格会覆盖第三课上次输出。要重新做实验，应从第 1 格重跑；只重跑训练格会在已更新的 actor 上继续训练，不能当成相同起点的对照。

| 本课对象 | verl 中的对应概念 |
|---|---|
| `actor` | actor 策略模型 |
| `rollout(..., group=8)` | rollout 与 `rollout.n` |
| 精确匹配奖励 | 自定义 reward function |
| `group_advantages` | `algorithm.adv_estimator=grpo` 的组内比较思想 |
| 冻结 `reference` 与 KL | ref policy 与 actor KL loss |
| `PROMPTS / GROUP / UPDATES` | 问题批量、回答数量与策略更新次数 |

这些是概念映射，不保证教学代码与某个 verl 版本数值等价。这里的字符词表也还不是可直接交给 vLLM 的完整 tokenizer 集成。''')
code('''out=Path('experiment03_output')
out.mkdir(exist_ok=True)
actor.save_pretrained(out/'model')
record=dict(seed=SEED,source_model_sha256=source_hash,vocab=vocab,
    train_pairs=train_pairs,test_pairs=test_pairs,
    config=dict(steps=STEPS,group=GROUP,prompts=PROMPTS,updates=UPDATES,lr=LR,beta=BETA,clip=CLIP),
    history=history,baseline_answers=baseline_answers,final_answers=final_answers,
    baseline_test=baseline_test,final_test=final_test,final_train=final_train,
    elapsed_seconds=elapsed,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0))
(out/'results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('已保存:',out.resolve())''')
md('''## 7 · 练习：观察约束与奖励信号

先保存本次结果副本，然后每次从第 1 格重新开始，只改一个因素：

1. 将 `GROUP=8` 改为 `GROUP=4`，观察有效组比例及训练时间。
2. 将 `BETA=0.02` 改为 `0.0`，观察策略偏离和测试变化；奖励高不意味着泛化更好。
3. 思考为什么一组回答全错时无法用相对奖励判断哪个方向更好，以及 SFT 初始能力为什么重要。

不要反复按这 80 道测试题的成绩挑选配置后仍声称是独立最终评估。如果要比较方法，应预先定义验证集和新的最终测试集，并运行多个种子。

参考：[verl GRPO 示例](https://verl.readthedocs.io/en/latest/examples/grpo_example.html)、[DeepSeekMath](https://arxiv.org/abs/2402.03300)。''')
for i,c in enumerate(cells): c['id']=f'exp03-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
    kernelspec=dict(display_name='Python (verl GPU)',language='python',name='verl-gpu'),
    language_info=dict(name='python',version='3.12')))
Path(__file__).with_name('03_group_reward_learning.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
