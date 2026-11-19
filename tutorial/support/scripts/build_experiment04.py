"""Build an auditable undertrained-baseline comparison, reusing verified lesson code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
n2 = json.loads((ROOT/'02_tiny_language_model.ipynb').read_text(encoding='utf-8'))
n3 = json.loads((ROOT/'03_group_reward_learning.ipynb').read_text(encoding='utf-8'))
def sources(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
c2,c3=sources(n2),sources(n3)
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))

md('''# 实验 04 · 留出学习空间，看清训练带来的变化

上一课的初始模型在参考运行中已有 97.5% 的训练准确率，进一步改善的空间较小。为了观察学习过程，本课重新创建相同结构的微型 Qwen2，只做 200 步监督预热，再复制成三个参数完全相同的模型。

1. **冻结基线**：不再训练。
2. **继续 SFT**：直接使用标准答案，训练 1200 次参数更新。
3. **奖励微调 RL**：每题采样 8 个答案，训练 600 轮，每轮 2 次参数更新。

本课比较两件事：继续训练能否超过冻结基线，以及奖励微调与继续 SFT 各自表现如何。结果只适用于这个重新预热的起点，RL 是否更好要看实际评估。 数据划分沿用第二课；三个模型面对完全相同的 80 道测试题。固定训练步数，不选最优测试检查点。

这是单随机种子的教学演示，不是统计显著性结论。两种训练有相同的更新次数和每次 128 条回答序列，但采样成本、题目多样性、监督信息量不同，不能称为等算力比较。

选择 Python (verl GPU)，从头 Shift+Enter。无需模型下载，结果写入独立的 `support/experiment04_output/`，不覆盖前三课。''')
md('''## 1 · 固定数据和随机种子
沿用第二课 320 道训练题、80 道测试题；交换加数的题目仍在同一分组。''')
code(c2[0]+'\nimport copy\n'+c2[1])
md('''## 2 · 只预热 200 步，建立相同起点
这里只用训练题。初始化后先记下未训练模型表现；预热步数预先固定，不按测试准确率挑起点。''')
code(c2[2]+'''
WARMUP = 200
warm_optimizer = torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01)
for step in range(WARMUP):
    model.train()
    idx=torch.randint(len(train_pairs),(64,),device='cuda')
    loss=model(input_ids=train_ids[idx],attention_mask=train_mask[idx],labels=train_labels[idx]).loss
    warm_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    warm_optimizer.step()
model.eval()
actor=copy.deepcopy(model).eval()
reference=copy.deepcopy(model).eval().requires_grad_(False)
sft_model=copy.deepcopy(model).eval()
initial_state={k:v.detach().cpu().clone() for k,v in reference.state_dict().items()}
assert all(torch.equal(actor.state_dict()[k],sft_model.state_dict()[k]) for k in initial_state)
print('预热结束，三个分支参数完全相同。')
del warm_optimizer
''')
md('''## 3 · 建立奖励与基线
与第三课相同：只给采样得到的完整正确答案奖励 1，错误为 0。没有答案前缀泄漏，也不在 rollout 中提供标准答案。''')
code(c3[1]+'\n'+c3[2])
md('''## 4 · 奖励微调：实时看模型学习

采用第三课的组内优势、策略裁剪和 KL 约束。预先固定学习率 3e-5、600 轮、组大小 8；每 60 轮显示一次曲线。测试准确率只观察，不影响更新或停止时点。

关注训练与测试两条线，以及有效组比例。如果一组回答全对或全错，就没有相对奖励信号。''')
rl=c3[3].replace('200, 8, 16, 2','600, 8, 16, 2').replace('2e-5, 0.02, 0.2','3e-5, 0.02, 0.2').replace('step%20==0','step%60==0')
code(rl+'''
rl_seconds=elapsed
rl_history=history
rl_test,rl_answers=evaluate(actor,test_pairs)
rl_train,_=evaluate(actor,train_pairs)
''')
md('''## 5 · 继续 SFT：比较另一种训练方式

从同一个预热起点训练另一个副本，直接用标准答案计算交叉熵。固定学习率 0.001、1200 次更新、批量 128。SFT 有更直接的 token 级监督；这个任务的标准答案可以计算，因此它是必要且强的对照。

用时分别记录。此处的实时曲线可以帮助你看到 SFT 与 RL 的不同学习过程，但不能凭一次运行推断普遍优劣。''')
code('''SFT_STEPS=1200
sft_optimizer=torch.optim.AdamW(sft_model.parameters(),lr=0.001,weight_decay=0.01)
sft_history=[dict(step=0,train=baseline_train,test=baseline_test)]
sft_live=display('',display_id=True)
sft_start=time.perf_counter()
for step in range(1,SFT_STEPS+1):
    sft_model.train()
    idx=torch.randint(len(train_pairs),(128,),device='cuda')
    loss=sft_model(input_ids=train_ids[idx],attention_mask=train_mask[idx],labels=train_labels[idx]).loss
    assert torch.isfinite(loss)
    sft_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(),1.0)
    sft_optimizer.step()
    if step%120==0:
        sft_model.eval()
        tr,_=evaluate(sft_model,train_pairs)
        te,_=evaluate(sft_model,test_pairs)
        sft_history.append(dict(step=step,train=tr,test=te))
        fig,ax=plt.subplots(figsize=(7,3))
        for key in ['train','test']:
            ax.plot([r['step'] for r in sft_history],[r[key] for r in sft_history],label=key)
        ax.axhline(baseline_test,ls='--',color='gray',label='Frozen test baseline')
        ax.set(xlabel='SFT updates',ylabel='Exact-match accuracy',ylim=(0,1.05))
        ax.legend();fig.tight_layout();sft_live.update(fig);plt.close(fig)
        print(f'SFT {step:4d} | train {tr:.2%} | test {te:.2%}',flush=True)
torch.cuda.synchronize()
sft_seconds=time.perf_counter()-sft_start
sft_model.eval()
sft_test,sft_answers=evaluate(sft_model,test_pairs)
sft_train,_=evaluate(sft_model,train_pairs)
''')
md('''## 6 · 三组同题比较：看百分点、题数和错误转移

柱状图纵轴从 0 开始，不截断坐标放大差异。每道测试题占 1.25 个百分点。展示完整测试答案表，以及改善/退化题数；不只选择成功样本。

测试结果是域内、未在训练中出现的数对，不代表更大数字或通用推理能力。''')
code('''names=['Frozen baseline','Continue SFT','Reward RL']
scores=[baseline_test,sft_test,rl_test]
fig,ax=plt.subplots(figsize=(8,4))
bars=ax.bar(names,scores,color=['#94a3b8','#2563eb','#059669'])
ax.set(ylim=(0,1.12),ylabel='Held-out exact-match accuracy',title='Same 80 test problems / same initial checkpoint')
for bar,value in zip(bars,scores):
    ax.text(bar.get_x()+bar.get_width()/2,value+0.02,f'{value:.2%}',ha='center')
plt.tight_layout();plt.show()
for label,score,answers,seconds in [('SFT',sft_test,sft_answers,sft_seconds),('RL',rl_test,rl_answers,rl_seconds)]:
    improved=sum(b!=str(a+c) and f==str(a+c) for (a,c),b,f in zip(test_pairs,baseline_answers,answers))
    regressed=sum(b==str(a+c) and f!=str(a+c) for (a,c),b,f in zip(test_pairs,baseline_answers,answers))
    print(f'{label}: {score:.2%}，变化 {(score-baseline_test)*100:+.2f} 个百分点；改善 {improved} 题 / 退化 {regressed} 题；{seconds:.1f} 秒')
import pandas as pd
table=pd.DataFrame([dict(question=f'{a}+{b}=',target=str(a+b),baseline=b0,sft=s,rl=r)
    for (a,b),b0,s,r in zip(test_pairs,baseline_answers,sft_answers,rl_answers)])
display(table.style.apply(lambda row:['background-color: #dcfce7' if col in ('baseline','sft','rl') and row[col]==row['target'] else '' for col in table.columns],axis=1))
assert all(torch.equal(initial_state[k],v.detach().cpu()) for k,v in reference.state_dict().items())
assert any(not torch.equal(initial_state[k],v.detach().cpu()) for k,v in actor.state_dict().items())
print('验证通过：冻结对照没有变化，RL 模型已更新。')
''')
md('''## 7 · 保存证据
保存三组权重、固定划分、所有测试答案和曲线记录。重复保存会覆盖本实验结果；与前三课输出分开。''')
code('''out=Path('support/experiment04_output')
out.mkdir(exist_ok=True)
for name,m in [('baseline',reference),('sft',sft_model),('rl',actor)]:
    m.save_pretrained(out/name)
table.to_csv(out/'all_test_answers.csv',index=False)
record=dict(seed=SEED,warmup=WARMUP,train_pairs=train_pairs,test_pairs=test_pairs,vocab=vocab,
    baseline_train=baseline_train,baseline_test=baseline_test,sft_train=sft_train,sft_test=sft_test,rl_train=rl_train,rl_test=rl_test,
    rl_seconds=rl_seconds,sft_seconds=sft_seconds,rl_history=rl_history,sft_history=sft_history,
    config=dict(rl_steps=STEPS,rl_updates=UPDATES,rl_lr=LR,group=GROUP,prompts=PROMPTS,beta=BETA,clip=CLIP,
                sft_steps=SFT_STEPS,sft_lr=0.001,sft_batch=128,warmup_lr=0.002,warmup_batch=64),
    torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0))
(out/'results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('已保存:',out.resolve())
''')
md('''## 怎样解读“明显”

- 如果 RL 比冻结基线明显进步，说明这个起点上奖励更新产生了可见效果。
- 如果 SFT 更好，应报告 SFT 更好，不能宣称 RL 最优。
- 这个起点刻意只预热 200 步，是展示学习的设计条件，不是上一课已充分训练模型上的提升。
- 本实验仍然是单种子演示。要做研究结论，需要多个随机种子、验证集和未反复查看的新最终测试集。

本教程是教学版 GRPO 风格循环，未运行完整 verl 分布式训练器。''')
for i,c in enumerate(cells):c['id']=f'exp04-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=n3['metadata'])
(ROOT/'04_visible_learning_comparison.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
