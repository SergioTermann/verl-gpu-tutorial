"""Build the data-coverage and isolated-evaluation tutorial."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
previous=json.loads((ROOT/'05_model_playground.ipynb').read_text(encoding='utf-8'))
generate_source=[''.join(c['source']) for c in previous['cells'] if c['cell_type']=='code'][1]
generate_source=generate_source[:generate_source.index('for name,model in models.items():')]
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))

md('''# 实验 06 · 用数据覆盖修复模型的薄弱环节

上一课发现：第四课 SFT 模型在新数字范围和前导零格式上很不可靠。这里保留它作为冻结对照，复制一份继续监督训练，训练数据扩展为 **0–69 的加法 + 普通/前导零两种格式**。

**本课重点是数据设计和评估纪律，不是 RL 算法。** 以前的分布外题目在扩大覆盖后可能成为训练范围内问题，因此不能把新结果叫作同样条件下的“外推能力”。

我们另建从未被前面课程训练或评估过的最终测试数对，并隔离交换加数和格式变体。训练过程只查看验证集，固定跑完 2400 步；最终测试在训练结束后才运行一次。

选择 **Python (verl GPU)**，从头 Shift+Enter。无需下载模型，依赖第四、第五课的保存结果。预期训练几十秒至数分钟。所有新文件写入 `experiment06_output/`。

## 1 · 冻结旧模型，复制待改进模型

继续使用同样大小的 Qwen2，模型结构和字符词表不变。这里同时改变数字覆盖、格式覆盖和训练步数，结果不能归因于某一个因素；要分离各因素，需要额外消融实验。''')
code('''import json, random, time, copy, hashlib
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from transformers import Qwen2ForCausalLM

assert torch.cuda.is_available(), '请选择 Python (verl GPU)'
torch.set_num_threads(2)
SEED=2027
random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
root=Path('experiment04_output')
prior=json.loads((root/'results.json').read_text(encoding='utf-8'))
prior_eval=json.loads(Path('experiment05_output/evaluation.json').read_text(encoding='utf-8'))
vocab=prior['vocab'];stoi={c:i for i,c in enumerate(vocab)};EOS=stoi['#'];PAD=stoi['_']
baseline=Qwen2ForCausalLM.from_pretrained(root/'sft').cuda().eval().requires_grad_(False)
model=copy.deepcopy(baseline).requires_grad_(True)
model.config.use_cache=False
source_hash=hashlib.sha256((root/'sft/model.safetensors').read_bytes()).hexdigest()
print('GPU:',torch.cuda.get_device_name(0))
print('参数量:',sum(p.numel() for p in model.parameters()))''')
md('''## 2 · 先划分数对，再生成格式变体

把 `(a,b)` 和 `(b,a)` 归为同一组。先从候选组中排除之前所有训练/评估数对，再固定抽取 200 组最终测试、100 组验证；其余 0–69 数对用于训练。

每组最终评估只取一个固定方向，普通格式和前导零格式分别评分。同一数对的两种格式是配对观测，不能当成两个独立样本来夸大样本量。

额外保留 70–99 的 100 组题作为真正超出本次训练数字范围的测试，不参与训练或验证。此前已看过的第五课题目不进入新的最终测试。''')
code('''def canonical(pair):return tuple(sorted(pair))
known={canonical(p) for p in prior['train_pairs']+prior['test_pairs']}
for suite in prior_eval['suites']:
    known.update(canonical(p) for p in suite['pairs'])
rng=random.Random(SEED)
pool=[(a,b) for a in range(70) for b in range(a,70)]
fresh=[p for p in pool if p not in known]
rng.shuffle(fresh)
final_groups=set(fresh[:200]);val_groups=set(fresh[200:300])
train_groups=set(pool)-final_groups-val_groups
assert not final_groups & known
assert not (train_groups & final_groups or train_groups & val_groups or final_groups & val_groups)
train_pairs=[(a,b) for a in range(70) for b in range(70) if canonical((a,b)) in train_groups]
def orient(groups):
    return [(b,a) if rng.random()<0.5 else (a,b) for a,b in sorted(groups)]
val_pairs=orient(val_groups);final_pairs=orient(final_groups)
outside_pairs=orient(rng.sample([(a,b) for a in range(70,100) for b in range(a,100)],100))
assert all(canonical(p) not in train_groups|val_groups|final_groups|known for p in outside_pairs)
print('训练有序题数:',len(train_pairs),'验证数对:',len(val_pairs),'最终新数对:',len(final_pairs),'外推数对:',len(outside_pairs))
print('验证通过：数对、交换顺序与格式变体均不会跨训练/验证/最终测试划分。')''')
md('''## 3 · 建立数据与统一生成规则

一个训练数对有两种文本：`3+7=10#` 与 `03+07=10#`。对两位数来说，两种提示可能完全相同；我们会去重，避免重复样本增加不透明的权重。

只在答案和结束符上计算交叉熵；提示与填充的标签为 -100。生成上限统一为 4 个 token，足够覆盖三位答案和结束符。严格正确要求答案完全匹配且正常结束。

下面只报告旧模型在验证集上的表现，不提前查看最终测试成绩。''')
code(generate_source+'''
examples=sorted({(prompt_for(a,b,padded),str(a+b)) for a,b in train_pairs for padded in [False,True]})
MAX_LEN=12
ids=torch.full((len(examples),MAX_LEN),PAD,dtype=torch.long)
labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
for i,(prompt,target) in enumerate(examples):
    sequence=[stoi[c] for c in prompt+target+'#']
    assert len(sequence)<=MAX_LEN
    ids[i,:len(sequence)]=torch.tensor(sequence)
    labels[i,len(prompt):len(sequence)]=ids[i,len(prompt):len(sequence)]
    mask[i,:len(sequence)]=1
ids,labels,mask=ids.cuda(),labels.cuda(),mask.cuda()
def evaluate(m,pairs,padded=False):
    results=generate(m,[prompt_for(a,b,padded) for a,b in pairs])
    return sum(score(r,a+b) for r,(a,b) in zip(results,pairs))/len(pairs),results
base_val={fmt:evaluate(baseline,val_pairs,pad)[0] for fmt,pad in [('plain',False),('zeros',True)]}
print('去重后训练文本:',len(examples))
print('冻结模型验证准确率:',base_val)''')
md('''## 4 · 扩大覆盖后的监督训练

固定 2400 次更新，batch 128，AdamW 学习率 0.001。每 200 步展示最近一段训练损失及两种格式的验证准确率。不用最终测试选模型，也不保存验证分数最高的检查点。

如果曲线波动，先观察是否总体上升；不要根据最终测试反复调整参数后仍称它为独立测试。''')
code('''STEPS=2400;BATCH=128;LR=0.001
optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
history=[dict(step=0,loss=None,**base_val)]
live=display('',display_id=True)
start=time.perf_counter();losses=[]
for step in range(1,STEPS+1):
    model.train()
    idx=torch.randint(len(examples),(BATCH,),device='cuda')
    loss=model(input_ids=ids[idx],labels=labels[idx],attention_mask=mask[idx]).loss
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    losses.append(loss.item())
    if step%200==0:
        model.eval()
        plain,_=evaluate(model,val_pairs,False);zeros,_=evaluate(model,val_pairs,True)
        row=dict(step=step,loss=sum(losses)/len(losses),plain=plain,zeros=zeros)
        history.append(row);losses=[]
        fig,axes=plt.subplots(1,2,figsize=(10,3.2))
        axes[0].plot([r['step'] for r in history[1:]],[r['loss'] for r in history[1:]])
        axes[0].set(title='Answer-token training loss',xlabel='Step')
        for key in ['plain','zeros']:
            axes[1].plot([r['step'] for r in history],[r[key] for r in history],label=key)
        axes[1].set(title='Validation only / final test stays unused',xlabel='Step',ylim=(0,1.05))
        axes[1].legend();fig.tight_layout();live.update(fig);plt.close(fig)
        print(f"step {step:4d} | loss {row['loss']:.4f} | validation plain {plain:.1%} / zeros {zeros:.1%}",flush=True)
torch.cuda.synchronize();elapsed=time.perf_counter()-start
model.eval()
print(f'固定训练结束，用时 {elapsed:.1f} 秒。现在才开始最终测试。')''')
md('''## 5 · 最终评估：覆盖范围内与范围外分别报告

用同一批最终题对比旧模型与新模型。两种格式都测试，并额外报告真正范围外的 70–99 加法。旧模型从未覆盖 20–69 的训练数字；新模型覆盖了这些数字，但没有见过这里保留的具体最终数对。

因此，覆盖范围内的改善支持“扩大训练分布有帮助”，不能单独证明获得了无范围限制的算法能力。''')
code('''rows=[]
for suite,pairs in [('Fresh in-range',final_pairs),('Outside 70-99',outside_pairs)]:
    for fmt,padded in [('plain',False),('zeros',True)]:
        for name,m in [('before',baseline),('after',model)]:
            acc,outputs=evaluate(m,pairs,padded)
            print(f'{suite} / {fmt} / {name}: {acc:.1%}')
            for (a,b),r in zip(pairs,outputs):
                rows.append(dict(suite=suite,format=fmt,model=name,a=a,b=b,
                    prompt=prompt_for(a,b,padded),target=str(a+b),**r,correct=score(r,a+b)))
details=pd.DataFrame(rows)
summary=details.groupby(['suite','format','model'],sort=False).agg(count=('correct','size'),accuracy=('correct','mean')).reset_index()
display(summary.style.format({'accuracy':'{:.1%}'}))
wide=summary.pivot(index=['suite','format'],columns='model',values='accuracy')
ax=wide[['before','after']].plot.bar(figsize=(9,4),color=['#94a3b8','#059669'])
ax.set(ylim=(0,1.05),ylabel='Strict exact-match accuracy',xlabel='',title='Frozen old model vs expanded-coverage model')
plt.xticks(rotation=15,ha='right');plt.tight_layout();plt.show()
failures=details[(details.model=='after') & ~details.correct]
display(failures.groupby(['suite','format'],sort=False).head(5)[['suite','format','prompt','target','answer','ended']])
print('以上错误例子按固定题目顺序展示；完整预测将在下一格保存。')''')
md('''## 6 · 保存模型、数据划分与实验记录

重跑保存格会覆盖本课输出，不修改第四课模型。记录最终划分和全部预测，方便复查是否发生数据泄漏。首次查看最终结果后，它就不再是可反复调参的“新测试集”。

### 练习

保留当前结果，另建实验仅增加数字范围、不加前导零，然后用新的验证/最终划分比较格式泛化。这样才能更清楚地区分数字覆盖与格式增强各自的贡献。

本课依然是监督训练；verl 强化学习流程需要另外集成 tokenizer、数据和奖励。先把数据边界和评估做好，再比较训练算法。''')
code('''out=Path('experiment06_output');out.mkdir(exist_ok=True)
model.save_pretrained(out/'model')
details.to_csv(out/'all_predictions.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
record=dict(seed=SEED,steps=STEPS,batch=BATCH,lr=LR,elapsed_seconds=elapsed,vocab=vocab,
    source_sha256=source_hash,train_pairs=train_pairs,val_pairs=val_pairs,final_pairs=final_pairs,
    outside_pairs=outside_pairs,examples_count=len(examples),history=history,
    torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0))
(out/'results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
assert source_hash==hashlib.sha256((root/'sft/model.safetensors').read_bytes()).hexdigest()
assert all(torch.equal(a,b) for a,b in zip(baseline.state_dict().values(),Qwen2ForCausalLM.from_pretrained(root/'sft').cuda().state_dict().values()))
print('已保存:',out.resolve(),'；旧模型权重保持不变。')''')
for i,c in enumerate(cells):c['id']=f'exp06-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=previous['metadata'])
nb['metadata'].pop('widgets',None)
(ROOT/'06_data_coverage.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
