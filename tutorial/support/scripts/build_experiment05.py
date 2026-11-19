"""Build an interactive, source-grounded model evaluation lesson."""
import json
from pathlib import Path

cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))

md('''# 实验 05 · 模型真的学会加法了吗？

第四课保存了冻结基线、继续 SFT 和奖励微调三个模型。本课先把它们放进同一个交互面板，逐题比较答案，再改变数字范围和输入格式，检查它们在什么条件下容易出错。

本课只进行推理和评估。模型答案由第四课保存的 Qwen2 在 GPU 上生成；Python 另行计算标准答案，用于评分。整个过程保持模型权重不变。

选择 **Python (verl GPU)**，从上到下 Shift+Enter。需要先完成第四课的保存格。前面的曲线反映固定小数据集上的表现，本课进一步区分记忆、域内泛化和分布外表现。

按顺序运行后，页面会显示评估图表。第 5 节的交互按钮需要执行面板代码后才能使用。

## 1 · 加载三组权重

只读取 `support/experiment04_output/`。每组权重的 SHA-256 会写入评估报告，方便追溯实际测试的是哪个模型。''')
code('''import json, hashlib, random, time
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from transformers import Qwen2ForCausalLM

assert torch.cuda.is_available(), '请选择 Python (verl GPU) 内核'
torch.set_num_threads(2)
root=Path('support/experiment04_output')
assert (root/'results.json').exists(), '请先完成第四课的保存格'
source=json.loads((root/'results.json').read_text(encoding='utf-8'))
vocab=source['vocab']; stoi={c:i for i,c in enumerate(vocab)}; EOS=stoi['#']
models={name:Qwen2ForCausalLM.from_pretrained(root/name).cuda().eval().requires_grad_(False)
        for name in ['baseline','sft','rl']}
hashes={name:hashlib.sha256((root/name/'model.safetensors').read_bytes()).hexdigest() for name in models}
train_pairs={tuple(p) for p in source['train_pairs']}
test_pairs=[tuple(p) for p in source['test_pairs']]
print('GPU:',torch.cuda.get_device_name(0))
print('已加载：',list(models),'；模型均冻结，无梯度更新。')''')
md('''## 2 · 用统一规则生成答案

三个模型都采用贪心解码，最多生成 4 个 token，可容纳三位答案和结束符。原训练最多只需要两位答案；统一放宽上限避免把三位数测试的失败仅归因于生成长度不够。

只把题目传入模型，答案只用于之后评分。我们保存原始生成字符串、是否出现结束符，以及结束符之前的答案文本。严格正确要求“答案完全正确且正常结束”。另外报告文本匹配率，便于与第四课的评分方式对照。

本模型词表只有数字与少量符号，不支持中文、空格或自然语言提问。前导零测试仍使用词表内字符，但改变了训练中未见过的格式。''')
code('''def prompt_for(a,b,leading_zero=False):
    return f'{a:02d}+{b:02d}=' if leading_zero else f'{a}+{b}='

@torch.inference_mode()
def generate(model,prompts,max_new=4):
    results=[None]*len(prompts)
    buckets={}
    for i,p in enumerate(prompts):
        if any(c not in stoi for c in p):
            raise ValueError('提示中含有词表外字符')
        if len(p)+max_new>model.config.max_position_embeddings:
            raise ValueError('超出模型上下文长度')
        buckets.setdefault(len(p),[]).append((i,p))
    for length,rows in buckets.items():
        for offset in range(0,len(rows),128):
            chunk=rows[offset:offset+128]
            x=torch.tensor([[stoi[c] for c in p] for _,p in chunk],device='cuda')
            for _ in range(max_new):
                nxt=model(input_ids=x,use_cache=False).logits[:,-1].argmax(-1,keepdim=True)
                x=torch.cat([x,nxt],dim=1)
            for (i,_),tokens in zip(chunk,x[:,length:].tolist()):
                raw=''.join(vocab[t] for t in tokens)
                results[i]=dict(answer=raw.split('#')[0],ended='#' in raw,raw=raw)
    return results

def score(result,target):
    return result['answer']==str(target) and result['ended']

# 验证评分不会把格式错误或未结束的输出算作严格正确。
assert score(dict(answer='12',ended=True),12)
assert not score(dict(answer='12',ended=False),12)
assert not score(dict(answer='012',ended=True),12)
for name,model in models.items():
    print(name,generate(model,['3+7=','19+19=']))''')
md('''## 3 · 提前固定五类评估

| 分组 | 问题 | 用途 |
|---|---|---|
| Seen train | 原训练集全部 320 题 | 观察记忆效果，不作为泛化证据 |
| Held-out | 原独立测试集全部 80 题 | 复查域内泛化 |
| Leading zeros | 同一 80 题，写成 `03+07=` | 同题格式变化对照 |
| New operands | 从两个加数均为 20 至 39 的题中固定抽取 100 题 | 新数字范围；答案仍只有两位 |
| Three-digit sum | 从两个加数均为 50 至 69 的题中固定抽取 100 题 | 新数字范围 + 三位答案，两个变化同时存在 |

后两组都不在训练集。三位答案组不能单独归因于答案长度，因为操作数也改变了。测试集合在生成结果之前固定，随机种子为 2026，不挑选成功案例。''')
code('''rng=random.Random(2026)
new_operands=rng.sample([(a,b) for a in range(20,40) for b in range(20,40)],100)
three_digit=rng.sample([(a,b) for a in range(50,70) for b in range(50,70)],100)
suites=[('Seen train',sorted(train_pairs),False),('Held-out',test_pairs,False),
        ('Leading zeros',test_pairs,True),('New operands',new_operands,False),
        ('Three-digit sum',three_digit,False)]
assert all(p not in train_pairs for p in new_operands+three_digit)
rows=[]
start=time.perf_counter()
for suite,pairs,padded in suites:
    prompts=[prompt_for(a,b,padded) for a,b in pairs]
    for name,model in models.items():
        outputs=generate(model,prompts)
        for (a,b),prompt,result in zip(pairs,prompts,outputs):
            rows.append(dict(suite=suite,model=name,a=a,b=b,prompt=prompt,target=str(a+b),
                **result,correct=score(result,a+b),text_match=result['answer']==str(a+b)))
        print(f'{suite:16s} / {name:8s}：完成 {len(pairs)} 题',flush=True)
details=pd.DataFrame(rows)
summary=details.groupby(['suite','model'],sort=False).agg(
    count=('correct','size'),accuracy=('correct','mean'),text_match=('text_match','mean'),
    termination_rate=('ended','mean')).reset_index()
print(f'评估用时 {time.perf_counter()-start:.1f} 秒')
display(summary.style.format({'accuracy':'{:.1%}','text_match':'{:.1%}','termination_rate':'{:.1%}'}))''')
md('''## 4 · 一张图看清能力边界

纵轴固定 0 至 100%。对比同一模型在不同组上的表现，比只看一个总分更有意义。前导零组与原测试题是配对比较，其差异反映格式改变的影响。

以下静态表展示每个非训练分组的前 3 题与三组模型答案，不按正确与否挑选。所有题目的完整结果都会保存为 CSV。''')
code('''order=[name for name,_,_ in suites]
wide=summary.pivot(index='suite',columns='model',values='accuracy').reindex(order)
ax=wide[['baseline','sft','rl']].plot.bar(figsize=(10,4),color=['#94a3b8','#2563eb','#059669'])
ax.set(ylim=(0,1.05),ylabel='Strict exact-match accuracy',xlabel='',title='Same models, different evaluation conditions')
plt.xticks(rotation=15,ha='right');plt.tight_layout();plt.show()
sample=details[details.suite!='Seen train'].groupby(['suite','model'],sort=False).head(3)
display(sample[['suite','model','prompt','target','answer','ended','correct']])
for name in models:
    original=summary[(summary.suite=='Held-out')&(summary.model==name)].accuracy.iloc[0]
    changed=summary[(summary.suite=='Leading zeros')&(summary.model==name)].accuracy.iloc[0]
    print(f'{name}：前导零格式造成 {(changed-original)*100:+.1f} 个百分点变化')''')
md('''## 5 · 交互实验台：输入一道题，三个模型现场作答

输入 0 至 99 的两个整数，点击“比较三个模型”。右侧会标明这是训练题、原测试题还是分布外输入。勾选前导零可以比较同一道题的格式变化。

结果中“模型答案”来自 GPU 推理；“标准答案”由 Python 独立计算。输入范围超过训练范围时不要期待可靠结果。按钮只触发推理，不更新模型参数。

你可以尝试 `3+7`、`19+19`、`25+30`、`60+65`。交互面板只在当前 Notebook 内核活跃时可用。''')
code('''a_input=widgets.BoundedIntText(value=3,min=0,max=99,description='加数 A')
b_input=widgets.BoundedIntText(value=7,min=0,max=99,description='加数 B')
format_input=widgets.Checkbox(value=False,description='前导零格式')
run_button=widgets.Button(description='比较三个模型',button_style='success',icon='play')
output=widgets.Output()

def compare_values(a,b,padded=False):
    prompt=prompt_for(a,b,padded)
    group='训练题' if (a,b) in train_pairs else '原测试题' if (a,b) in test_pairs else '分布外数字'
    if padded and prompt!=prompt_for(a,b): group+=' + 新格式'
    print(f'输入：{prompt} | 分类：{group} | 标准答案：{a+b}')
    records=[]
    for name,model in models.items():
        result=generate(model,[prompt])[0]
        records.append({'模型':name,'模型答案':result['answer'],'标准答案':a+b,
                        '正常结束':result['ended'],'严格正确':score(result,a+b)})
    return pd.DataFrame(records)

def on_compare(_):
    run_button.disabled=True
    try:
        with output:
            clear_output(wait=True)
            display(compare_values(a_input.value,b_input.value,format_input.value))
    finally:
        run_button.disabled=False

run_button.on_click(on_compare)
display(widgets.VBox([widgets.HBox([a_input,b_input]),format_input,run_button,output]))
on_compare(None)
print('静态示例（无需按钮也能查看）：')
display(compare_values(25,30))''')
md('''## 6 · 保存报告，不改变模型

保存全部题目、原始输出、分组指标和来源权重哈希。最后重新计算权重文件哈希，确认没有覆盖第四课的模型。结果写入 `support/experiment05_output/`，重复运行会更新本课的报告。

**复盘：** 高训练准确率不等于会做任意加法。若域内表现好、换数字或格式就失败，说明当前学习依赖训练分布。改进方向是更合理的数据覆盖、表示方式与独立评估，而不是只增加同一批题目的训练步数。

后续可以设计“增加格式与范围覆盖”的训练实验，用训练/验证集决定设置，并另外保留全新的最终测试集。本课已查看的分布外题目之后就不能再声称是完全未见的最终评估。''')
code('''out=Path('support/experiment05_output');out.mkdir(exist_ok=True)
details.to_csv(out/'all_predictions.csv',index=False)
summary.to_csv(out/'summary.csv',index=False)
metadata=dict(seed=2026,decoding='greedy',max_new_tokens=4,strict_requires_eos=True,
    weights_sha256=hashes,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
    suites=[dict(name=name,pairs=pairs,leading_zero=padded) for name,pairs,padded in suites])
(out/'evaluation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
assert hashes=={name:hashlib.sha256((root/name/'model.safetensors').read_bytes()).hexdigest() for name in models}
print('评估结果已保存:',out.resolve())
print('验证通过：第四课权重文件未改变。')''')
for i,c in enumerate(cells): c['id']=f'exp05-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(
    display_name='Python (verl GPU)',language='python',name='verl-gpu'),language_info=dict(name='python',version='3.12')))
(Path(__file__).resolve().parents[2] / '05_model_playground.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
