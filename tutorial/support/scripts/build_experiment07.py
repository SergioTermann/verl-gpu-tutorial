"""Build learned digit transition + explicit algorithm tutorial."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
nb5=json.loads((ROOT/'05_model_playground.ipynb').read_text(encoding='utf-8'))
generation=[''.join(c['source']) for c in nb5['cells'] if c['cell_type']=='code'][1]
generation=generation[:generation.index('for name,model in models.items():')]
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
md('''# 实验 07 · 学会一列，再逐位完成长加法

第六课的模型在训练覆盖范围内表现较好，但仍无法处理 70 至 99 的测试题。本课尝试另一种表示方式：小网络只学习一列的加法，即“两个数字和输入进位，得到结果数字和输出进位”，程序再从右到左重复调用网络。

**这是“学习局部规则 + 人工提供组合算法”的混合系统，不是 Qwen 自己发现长加法，也不是 verl RL 实验。** 程序负责倒序遍历、位对齐和传递进位；神经网络负责每一列的结果。推理函数不会调用 Python 整数加法计算答案，Python 加法只用于生成局部训练标签和评估标准答案。

局部状态只有 10×10×2=200 种，本课全部用于训练。之后的长数字测试衡量已见局部规则的组合能力，**不声称泛化到未见局部状态**。这种预先规定的计算结构称为归纳偏置：它限制了模型需要学习的内容，也改变了模型的泛化方式。

选择 Python (verl GPU)，按顺序 Shift+Enter。无需下载模型。三位及更长测试不是同架构公平对比；第六课 Qwen 只在其上下文可容纳的两位数任务上作参考。

## 1 · 构造完整的局部状态表

每列输入为 `(a_digit,b_digit,carry_in)`，标签是 0 至 19 的分类：个位与输出进位从预测类别解码。这些标签是完整的局部加法表，模型没有被要求从少量样例推导数学定律。''')
code('''import json, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output

assert torch.cuda.is_available(), '请选择 Python (verl GPU)'
torch.set_num_threads(2)
SEED=2028
random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
states=[(a,b,c) for a in range(10) for b in range(10) for c in range(2)]
inputs=torch.tensor(states,device='cuda')
targets=inputs.sum(dim=1)
def features(x):
    return torch.cat([F.one_hot(x[:,0],10),F.one_hot(x[:,1],10),F.one_hot(x[:,2],2)],dim=1).float()
network=nn.Sequential(nn.Linear(22,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,20)).cuda()
print('GPU:',torch.cuda.get_device_name(0))
print('全部局部状态:',len(states),'；参数量:',sum(p.numel() for p in network.parameters()))
display(pd.DataFrame([(*s,int(t)) for s,t in zip(states[:12],targets[:12].tolist())],columns=['a','b','carry_in','target_class']))''')
md('''## 2 · GPU 学习局部规则

全表训练固定 1000 步，不按长数字测试成绩选检查点。每 100 步显示局部表准确率。这里的 100% 是 **200 个训练状态的穷举检查**，不是独立测试准确率。

这种小任务中 GPU 未必比 CPU 快；本课使用 GPU 是为了沿用当前环境，而不是宣称这种规模存在加速优势。''')
code('''optimizer=torch.optim.AdamW(network.parameters(),lr=0.003,weight_decay=0.0)
x=features(inputs)
history=[];live=display('',display_id=True);start=time.perf_counter()
for step in range(1001):
    logits=network(x)
    loss=F.cross_entropy(logits,targets)
    if step%100==0:
        acc=(logits.argmax(-1)==targets).float().mean().item()
        history.append(dict(step=step,loss=loss.item(),table_accuracy=acc))
        fig,axes=plt.subplots(1,2,figsize=(9,3))
        axes[0].plot([r['step'] for r in history],[r['loss'] for r in history]);axes[0].set(title='Local classification loss',xlabel='Step')
        axes[1].plot([r['step'] for r in history],[r['table_accuracy'] for r in history]);axes[1].set(title='All 200 training states',xlabel='Step',ylim=(0,1.05))
        fig.tight_layout();live.update(fig);plt.close(fig)
        print(f'step {step:4d} | loss {loss.item():.5f} | local table {acc:.1%}',flush=True)
    if step==1000:break
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
network.eval()
with torch.inference_mode():
    predicted=network(x).argmax(-1)
bad=(predicted!=targets).nonzero().flatten().tolist()
print('穷举检查：',200-len(bad),'/ 200 正确；错误状态：',[states[i] for i in bad])
print(f'训练用时 {time.perf_counter()-start:.2f} 秒')''')
md('''## 3 · 显式竖式组合器

从右到左，每列调用网络。程序按预测类别拆出个位和进位，然后把预测进位送入下一列。即使某列预测错误，也照常传递，**没有用标准答案纠正中间步骤**。

数字字符串允许前导零，输出统一去掉前导零。界面上限 100 位只是为了便于显示。位对齐、循环和“类别除以 10 取商/余数”的解码都是人为提供的结构。''')
code('''def validate_number(value):
    if not isinstance(value,str) or not value or len(value)>100 or any(c not in '0123456789' for c in value):
        raise ValueError('请输入 1–100 位非负十进制整数，不含空格或符号')
    return value

@torch.inference_mode()
def column_add_many(pairs,trace=False):
    for a,b in pairs:validate_number(a);validate_number(b)
    if not pairs:return [],[]
    width=max(max(len(a),len(b)) for a,b in pairs)
    carries=torch.zeros(len(pairs),dtype=torch.long,device='cuda')
    output=[[] for _ in pairs];traces=[[] for _ in pairs]
    for position in range(width):
        aa=[int(a[-1-position]) if position<len(a) else 0 for a,b in pairs]
        bb=[int(b[-1-position]) if position<len(b) else 0 for a,b in pairs]
        state=torch.stack([torch.tensor(aa,device='cuda'),torch.tensor(bb,device='cuda'),carries],dim=1)
        predicted=network(features(state)).argmax(-1)
        digits=(predicted%10).tolist();next_carries=predicted//10
        for i,digit in enumerate(digits):
            output[i].append(str(digit))
            if trace:traces[i].append(dict(column=position+1,a=aa[i],b=bb[i],carry_in=int(carries[i]),digit=digit,carry_out=int(next_carries[i])))
        carries=next_carries
    results=[]
    for digits,carry in zip(output,carries.tolist()):
        if carry:digits.append(str(carry))
        results.append(''.join(reversed(digits)).lstrip('0') or '0')
    return results,traces

answers,traces=column_add_many([('999','1'),('0007','03')],trace=True)
print('999 + 1 →',answers[0]);display(pd.DataFrame(traces[0]))
print('0007 + 03 →',answers[1])
for invalid in ['', '-1','1.2','一','1 '*5]:
    try:validate_number(invalid)
    except ValueError:pass
    else:raise AssertionError('输入校验失败')''')
md('''## 4 · 固定长度测试与长进位链

固定种子 2028，每种长度生成 100 个随机题目，另测 `99…9 + 1` 的长进位链。两位数组全部来自 70 至 99，超出第六课的训练操作数范围。

神经网络局部分类只覆盖 200 个已训练状态；长序列成功来自这些状态的正确组合。随机 100 题全对不证明对所有输入都正确；但若全部 200 状态均正确，配合正确的竖式组合器，可以用数学归纳解释为何能处理任意有限长度。''')
code('''rng=random.Random(SEED)
def number(length):
    return str(rng.randrange(1,10))+''.join(str(rng.randrange(10)) for _ in range(length-1))
test_sets={2:[(str(rng.randrange(70,100)),str(rng.randrange(70,100))) for _ in range(100)]}
for length in [4,8,16,32]:test_sets[length]=[(number(length),number(length)) for _ in range(100)]
rows=[]
for length,pairs in test_sets.items():
    outputs,_=column_add_many(pairs)
    for (a,b),answer in zip(pairs,outputs):
        target=str(int(a)+int(b))  # 仅在评估时计算标准答案
        rows.append(dict(suite='random',length=length,a=a,b=b,prediction=answer,target=target,correct=answer==target))
for length in [2,4,8,16,32,64]:
    a,b='9'*length,'1'
    answer=column_add_many([(a,b)])[0][0]
    target=str(int(a)+int(b))
    rows.append(dict(suite='carry_chain',length=length,a=a,b=b,prediction=answer,target=target,correct=answer==target))
details=pd.DataFrame(rows)
summary=details.groupby(['suite','length']).agg(count=('correct','size'),accuracy=('correct','mean')).reset_index()
display(summary.style.format({'accuracy':'{:.1%}'}))
display(details[details.suite=='carry_chain'][['length','a','b','prediction','correct']])
ax=summary[summary.suite=='random'].plot.bar(x='length',y='accuracy',legend=False,color='#059669',figsize=(7,3))
ax.set(ylim=(0,1.05),xlabel='Digits per operand',ylabel='Exact-match accuracy',title='Learned local transition + explicit column loop')
plt.tight_layout();plt.show()''')
md('''## 5 · 和第六课端到端模型做有限对照

只用相同的 100 道两位数题进行比较，Qwen 的生成上限为 4 token，足够容纳三位答案与结束符。它和逐位系统的结构、监督信息、训练任务都不同，不能据此声称某种网络架构普遍更好。

第六课模型并没有接受完整的 200 状态竖式表监督，也没有得到显式位循环。这一对照用于展示**结构先验如何改变泛化方式**。''')
code('''from transformers import Qwen2ForCausalLM
prior=Path('support/experiment06_output')
assert (prior/'model/model.safetensors').exists(), '请先完成第六课保存格'
vocab=json.loads((prior/'results.json').read_text(encoding='utf-8'))['vocab']
stoi={c:i for i,c in enumerate(vocab)};EOS=stoi['#']
qwen=Qwen2ForCausalLM.from_pretrained(prior/'model').cuda().eval().requires_grad_(False)
'''+generation+'''
pairs=test_sets[2]
outputs=generate(qwen,[f'{a}+{b}=' for a,b in pairs])
qwen_accuracy=sum(score(r,int(a)+int(b)) for r,(a,b) in zip(outputs,pairs))/len(pairs)
local_accuracy=details[(details.suite=='random')&(details.length==2)].correct.mean()
print(f'相同 100 道 70–99 加法：Qwen {qwen_accuracy:.1%} / 逐位混合系统 {local_accuracy:.1%}')
comparison=pd.DataFrame([dict(a=a,b=b,target=str(int(a)+int(b)),qwen=r['answer'],qwen_ended=r['ended']) for (a,b),r in zip(pairs,outputs)])
display(comparison.head(10))
del qwen
torch.cuda.empty_cache()''')
md('''## 6 · 交互竖式面板

输入两个最多 100 位的整数，点击按钮可以看到每一列的网络预测。Python 加法只显示标准答案和核验结果，不参与模型推理。按钮需要当前内核运行；重新打开 Notebook 后应按顺序执行前面的代码格。''')
code('''a_box=widgets.Text(value='9999999999999999',description='加数 A')
b_box=widgets.Text(value='1',description='加数 B')
button=widgets.Button(description='查看逐位推理',button_style='success')
area=widgets.Output()
def clicked(_):
    button.disabled=True
    try:
        with area:
            clear_output(wait=True)
            try:
                a,b=validate_number(a_box.value),validate_number(b_box.value)
                answers,traces=column_add_many([(a,b)],trace=True)
                print('混合系统答案:',answers[0])
                print('Python 标准答案:',str(int(a)+int(b)))
                print('一致:',answers[0]==str(int(a)+int(b)))
                display(pd.DataFrame(traces[0]))
            except ValueError as error:print(error)
    finally:button.disabled=False
button.on_click(clicked)
display(widgets.VBox([a_box,b_box,button,area]))
clicked(None)''')
md('''## 7 · 保存结果与复盘

保存局部网络权重、全部长数字测试结果、两位数 Qwen 对照和训练记录。输出独立保存到 `support/experiment07_output/`，不修改第六课权重。

回顾两种方法： 端到端模型需要从文本里学表示和计算；本课由人提供分解与组合规则，网络只拟合有限的局部真值表。长度泛化明显改善，但计算结构主要来自程序设计。若目标只是可靠加法，直接用确定性的计算器更简单，本实验的价值是理解归纳偏置和组合泛化。

练习：在新实验中故意移除带进位的局部训练状态，观察长进位链怎样放大局部错误。不要把这项实验与本课完整真值表训练混为同一设置。

这是独立的 PyTorch 教学实验，不是 verl 大语言模型训练结果。''')
code('''out=Path('support/experiment07_output');out.mkdir(exist_ok=True)
torch.save(network.state_dict(),out/'digit_network.pt')
details.to_csv(out/'all_predictions.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
comparison.to_csv(out/'qwen_comparison.csv',index=False)
record=dict(seed=SEED,steps=1000,lr=0.003,training_states=states,
    local_correct=200-len(bad),history=history,qwen_accuracy=qwen_accuracy,
    two_digit_hybrid_accuracy=float(local_accuracy),architecture=[22,128,128,20],
    torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),
    disclosure='All 200 local states trained; explicit column traversal and carry propagation supplied by programmer.')
(out/'results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('已保存:',out.resolve())''')
for i,c in enumerate(cells):c['id']=f'exp07-{i:02d}'
metadata=dict(kernelspec=dict(display_name='Python (verl GPU)',language='python',name='verl-gpu'),language_info=dict(name='python',version='3.12'))
(ROOT/'07_column_reasoning.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=cells),ensure_ascii=False,indent=2),encoding='utf-8')
