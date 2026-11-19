import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
md('''# 实验 12 · 诊断稀疏奖励

上一课证明真实 verl 工程链路可运行。本课把合成字节奖励换成明确任务：读入两个十进制数字，判断数值是否为偶数，只接受单个 `yes` 或 `no`。

实验使用 Ray、vLLM 和 FSDP 完成固定 10 步 GRPO，按数值划分 70 个训练样本、10 个验证样本和 20 个最终测试样本，随机种子为 12012。历史实测中，训练采样偶尔获得奖励，只有 3 步出现非零梯度；训练前后的贪心测试输出均不符合协议，准确率都为 0。本课据此区分三个问题：训练是否执行、输出格式是否合格，以及任务是否学会。

最终评估使用预先固定的第 10 步模型和原始测试题，不根据测试成绩挑选检查点或修改提示。测试集较小，每题占 5 个百分点，不能从单次成绩得出稳定结论。

## 1 · 查看真实配置和数据

初次排错修正了两个配置：mini batch 按提示数设置为 14；奖励和 AgentLoop worker 都设为 1，以免 WSL 主机内存耗尽。失败试跑保存在各自 run 目录，最终这轮没有内存压力。

提示中的 `or` 不在词表，编码为 UNK，这是当前教学模型输入协议的限制；合法输出 yes/no 都在词表。后续改进应一并规范提示词表，不能暗中当成完全相同的实验。''')
code('''import json,sys,importlib.util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
scripts=Path('support/advanced12').resolve()
# latest_run.txt 保存的是本次训练目录的路径。先读出路径，再去那个目录找日志、指标或检查点。
run_root=Path((scripts/'latest_run.txt').read_text().strip())
print('本课读取已完成的真实训练，不会自动重跑。')
print('运行目录:',run_root)
print((scripts/'run.sh').read_text())
splits=json.loads((run_root/'splits.json').read_text())
display(pd.DataFrame([dict(split=k,count=len(v),even=sum(n%2==0 for n in v),odd=sum(n%2==1 for n in v)) for k,v in splits.items()]))''')
md('''## 2 · 严格奖励与漏洞检查

奖励是答案正确性，而不是是否包含某个词。只输出 `yes no`、重复答案或附加解释均不得分。这种协议让任务容易评分，但随机模型冷启动时可能几乎无法获得奖励。

大小写和首尾空白会归一化；这与评分标准一致。这里不要求 EOS，只按解码后的文本评分，与训练奖励保持一致。''')
code('''spec=importlib.util.spec_from_file_location('reward12',scripts/'reward.py')
reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
# 每个三元组依次是模型回答、标准答案、预期奖励。逐项喂给真实评分函数，检查大小写、空白和含糊回答的边界。
cases=[('yes','yes',1),('no','yes',0),('yes no','yes',0),('YES ','yes',1),('','yes',0),('yes yes','yes',0)]
for answer,target,expected in cases:assert reward.compute_score('decimal_parity',answer,target)==expected
print((scripts/'reward.py').read_text())
print('PASS：正确答案、错误答案、混合答案、重复答案与空输出测试通过。')''')
md('''## 3 · 固定检查点评估与梯度诊断

评估脚本直接加载初始权重和第 10 步 FSDP 权重，用相同的贪心解码、2-token 上限、同一个 tokenizer 比较。最终测试题仅在训练结束后评估，没有进入 verl 的 validation 文件。

把合法输出比例、任务准确率和梯度范数放在一起看。本轮有少量非零梯度，说明发生过更新，但最终测试准确率没有改善。单独查看 loss 或参数变化，都不足以判断模型是否学会了任务。

优化器 weight decay 可能在梯度为零时引起微小权重变化，所以“参数变了”也不是“任务学会了”的充分证据。''')
code('''spec=importlib.util.spec_from_file_location('evaluate12',scripts/'evaluate.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
report=evaluation.evaluate(run_root)
summary=pd.read_csv(run_root/'evaluation.csv')
display(summary)
print(json.dumps(report,indent=2))
metrics=pd.read_csv(run_root/'metrics.csv')
fig,axes=plt.subplots(1,2,figsize=(10,3))
# 按同一个训练步对照两张图：平均奖励描述这一批的得分，梯度范数描述 actor 收到的更新信号。组内同分也可能出现非零平均奖励。
axes[0].plot(metrics['training/global_step'],metrics['critic/rewards/mean'],marker='o')
axes[0].set(title='Actual rollout mean reward',xlabel='Step',ylim=(-.05,1.05))
axes[1].plot(metrics['training/global_step'],metrics['actor/grad_norm'],marker='o')
axes[1].set(title='Actual actor gradient norm',xlabel='Step')
plt.tight_layout();plt.show()
predictions=pd.read_csv(run_root/'predictions.csv',keep_default_na=False)
display(predictions[predictions.split=='test'])
assert report['steps']==10 and not report['memory_pressure']
''')
md('''## 4 · 为什么从零 RL 容易失败？

随机策略要先选中 yes/no，并在第二 token 停止或输出会被忽略的特殊 token，才有机会得到严格奖励。每个问题只有 4 个样本；若全部奖励为 0，组内优势为 0，没有方向信息。增加训练步数并不自动解决这个问题。

合理的下一步是：只用训练集做明确记录的 SFT 格式/任务预热，再从同一个预热起点比较冻结、继续 SFT 和 GRPO。验证集可以用于诊断，但看过的最终测试集不能继续被当成全新测试集。

如果将评分改成“包含 yes 就得分”，就改变了任务目标，并可能引入奖励漏洞。评估时也应与多数类别基线比较，区分固定猜测与根据输入作答。

## 5 · 重跑与报告

本 Notebook 读取并评估本地训练产物。首次使用或希望完整重跑时，在 WSL 已激活环境的项目目录执行：

```bash
bash tutorial/support/advanced12/run.sh
```

它会新建独立运行目录，从随机初始模型重来。完整日志和两次检查点留在 WSL；本格保存精简报告到工作区。预计数分钟；先关闭其他训练任务，别并发运行多个副本。

这一课的结论是“真实训练完成，但任务奖励冷启动失败”，不是性能提升。下一课再设计预热与可比对照。''')
code('''out=Path('support/experiment12_output');out.mkdir(exist_ok=True)
(out/'report.json').write_text(json.dumps(report,indent=2))
for name in ['evaluation.csv','metrics.csv','predictions.csv']:
    (out/name).write_bytes((run_root/name).read_bytes())
print('已保存:',out.resolve())
print('PASS：10 步真实训练完成，最终测试和失败诊断已记录；不宣称性能提升。')''')
md('''---

<a href="https://github.com/SergioTermann"><img src="support/assets/logo.png" alt="SergioTermann" width="44" height="44" align="left" style="margin: 0 12px 0 0;"></a>

**更多有趣的 AI 课程**  
[← 访问 SergioTermann 的 GitHub 主页](https://github.com/SergioTermann)''')
for i,c in enumerate(cells):c['id']=f'advanced12-{i:02d}'
meta=dict(kernelspec=dict(display_name='Python (verl GPU)',language='python',name='verl-gpu'),language_info=dict(name='python',version='3.12'))
(root/'12_诊断稀疏奖励.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=meta,cells=cells),ensure_ascii=False,indent=2),encoding='utf-8')
