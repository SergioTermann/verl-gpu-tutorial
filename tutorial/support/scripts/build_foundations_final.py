"""Generate final foundational lessons without rewriting previous experiments."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
metadata=dict(kernelspec=dict(display_name='Python (verl GPU)',language='python',name='verl-gpu'),language_info=dict(name='python',version='3.12'))
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
def save(name):
    for i,c in enumerate(cells):c['id']=f'cell-{i:02d}'
    (ROOT/name).write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=cells),ensure_ascii=False,indent=2),encoding='utf-8')
    cells.clear()

md('''# 第 08 课 · 采样、探索与奖励漏洞

前面的奖励学习先生成多个答案，再对答案评分。本课分别检查这两个环节：改变温度，观察答案分布如何变化；构造容易误判的回答，检查奖励函数是否真的符合任务要求。

本课只做推理，不更新权重。加载第四课只预热了 200 步的基线模型，以便观察多样性。固定 5 道题，每题每种温度采样 64 次；这些是诊断样例，不用于宣称测试性能。

## 1 · 温度与采样

温度 T 用 logits/T 改变采样分布。较高温度通常增加多样性，但不保证提升准确率。贪心解码直接取最大概率 token，不等同于某个非零温度的随机采样。

比较温度 0.5、1.0、1.5；输出上限均为 4 token。EOS 后字符不计入答案。''')
code('''import json, torch, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from transformers import Qwen2ForCausalLM
assert torch.cuda.is_available()
torch.set_num_threads(2)
root=Path('support/experiment04_output')
info=json.loads((root/'results.json').read_text())
vocab=info['vocab'];stoi={c:i for i,c in enumerate(vocab)}
model=Qwen2ForCausalLM.from_pretrained(root/'baseline').cuda().eval().requires_grad_(False)
@torch.inference_mode()
def sample(a,b,temperature,count=64):
    prompt=f'{a}+{b}='
    x=torch.tensor([[stoi[c] for c in prompt]],device='cuda').repeat(count,1)
    for _ in range(4):
        logits=model(input_ids=x,use_cache=False).logits[:,-1]/temperature
        token=torch.distributions.Categorical(logits=logits).sample()
        x=torch.cat([x,token[:,None]],dim=1)
    raw=[''.join(vocab[t] for t in row) for row in x[:,len(prompt):].tolist()]
    return [(r.split('#')[0],'#' in r) for r in raw]
rows=[]
for temperature in [0.5,1.0,1.5]:
    torch.manual_seed(2029);torch.cuda.manual_seed_all(2029)
    for a,b in [(3,7),(12,7),(19,19),(8,9),(15,6)]:
        for answer,ended in sample(a,b,temperature):
            rows.append(dict(temperature=temperature,prompt=f'{a}+{b}=',answer=answer,
                target=str(a+b),ended=ended,correct=ended and answer==str(a+b)))
predictions=pd.DataFrame(rows)
per_question=predictions.groupby(['temperature','prompt']).agg(accuracy=('correct','mean'),distinct=('answer','nunique')).reset_index()
summary=per_question.groupby('temperature')[['accuracy','distinct']].mean()
display(summary)
summary.plot(subplots=True,marker='o',figsize=(7,5),title=['Sample accuracy','Mean distinct answers per question'])
plt.tight_layout();plt.show()
display(predictions.groupby(['temperature','prompt','answer']).size().rename('count').reset_index().head(20))''')
md('''## 2 · 奖励漏洞：得分高不等于回答好

若奖励只检查回答中是否出现目标字符串，`120`、`12 或 99` 甚至列举所有数字都可能被误判为正确。这是奖励函数设计问题，不代表模型能力提升。

下面是人工构造的攻击样例，**不是声称训练模型自动发现了漏洞**。本教程的纯数字任务要求规范整数字符串；如果将来允许自然语言解释，应重新定义输出协议与解析规则，不能直接照搬严格匹配。''')
code('''def loose_reward(answer,target):return float(str(target) in answer)
def strict_reward(answer,target):return float(answer.strip()==str(target))
cases=[('12',1),('120',0),('答案是 12 或 99',0),('012',0),('',0),(' 12 ',1),(' '.join(map(str,range(200))),0)]
audit=pd.DataFrame([dict(answer=s if len(s)<50 else s[:50]+'…',expected=e,loose=loose_reward(s,12),strict=strict_reward(s,12)) for s,e in cases])
assert all(strict_reward(s,12)==e for s,e in cases)
display(audit)
print('宽松奖励误判数量:',sum(loose_reward(s,12)!=e for s,e in cases))
out=Path('support/experiment08_output');out.mkdir(exist_ok=True)
predictions.to_csv(out/'sampling.csv',index=False);audit.to_csv(out/'reward_audit.csv',index=False)
summary.to_csv(out/'summary.csv')''')
md('''## 3 · 小结与练习

GRPO 的组大小、温度和奖励函数共同影响有效学习信号。只增加采样数并不能修复错误奖励。

练习：为 `-12`、`12.0`、多行文本、空白字符各写一个测试，并先说明是否应该接受。用业务目标定义预期，再实现奖励。

这节结果只涵盖固定诊断题；若要比较解码策略，应预先规定完整评估集、随机种子和采样预算。''')
save('08_sampling_and_rewards.ipynb')

md('''# 第 09 课 · 保存、恢复与可复现训练

恢复模型权重后可以继续推理；要让训练沿原来的轨迹继续，还需恢复优化器动量、随机数状态和训练步数。本课用一个小型 GPU 分类器比较连续训练与中断后恢复训练，直接检查最终参数是否一致。

checkpoint（检查点）是某个训练时刻保存的状态。为了集中观察保存与恢复机制，本课使用合成分类数据，不代表语言模型性能。只加载本课自己创建的可信 checkpoint；Python pickle 格式不应用于不可信来源。

## 1 · 连续运行、权重恢复、完整恢复三条路径

先训练 20 步保存，再连续训练到 40 步；然后从同一个保存点恢复，分别测试完整状态恢复和只恢复权重。保存是深拷贝，避免之后训练修改内存中的 checkpoint。''')
code('''import copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
assert torch.cuda.is_available()
torch.set_num_threads(2)
torch.manual_seed(2030);torch.cuda.manual_seed_all(2030)
random.seed(2030);np.random.seed(2030)
data=torch.randn(256,8,device='cuda');labels=(data[:,0]+data[:,1]>0).long()
def new_model():return nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,2)).cuda()
def advance(model,opt,count):
    model.train();losses=[]
    for _ in range(count):
        idx=torch.randint(len(data),(32,),device='cuda')
        loss=nn.functional.cross_entropy(model(data[idx]),labels[idx])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
    return losses
model=new_model();opt=torch.optim.AdamW(model.parameters(),lr=.01)
advance(model,opt,20)
checkpoint=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=20,
    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
    python_rng=random.getstate(),numpy_rng=np.random.get_state()))
out=Path('support/experiment09_output');out.mkdir(exist_ok=True)
torch.save(checkpoint,out/'trusted_training_checkpoint.pt')
continuous_losses=advance(model,opt,20)
continuous=copy.deepcopy(model.state_dict())

def restore_rng(saved):
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
saved=torch.load(out/'trusted_training_checkpoint.pt',weights_only=False,map_location=None)
restored=new_model();restored.load_state_dict(saved['model'])
restored_opt=torch.optim.AdamW(restored.parameters(),lr=.01);restored_opt.load_state_dict(saved['optimizer'])
restore_rng(saved)
restored_losses=advance(restored,restored_opt,20)
full_difference=max((continuous[k]-v).abs().max().item() for k,v in restored.state_dict().items())
assert full_difference<1e-6, f'恢复结果超出容差: {full_difference}'

weights_only=new_model();weights_only.load_state_dict(saved['model'])
fresh_opt=torch.optim.AdamW(weights_only.parameters(),lr=.01)
restore_rng(saved)
advance(weights_only,fresh_opt,20)
weights_difference=max((continuous[k]-v).abs().max().item() for k,v in weights_only.state_dict().items())
print('完整状态恢复后的最大参数差:',full_difference)
print('仅恢复权重后的最大参数差:',weights_difference)
assert weights_difference>full_difference
(out/'verification.json').write_text(json.dumps(dict(full_difference=full_difference,weights_only_difference=weights_difference,step=40),indent=2))''')
md('''## 2 · 推理模式、显存与复现边界

`eval()` 关闭 dropout 等训练行为，不会自动关闭梯度；`inference_mode()` 关闭梯度记录。二者用途不同。删除引用后 `empty_cache()` 可归还未使用的缓存，但不能释放仍被变量引用的 GPU 张量。

本课在相同硬件、软件、数据与单进程设置下比较恢复。实际分布式训练还需保存学习率调度器、混合精度 scaler、数据采样器和每个 rank 的状态。不同 CUDA/库版本不保证逐位一致。

## 3 · 练习

把 dropout 从 0.2 改成 0.5，从头执行；再在独立副本中故意不恢复随机状态，观察差异。恢复训练前检查权重、优化器、随机状态和当前步数是否配套。

Notebook 中“重跑训练格”通常意味着继续训练；要复现同一起点，应重启内核并按顺序运行。''')
code('''restored.eval()
with torch.inference_mode():
    accuracy=(restored(data).argmax(-1)==labels).float().mean().item()
print(f'合成训练数据准确率: {accuracy:.1%}（不是独立测试）')
print(f'当前 PyTorch 已分配显存: {torch.cuda.memory_allocated()/2**20:.1f} MiB')
print('PASS：完整 checkpoint 恢复验证通过。')''')
save('09_checkpoints_and_reproducibility.ipynb')

md('''# 第 10 课 · 把基础知识接到真实 verl

前面已经分别练习了训练、评估和保存。本课把这些组件接到 verl：导出 Hugging Face 模型与 tokenizer（分词器），准备 Parquet 数据和奖励函数，再用当前安装的 verl 加载数据并检查训练配置。

本课的检查范围是文件格式、数据接口与配置解析。完整的 Ray/vLLM 训练将在第 11 课使用另一套微型模型与配置验证。 模型是第六课的小字符模型，奖励纯数字答案；不具备通用聊天能力。

## 1 · 补全可保存的字符 tokenizer

保留第六课的原始 token ID，按单字符切分，并验证编码、解码和 chat template 与训练时的提示一致。聊天模板只拼接内容，适合本教学单轮任务，不是生产多轮对话模板。新增未知字符 token 后扩展嵌入，但已训练 token ID 不变。''')
code('''import json,sys,subprocess,importlib.util
from pathlib import Path
import pandas as pd
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,decoders
from transformers import PreTrainedTokenizerFast,Qwen2ForCausalLM,AutoTokenizer
from omegaconf import OmegaConf

source=Path('support/experiment06_output')
record=json.loads((source/'results.json').read_text())
vocab=record['vocab']
out=Path('support/experiment10_output').resolve();out.mkdir(exist_ok=True)
model_dir=out/'model'
mapping={c:i for i,c in enumerate(vocab)};mapping['[UNK]']=len(mapping)
raw=Tokenizer(models.WordLevel(mapping,unk_token='[UNK]'))
raw.pre_tokenizer=pre_tokenizers.Split(pattern='',behavior='isolated')
raw.decoder=decoders.Fuse()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='_',eos_token='#',unk_token='[UNK]')
tokenizer.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
for text in ['3+7=','03+07=10#','69+69=138#']:
    ids=tokenizer.encode(text,add_special_tokens=False)
    assert ids==[mapping[c] for c in text]
    assert tokenizer.decode(ids,skip_special_tokens=False)==text
tokenizer.save_pretrained(model_dir)
model=Qwen2ForCausalLM.from_pretrained(source/'model')
model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
model.save_pretrained(model_dir)
loaded=AutoTokenizer.from_pretrained(model_dir)
assert loaded.apply_chat_template([dict(role='user',content='3+7=')],tokenize=True,add_generation_prompt=True)==[mapping[c] for c in '3+7=']
print('PASS：token ID、解码、chat template 和保存重载一致。')''')
md('''## 2 · 创建真实 Parquet 与奖励函数

使用第六课训练数对的前 32 题和验证数对的前 8 题作为接口检查样本，不使用最终测试题。每行包含 `prompt` 消息列表、`data_source` 和 `reward_model.ground_truth`。

奖励函数接受 verl 的调用参数。它只能读取正确答案用于评分，不能在生成提示中泄漏答案。这里是严格数字协议，输出两种候选答案或解释文本均不得分。''')
code('''def make_rows(pairs):
    return [dict(data_source='tutorial_addition',prompt=[dict(role='user',content=f'{a}+{b}=')],
                 reward_model=dict(style='rule',ground_truth=str(a+b)),extra_info=dict(index=i)) for i,(a,b) in enumerate(pairs)]
train=record['train_pairs'][:32];validation=record['val_pairs'][:8]
assert not {tuple(sorted(p)) for p in train}&{tuple(sorted(p)) for p in validation}
pd.DataFrame(make_rows(train)).to_parquet(out/'train.parquet')
pd.DataFrame(make_rows(validation)).to_parquet(out/'val.parquet')
reward_source=chr(10).join(['def compute_score(data_source, solution_str, ground_truth, extra_info=None):','    return float(solution_str.strip() == str(ground_truth))',''])
(out/'reward.py').write_text(reward_source,encoding='utf-8')
spec=importlib.util.spec_from_file_location('tutorial_reward',out/'reward.py')
reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
assert reward.compute_score('tutorial_addition','10','10')==1
assert reward.compute_score('tutorial_addition','100','10')==0
assert reward.compute_score('tutorial_addition','10 or 12','10')==0
print('PASS：Parquet 导出、划分与奖励函数测试通过。')''')
md('''## 3 · 用当前 verl 数据集类读取

导入 verl 的 `RLHFDataset` 和 `DataProto`，检查数据批次。当前安装版本把 prompt 的最终分词放在 rollout AgentLoop 中，所以 dataset 返回 `raw_prompt`，不要误以为每个版本都直接返回 token 张量。

还会用导出的 tokenizer 和模型做一次 GPU 生成，确认接口能连接。单次答案是否正确与接口是否可运行是两件事。''')
code('''from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
from verl import DataProto
cfg=OmegaConf.create(dict(max_prompt_length=32,filter_overlong_prompts=True,filter_overlong_prompts_workers=1,cache_dir=str(out/'cache')))
dataset=RLHFDataset(str(out/'train.parquet'),loaded,cfg)
batch=DataProto.from_single_dict(collate_fn([dataset[i] for i in range(4)]))
assert len(dataset)==32 and len(batch)==4
print('真实 verl 数据批次:',len(batch),'条；字段:',list(dataset[0]))
prompt=dataset[0]['raw_prompt']
tokens=loaded.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_tensors='pt').cuda()
model=model.cuda().eval()
with torch.inference_mode():
    response=model.generate(tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=4,do_sample=False,
        pad_token_id=loaded.pad_token_id,eos_token_id=loaded.eos_token_id)
answer=loaded.decode(response[0,tokens.shape[1]:],skip_special_tokens=True)
truth=dataset[0]['reward_model']['ground_truth']
print('题目:',prompt,'答案:',repr(answer),'奖励:',reward.compute_score('tutorial_addition',answer,truth))
del model;torch.cuda.empty_cache()''')
md('''## 4 · 加载真实训练配置，但不启动训练

使用 `--cfg job` 只生成配置，验证参数名能被当前 verl 接受。模板设为单 GPU、短序列、小批量，本课导出的模型与这份配置尚未验证完整训练，因此去掉 `--cfg job` 后仍需单独排查资源和运行问题。第 11 课提供的是另一套经过验证的微型训练配置。

actor 是更新策略的模型，rollout 是生成过程，ref 是 KL 参考策略；GRPO 的组内优势不需要 PPO 那样的独立 value critic。本课不启用 actor KL，因此没有验证 ref 工作进程。

第 11 课会先完成一次更新，检查显存、参数变化和检查点；第 12 课再加入任务奖励与固定评估。''')
code('''overrides=[
    'algorithm.adv_estimator=grpo',f'data.train_files={out}/train.parquet',f'data.val_files={out}/val.parquet',
    'data.train_batch_size=4','data.max_prompt_length=16','data.max_response_length=4',
    f'actor_rollout_ref.model.path={model_dir}','actor_rollout_ref.rollout.name=vllm',
    'actor_rollout_ref.rollout.tensor_model_parallel_size=1','actor_rollout_ref.rollout.n=4',
    'actor_rollout_ref.rollout.max_model_len=32','actor_rollout_ref.rollout.gpu_memory_utilization=0.2',
    'actor_rollout_ref.actor.ppo_mini_batch_size=4','actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
    'actor_rollout_ref.actor.use_kl_loss=false','trainer.n_gpus_per_node=1','trainer.nnodes=1',
    'trainer.total_training_steps=1','trainer.logger=[console]',
    f'custom_reward_function.path={out}/reward.py','custom_reward_function.name=compute_score']
command=[sys.executable,'-m','verl.trainer.main_ppo','--cfg','job']+overrides
result=subprocess.run(command,text=True,capture_output=True,timeout=120)
(out/'configuration_check.log').write_text(result.stdout+result.stderr,encoding='utf-8')
assert result.returncode==0,result.stderr
(out/'config_arguments.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
print('PASS：真实 verl 配置加载成功，未启动训练。')
print('完整输出:',out/'configuration_check.log')''')
md('''## 5 · 基础结业检查

运行结果检查能证明文件和接口存在；下面的理解题需要你自己回答，不等于系统替你通过了理解考核。

1. GPU 可用为什么不等于所有代码自动加速？
2. SFT 的监督信号与 GRPO 的奖励/优势有什么区别？
3. 一组回答全对/全错时，标准化组内优势有什么性质？
4. 为何要按数对而不是按文本行划分训练和测试？
5. 为什么只恢复模型权重不保证准确续训？
6. 第七课长加法成功中，哪些规则由程序员提供？
7. 为什么 `--cfg job` 成功不能证明训练成功？

<details><summary>参考答案（先自行回答）</summary>

1. 张量和模型要在 CUDA 设备上执行，数据传输也有开销。
2. SFT 用标准答案 token 的交叉熵；GRPO 对采样回答打分，比较组内相对奖励后更新策略。
3. 奖励相同则优势为零，奖励项没有区分方向。
4. 交换加数和格式变体可能泄漏同一道题的信息。
5. 优化器、随机状态、步数等也影响后续更新。
6. 位对齐、倒序循环、进位传递与类别解码均为人工提供。
7. 配置解析不执行 Ray/vLLM 初始化、生成、反向传播或检查点保存。

</details>

如果你已完成实验并能解释上述问题，就可以进入高级课程，进一步验证 verl 的完整训练流程。基础部分练习的 GPU 计算、训练信号、评估和状态恢复，会继续用于后面的排错。完整分布式训练、LoRA、大模型部署和多 GPU 优化属于后续进阶部分。''')
code('''required={
    'GPU experiment':'01_verl_gpu_beginner.ipynb','SFT':'support/experiment02_output/results.json',
    'RL':'support/experiment03_output/results.json','controls':'support/experiment04_output/results.json',
    'evaluation':'support/experiment05_output/evaluation.json','coverage':'support/experiment06_output/results.json',
    'composition':'support/experiment07_output/results.json','reward audit':'support/experiment08_output/reward_audit.csv',
    'resume':'support/experiment09_output/verification.json'}
checks={key:Path(value).exists() for key,value in required.items()}
checks.update(tokenizer=(model_dir/'tokenizer.json').exists(),parquet=len(dataset)==32,config=result.returncode==0)
display(pd.DataFrame(checks.items(),columns=['artifact_or_check','passed']))
(out/'foundation_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
assert all(checks.values()),'有前置课程产物缺失，请先完成对应课程'
print('基础实验产物与本课接口检查全部通过；理解题请自行作答。')''')
save('10_verl_bridge_and_graduation.ipynb')

md('''# GPU 与 verl 基础课程 · 首页

本课程通过可运行的 Notebook 讲解 GPU 计算与 verl 训练。实验使用随机初始化的小模型，无需下载大模型权重。参考环境为 WSL、Python (verl GPU) 内核和 RTX 4060 8GB；其他设备的运行时间和资源占用可能不同。

**使用方法：** 点击课程链接，选择 Python (verl GPU)，从上到下按 Shift+Enter。仓库中的 Notebook 已清空输出，请按顺序运行以生成自己的结果；交互按钮需要先执行初始化代码。不要同时训练多个 Notebook；完成后用 Kernel → Shut Down Kernel 释放显存，关闭标签不一定关闭内核。

## 学习路线

| 课次 | 内容 | 学会什么 |
|---|---|---|
| [01](01_verl_gpu_beginner.ipynb) | GPU 与奖励学习入门 | 设备、计时、梯度、奖励 |
| [02](02_tiny_language_model.ipynb) | 微型 Qwen2 SFT | token、标签、生成、评估 |
| [03](03_group_reward_learning.ipynb) | GRPO 风格教学循环 | 组采样、优势、裁剪、KL |
| [04](04_visible_learning_comparison.ipynb) | 三组对照 | 冻结/SFT/RL、公平比较的边界 |
| [05](05_model_playground.ipynb) | 交互评估台 | 分布变化与能力边界 |
| [06](06_data_coverage.ipynb) | 数据覆盖 | 分组隔离、验证与最终测试 |
| [07](07_column_reasoning.ipynb) | 逐位组合 | 人工算法先验与组合泛化 |
| [08](08_sampling_and_rewards.ipynb) | 采样和奖励漏洞 | 探索、多样性、奖励测试 |
| [09](09_checkpoints_and_reproducibility.ipynb) | 保存和恢复 | 权重、优化器与随机状态 |
| [10](10_verl_bridge_and_graduation.ipynb) | 真实 verl 接口与结业 | tokenizer、Parquet、奖励、配置 |

建议首次按顺序学习。已有 01 至 07 输出时，可以直接学习 08 至 10。每课通常阅读 10 至 25 分钟，计算时间以实际日志为准。

## 依赖与文件

- 02 → 03；04 建立新起点并保留旧结果。
- 04 → 05 → 06 → 07；08 使用 04；09 独立；10 使用 06 并检查课程产物。
- 各课输出在 `support/experimentXX_output/`，重跑保存格会覆盖对应课输出。
- `build_*.py` 是作者生成脚本，不是日常学习入口，运行会覆盖对应 Notebook 的修改和输出。
- 在 verl 项目根目录运行 `jupyter lab --notebook-dir=tutorial`，打开终端打印的本机链接。安装步骤见仓库 README。

## 常见问题

缺少变量：从初始化格重跑；错误内核：选 Python (verl GPU)；显存不足：关闭其他内核；旧警告：重启内核后重跑；按钮无响应：执行面板之前的代码格。进度条或按钮无法显示时，检查当前环境是否已安装 ipywidgets，并重启内核。

## 完成基础部分的标准

你应能独立解释 SFT 与奖励学习的差别、避免训练测试泄漏、判断真实改善、复现 checkpoint 恢复，并在第 10 课通过真实数据加载和配置检查。

**课程范围：** 第 03/04 课是 PyTorch 教学版 GRPO 风格实现；第 07 课是局部网络与显式竖式程序；第 10 课调用真实 verl 数据类和配置入口，但没有执行完整 Ray/vLLM 训练。完成基础课程不等同于已验证完整大模型 RL 训练。进阶阶段再做完整单步训练、显存优化和部署。''')
md('''## 高级课程

从第 11 课开始验证真实 verl 训练，再把失败诊断推进到梯度、预热、评估和资源测量。

| 课程 | 核心问题 | 运行依赖 |
|---|---|---|
| [11：单步 GRPO](11_real_verl_training.ipynb) | 参数更新和检查点是否真实发生？ | 先运行配套脚本，或开启训练开关 |
| [12：稀疏奖励诊断](12_sparse_reward_diagnosis.ipynb) | 流程完成后，任务为何仍失败？ | 先运行 support/advanced12/run.sh |
| [13：目标函数与梯度](13_objectives_and_gradients.ipynb) | 公式、mask 与梯度是否一致？ | CPU 即可，无前课权重依赖 |
| [14：SFT 预热与 verl](14_sft_warmup_and_verl.ipynb) | 预热能否改善协议与任务表现？ | CUDA；真实 GRPO 是单独启动的可选步骤 |
| [15：评估不确定性](15_evaluation_and_uncertainty.ipynb) | 改善有多大，证据有多强？ | 第 14 课逐题预测，CPU 即可 |
| [16：显存与吞吐](16_memory_and_throughput.ipynb) | 资源花在哪里，怎样正确计时？ | CUDA，无前课权重依赖 |
| [17：预训练模型 LoRA](17_pretrained_lora.ipynb) | 模板、掩码、适配器重载与合并是否可靠？ | CUDA、PEFT；首次下载约 1GB 权重 |
| [18：本地流式服务](18_streaming_service_benchmark.ipynb) | 服务输出与并发指标是否可信？ | 第 17 课合并模型、vLLM；仅监听本机 |

第 11 课使用合成奖励验证工程链路。第 12 课的历史结果是 10 步中 3 步非零梯度，贪心测试准确率未改善。第 14 课更换词表和数据范围，属于独立实验，不与第 12 课直接作同条件因果比较。

首次克隆后没有本地模型、日志和检查点。请先执行相应训练；公开 Notebook 已清空输出，小型 JSON/CSV 是历史运行记录。每次训练以本轮配置与输出为准。

学习要求、缺口审阅和后续全量微调对照、多卡及生产负载项目见 [工程学习路线与考核](support/docs/ENGINEERING_ROADMAP.md)。
''')
save('00_START_HERE.ipynb')
