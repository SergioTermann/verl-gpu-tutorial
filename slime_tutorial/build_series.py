"""Build standalone slime lessons. No training or external services are started."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.strip()+'\n'))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s.strip()+'\n',outputs=[],execution_count=None))
def save(name):
    md('''---

<a href="https://github.com/SergioTermann"><img src="assets/logo.png" alt="SergioTermann" width="44" height="44" align="left" style="margin: 0 12px 0 0;"></a>

**更多有趣的 AI 课程**  
[← 访问 SergioTermann 的 GitHub 主页](https://github.com/SergioTermann)''')
    for i,c in enumerate(cells):c['id']=f'slime-{name[:2]}-{i:02d}';c['source']=c['source'].splitlines(keepends=True)
    (ROOT/name).write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(name='verl-gpu',display_name='Python (verl GPU)',language='python'),language_info=dict(name='python',version='3.12'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');cells.clear()

md('''# slime 中文课程导航

从 S01 重新编号，不依赖 verl 教程的模型或实验产物。学习目标是理解 slime 的数据、生成、训练与权重同步契约，能够在合适的独立环境中设计和审查实验。

| 课程 | 内容 | 本机执行范围 |
|---|---|---|
| [S01：架构与源码](01_从训练主循环读懂slime架构.ipynb) | Megatron、SGLang、Ray 与训练入口 | 固定源码读取、依赖检查 |
| [S02：Sample 与奖励](02_理解样本、奖励与词元掩码.ipynb) | 真实数据对象、异步奖励、组与 mask | 官方 Sample 导入、接口与反例测试 |
| [S03：分离调试与验收](03_分离调试与训练验收.ipynb) | rollout-only、train-only、恢复和权重同步 | 源码检查、实验计划与配置审阅 |

先读 [安装与系列说明](README.md)。前三课不启动 Megatron/SGLang 完整训练；使用现有 Python 内核只运行 CPU 可执行内容。完整训练需要独立环境，不能把本系列的 PASS 当作端到端训练成功。

每课先预测结果，再运行代码，最后回答验收题。后续课程计划见 [学习路线](ROADMAP.md)。''')
save('00_课程导航.ipynb')

md('''# 实验 01 · 从训练主循环读懂 slime 架构

slime 将 Megatron 训练与 SGLang rollout 连接起来，允许自定义生成和奖励。Ray 负责进程与远程调用。数据缓冲与传输层连接 prompt、生成样本和训练批次，具体实现随版本演进。

本系列固定官方 commit `79aa41a884ab0bf65222c94418f6357d3c31192b`，不把 main 的后续变化混入课程。先按 README 克隆源码，并设置 SLIME_SOURCE（仅自定义目录需要）。

## 1 · 核对源码，而不是只看框架名称

下格记录关键文件哈希，并检查 train 函数的参数。AST 只解析源码，不导入训练入口、不启动 Ray。''')
code('''import importlib.util,json,platform
from pathlib import Path
import course
root=course.source()
report=course.source_report()
print('固定版本:',report['revision'])
print(course.function_signature(root/'train.py','train'))
print('Python:',platform.python_version())
for package in ['torch','ray','sglang']:
    print(package,importlib.util.find_spec(package) is not None)
course.save_report('01_source.json',report)''')
md('''## 2 · 谁负责什么

| 组件 | 主要职责 | 首先检查的故障 |
|---|---|---|
| Megatron 训练端 | log-prob、梯度、优化器与训练状态 | loss mask、并行配置、梯度与 checkpoint |
| SGLang 生成端 | prompt prefill、采样与生成返回值 | 模板、EOS、截断、采样参数 |
| Ray 与资源编排 | 调度 actor、资源与远程调用 | 资源分配、worker 退出、主机 RAM |
| 数据路径 | prompt、样本分组、buffer、训练批次 | 字段对齐、重复/丢样、数据进度 |
| 权重同步 | 将更新后的 actor 参数交给生成端 | 参数版本、名称映射、是否同步完成 |

已有 verl 经验可以帮助理解职责，但不能直接复制 Hydra 参数或 DataProto 字段。slime 的自定义奖励接收 args 和 Sample；完整 rollout 接口与逐样本生成接口也不同。

## 3 · 阅读实际调用顺序

下面只显示训练循环相关行，省略的上下文仍应在官方 train.py 中阅读。先找到首次权重同步，再找 generate、async_train 和下一次权重同步。

“训练完成”是运行时事件；数据进度与模型/优化器是否共同持久化，还要检查 checkpoint 与恢复协议。异步 producer 可能跨越更新边界，不能简单把提交顺序当作每条样本的实际权重版本。''')
code('''text=(root/'train.py').read_text(encoding='utf-8')
markers=['update_weights','training_ready','generate.remote','async_train','training_completed','save_checkpoint','pause_rollout_admission','resume_rollout_admission']
for i,line in enumerate(text.splitlines(),1):
    if any(key in line for key in markers):print(f'{i:4d} {line.strip()}')
assert all(key in text for key in ['actor_model.update_weights()','rollout_manager.generate.remote','actor_model.async_train'])
print('PASS: 源码中存在预期入口；未运行训练。')''')
md('''## 4 · 环境选择与验收

本次核对的官方快速入门推荐带匹配补丁的 Docker 镜像，GPU CI 主要覆盖 H100/H200，并列出 Blackwell 镜像。当前本机 RTX 4060 8GB、verl 环境没有 SGLang；导入 Sample 成功不能证明 Megatron/SGLang 能共同运行。

完整训练请新建环境，固定源码 commit、镜像 digest、驱动和模型 revision。不要把 `latest` 镜像标签当作可复现版本，也不要在原 verl venv 内直接升级整套 CUDA/推理依赖。

验收：画出一条样本的 prompt → rollout → reward → train → weight sync 路径。说明模型/优化器 checkpoint、rollout 缓冲和 Ray 进程各自保存什么，列出三个无法仅凭退出码判断的错误。''')
save('01_从训练主循环读懂slime架构.ipynb')

md('''# 实验 02 · 理解样本、奖励与词元掩码

本课直接导入固定版本 slime.utils.types.Sample，而非自定义一个同名替身。只使用 CPU；token ID 和 log-prob 为明确标注的人工测试夹具，不是模型生成结果。

单样本奖励接口为 `async def custom_rm(args, sample) -> float`；使用 `--group-rm` 时，应提供接受样本列表、返回等长分数列表的函数。我们的两个入口分别是 `course.strict_reward` 和 `course.batch_reward`。只有课程目录位于所有 worker 的 PYTHONPATH 时，路径才能在远端解析。

## 1 · 奖励是协议，不是字符串包含测试

只接受去除首尾空白、转小写后的 yes/no，必须与 label 完全匹配。错误标签视为数据错误并抛出异常，不能悄悄记零分。此协议不要求 EOS；停止行为需要另外检查。''')
code('''from types import SimpleNamespace
import course
Sample=course.load_sample()
args=SimpleNamespace()
cases=[('yes','yes',1),('YES ','yes',1),('no','yes',0),('yes no','yes',0),('yes because...','yes',0),('','yes',0)]
samples=[Sample(response=answer,label=label) for answer,label,_ in cases]
scores=await course.batch_reward(args,samples)
assert scores==[expected for _,_,expected in cases]
for case,score in zip(cases,scores):print(repr(case[0]),score)
try:
    await course.strict_reward(args,Sample(response='yes',label='invalid'))
except ValueError:print('PASS: 错误标签被显式拒绝。')
else:raise AssertionError('Invalid dataset label accepted')''')
md('''## 2 · response_length 与 loss_mask

tokens 包含提示及响应 token。response_length 指响应部分长度，不是字符数。loss_mask 与响应位置对齐，不能误用全序列长度。工具结果可以位于响应轨迹内，但它不是策略采样的动作，不应直接作为策略损失 token。

下格构造两个 prompt token 和四个响应位置；中间两个位置模拟工具返回，mask 为零。rollout_log_probs 的工具位置使用占位值，只在 mask 下排除。本格只检查课程夹具的对齐，复杂多轮生成需遵循实际接口并保留生成 token 的概率与权重版本。''')
code('''sample=Sample(group_index=0,index=0,rollout_id=0,prompt='Synthetic prompt',
    tokens=[10,11,20,30,31,21],response='Synthetic trajectory',response_length=4,
    label='yes',loss_mask=[1,0,0,1],rollout_log_probs=[-.2,0.,0.,-.3],
    status=Sample.Status.COMPLETED)
alignment=course.audit(sample)
assert alignment=={'prompt_tokens':2,'response_tokens':4,'loss_tokens':2}
print(alignment)
sample.loss_mask=[1,1]
try:course.audit(sample)
except AssertionError:print('PASS: 错位 mask 被拒绝。')
else:raise AssertionError('Misaligned mask accepted')
sample.loss_mask=[1,0,0,1]''')
md('''## 3 · 同一道题、多个 rollout、多个 segment

group_index 表示用于相对比较的提示组。一个完整 rollout 可能被拆成多个训练 segment；固定版本提供 rollout_id 来标识同一次完整执行。如果每个 segment 都使用新 rollout_id，就可能改变组统计与损失权重。

官方定制文档说明：同一完整轨迹只有一个结果奖励时，各 segment 使用相同奖励并保留 group_index 与 rollout_id；不要自行按 segment 数再除一次奖励。共享生成前缀也不能重复计入损失。实现细节必须与所固定版本一致。

下格仅演示去重后的完整轨迹奖励，不调用 slime 的优势实现。它展示为什么把 segment 当成独立回答会改变平均值。''')
code('''segments=[dict(group=0,rollout=100,reward=1.),dict(group=0,rollout=100,reward=1.),dict(group=0,rollout=101,reward=0.)]
unique={}
for s in segments:
    key=(s['group'],s['rollout'])
    if key in unique:assert unique[key]==s['reward']
    unique[key]=s['reward']
wrong=sum(s['reward'] for s in segments)/len(segments)
right=sum(unique.values())/len(unique)
assert right==.5 and wrong!=right
print('按 segment 误算:',wrong,'按完整轨迹:',right)
course.save_report('02_contracts.json',dict(real_upstream_sample=True,reward_cases=len(cases),
    invalid_label_rejected=True,misaligned_mask_rejected=True,alignment=alignment,
    segment_mean=wrong,rollout_mean=right,scope='CPU interface tests with synthetic tokens; no actual rollout or training.'))''')
md('''## 4 · 课后验收

为截断、空响应、只有工具返回、奖励超时分别定义处理规则。Sample 的 COMPLETED、TRUNCATED、ABORTED、FAILED 不应被一律归为“模型答错”；系统失败率与任务准确率要分开报告。

再尝试把奖励函数改成“包含 yes 就得分”，预测哪条攻击样例会通过。不要把这项修改带入正式评分。下一课用这些契约设计完整训练的调试顺序。''')
save('02_理解样本、奖励与词元掩码.ipynb')

md('''# 实验 03 · 分离调试与训练验收

本课生成可审查的实验计划，并核对官方调试参数。它不输出一个未经验证的“一键训练成功”命令。完整 Megatron+SGLang 作业必须结合模型转换、硬件、并行划分与容器版本确定。

## 1 · 先定位生成侧还是训练侧

官方支持 `--debug-rollout-only`、`--debug-train-only`，以及保存/重放 rollout 数据。使用固定 rollout 可以减少采样变化，帮助比较训练实现；只完成 rollout 不代表优化器和权重同步正确。

保存/加载路径支持 `{rollout_id}` 格式。读取 debug 的 PyTorch 文件仅限自己生成的可信产物。''')
code('''import json
import course
root=course.source()
doc=(root/'docs/en/developer_guide/debug.md').read_text(encoding='utf-8')
flags=['--debug-rollout-only','--debug-train-only','--save-debug-rollout-data','--load-debug-rollout-data','--save-debug-train-data']
for flag in flags:
    assert flag in doc
    print(flag)
print('PASS: 固定版本官方文档包含这些参数；未执行 CLI 解析或训练。')''')
md('''## 2 · 三阶段调试计划

| 阶段 | 必须观察 | 不足以证明什么 |
|---|---|---|
| rollout-only | 模板、token、EOS、奖励、失败率 | 反向传播正确 |
| 固定数据 train-only | mask、log-prob、优势、梯度、参数变化 | 训练权重已发给生成服务 |
| 完整至少两轮 | 参数同步、后续样本版本、保存和恢复 | 稳定泛化或多种子收益 |

至少两轮是为了检查训练更新后的生成。第一次 rollout 可能仍是初始权重；只看第一轮 reward 无法证明同步生效。若使用异步采样，需要同时记录权重版本、样本滞后与 off-policy 修正规则。

不要把 actor log-prob、rollout log-prob、reference log-prob 视为同一对象。采样温度、top-p、tokenization、数值精度与权重版本都可能造成差异，先核对定义再判断数值是否应该相等。''')
code('''plan=dict(source_revision=course.REVISION,
    hardware={'current_reference':'RTX 4060 Laptop 8GB','full_training_verified':False},
    environment={'image_digest':None,'driver':None,'sglang_version':None,'megatron_revision':None},
    task={'name':'strict_yes_no','reward_entry':'course.strict_reward','batch_reward_entry':'course.batch_reward',
          'data_split_hash':None,'model_revision':None},
    stages=['rollout-only','fixed-rollout train-only','two-or-more full iterations','checkpoint resume'],
    acceptance=['No silent token/mask mismatch','Report generation errors separately',
                'Verify actor parameter change','Verify synchronized generation weights',
                'Recover optimizer and data progress','Evaluate held-out prompts'])
course.save_report('03_run_plan.json',plan)
print(json.dumps(plan,indent=2))
missing=[key for key,value in plan['environment'].items() if value is None]
assert missing
print('尚未填写的环境条件:',missing,'；该计划不可当作已验证启动配置。')''')
md('''## 3 · 参数迁移与资源核对

Megatron 参数直接进入其训练配置；SGLang 参数通常增加 `--sglang-` 前缀，例如内存参数 `--sglang-mem-fraction-static`。这不是说任意新版引擎参数都能在旧镜像里使用，必须检查实际版本的参数解析。

总 batch 需要区分提示数、每提示采样数、序列数、micro batch、数据并行与累积。Tensor/Pipeline/Context Parallel 改变模型与序列的切分，不能简单都乘成训练样本数。共享 GPU 的训练/生成切换，还要考虑权重、KV cache、offload 和主机 RAM。

课程不建议在 8GB 设备直接照搬数据中心卡的模型和配方。更小模型或 offload 能否成功，需要独立实测，不能仅由参数量推断。

## 4 · 本系列下一步

在独立且兼容的 GPU 环境确定后，依次完成官方最小配方、真实自定义奖励、多轮工具轨迹、异步采样与恢复实验。每一节都保留配置、失败日志、逐样本输出和验收报告。

验收作业：给出你准备运行的镜像 digest、源码 commit、模型 revision、硬件拓扑和数据哈希；说明出错时如何切到固定 rollout 重放。仅打印配置或创建 Sample 不计作完成真实训练。''')
save('03_分离调试与训练验收.ipynb')
