"""Generate lessons 17 and 18 without executing or bundling model weights."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
cells=[]


def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s.strip()+'\n'))
def code(s): cells.append(dict(cell_type='code', metadata={}, source=s.strip()+'\n', execution_count=None, outputs=[]))


def save(name):
    for i,c in enumerate(cells):
        c['id']=f'{name[:2]}-{i:02d}';c['source']=c['source'].splitlines(keepends=True)
    (ROOT/name).write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(kernelspec=dict(name='verl-gpu',display_name='Python (verl GPU)',language='python'),
                      language_info=dict(name='python',version='3.12'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cells.clear()


md(r'''# 第 17 课 · 预训练模型上的 LoRA 与模型交付

前面的模型从随机参数学习小任务。本课使用固定版本的 Qwen2.5-0.5B-Instruct，训练一个合成工单路由器。目标是掌握预训练模型的输入协议、回答掩码、参数高效微调及模型交付验证，准确率是否提高要看实际结果。

需要 Linux/WSL、CUDA、transformers、PEFT、pandas 和 huggingface_hub。首次运行下载约 1GB 权重，完整模型与合并模型另需数 GB 磁盘。默认使用 BF16，参考设备为 8GB 显卡；先关闭其他 GPU 内核。

模型来自 [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，固定 revision 为 `7ae557604adf67be50417f59c2c2f167def9a775`。模型权重单独下载，不纳入教程 MIT 授权或 Git 提交，使用时遵循模型自身许可。

## 1 · LoRA 改变了什么

对冻结线性层 $W\in\mathbb R^{d_{out}\times d_{in}}$，LoRA 使用

$$W'=W+\frac{\alpha}{r}BA,\quad A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r}.$$

每层新增 $r(d_{in}+d_{out})$ 个可训练参数。常见初始化让 B 为零，A 随机；第一步 A 的梯度可能为零，这不等于实现失败。本课只适配 attention 的 q_proj、v_proj，rank=8、alpha=16，dropout=0。

这是未量化的 LoRA。BF16 基础权重仍需常驻，激活和 KV cache 仍消耗显存。QLoRA、全量微调和更大模型不在本课的实测范围。''')
code('''import importlib.util, json, os, time
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
spec=importlib.util.spec_from_file_location('lesson17',Path('support/advanced17/experiment.py'))
lesson=importlib.util.module_from_spec(spec);spec.loader.exec_module(lesson)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '本实验需要支持 BF16 的 CUDA GPU'
model_path=lesson.resolve_model()
print('模型来源:', lesson.MODEL_ID, 'revision:',lesson.REVISION)
print('本地权重:',model_path)
splits=lesson.dataset()
print({name:len(rows) for name,rows in splits.items()})
print('模板组数:',{name:len({r['group'] for r in rows}) for name,rows in splits.items()})
display(pd.DataFrame(splits['train']).head())''')
md('''## 2 · 先审查数据和标签，再训练

训练集有 18 个模板组、72 条工单；验证与测试各有 6 个模板组、24 条工单。每组的四条记录只改变编号，必须在同一划分内。它们不是四个独立语义样本，不能按 24 个独立问题估计泛化误差。

任务固定映射：账单 R0、账号访问 R1、物流 R2，输出严格为 `route=R0/R1/R2`。编号跨标签复用，不能单靠编号推断标签。数据是人工合成的英语短句，未覆盖多意图、模糊工单、业务变化或真实用户语言。

完整 chat template 包含 system、user、assistant 边界。我们验证完整对话的 token 前缀与生成提示完全一致，只对 assistant 答案和第一个 EOS 计算损失。padding 标签为 -100；超过 160 token 就报错，不静默截断答案。因果 shift 由模型内部完成。''')
code('''from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
row=splits['train'][0]
ids,labels=lesson.encode(tokenizer,row)
print('完整输入:',tokenizer.decode(ids))
print('参与损失的 token:',tokenizer.decode([x for x in labels if x!=-100]))
print('输入长度:',len(ids),'有效标签:',sum(x!=-100 for x in labels))
assert len(ids)==len(labels) and labels[-1]==tokenizer.eos_token_id
for name,rows in splits.items():
    for item in rows: lesson.encode(tokenizer,item)
print('PASS: 全部样本的模板前缀、回答掩码和长度检查通过。')''')
md('''## 3 · 先做小批量拟合检查，再重置训练

配套训练函数先对三条训练样本做 12 次更新，检查损失是否下降。这个检查只能发现明显的训练链路问题，不保证单批已完全过拟合。之后恢复全部 LoRA 初始张量，重新创建优化器，避免把诊断更新偷偷带入正式实验。

正式训练固定 60 次更新：micro batch=2、累积 4 次，有效 batch=8。不同微批按有效回答 token 数加权，使梯度对应整个累积批的 token 平均损失。每 10 步只查看验证集，固定最后一步用于测试。

函数还检查优化器只更新 LoRA 参数，并对三个冻结基础张量做训练前后哈希比较。哈希是抽查，不是整个基础模型的逐字节校验。完整实现见 [experiment.py](support/advanced17/experiment.py)。

首次学习保持 `RUN_TRAINING=True`。重跑本格会新建目录并重新训练；仅想读取本地上次实验时改为 False。训练中曲线会持续更新。''')
code('''RUN_TRAINING=True
out=Path('support/experiment17_output');out.mkdir(exist_ok=True)
def live(history):
    clear_output(wait=True)
    frame=pd.DataFrame(history)
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].plot(frame.step,frame.loss);axes[0].set(title='Answer-token training loss',xlabel='Update')
    axes[1].plot(frame.step,frame.validation_accuracy);axes[1].set(title='Validation accuracy',xlabel='Update',ylim=(-.05,1.05))
    plt.tight_layout();plt.show();plt.close(fig)
    print(frame.tail(1).to_string(index=False))
if RUN_TRAINING:
    run=Path.home()/'verl-advanced17/runs'/f'{time.strftime("%Y%m%d-%H%M%S")}-{time.time_ns()}'
    manifest,checks=lesson.train(run,callback=live)
    (out/'latest_run.txt').write_text(str(run))  # 成功完成后才写入
else:
    run=Path((out/'latest_run.txt').read_text().strip())
    manifest=json.loads((run/'manifest.json').read_text())
    checks=json.loads((run/'artifact_checks.json').read_text())
print('本次运行:',run)
print(json.dumps(manifest,indent=2))''')
md('''## 4 · 冻结基线、LoRA 与交付一致性

冻结基线在关闭 adapter 后使用相同基础权重评估，不重新抽一个随机模型。测试题在训练完成后才读取。比较所有预测与每类错误，不只报告准确率。

保存后重新加载 base+adapter，在固定验证输入上比较 logits。再分别检查 BF16 直接合并与 FP32 合并：第一次试跑中 BF16 直接合并的相对 L2 误差约为 4.2%，超过原先预设的 0.02 检查线。因此保留该诊断，不放宽阈值，并在 FP32 下核对合并运算。

FP32 比较使用同精度的未合并模型作为参考，阈值固定为 1e-4。通过后导出 FP32 合并权重，并记录相对于 BF16 LoRA 的测试答案分歧。它验证合并实现，不表示改变精度后 logits 必须与 BF16 相同。第 18 课若转换为 BF16 服务，还要重新做答案回归。

`adapter/` 需要原基础模型才能使用；`merged/` 包含合并后的完整模型。适配器文件不是完整训练恢复点，未保存优化器、随机状态和数据位置，不能用来声称无缝续训。''')
code('''predictions=pd.read_csv(run/'predictions.csv',keep_default_na=False)
merged=pd.read_csv(run/'merged_predictions.csv',keep_default_na=False)
display(pd.concat([predictions,merged]).groupby('stage')[['valid','correct','eos']].mean())
display(predictions[predictions.stage=='lora'][['group','text','target','answer','correct']])
display(pd.DataFrame([checks]))
for name in ['manifest.json','artifact_checks.json','history.csv','predictions.csv','merged_predictions.csv','dataset.json']:
    (out/name).write_bytes((run/name).read_bytes())
print('小型报告已保存；权重仍留在独立运行目录。')''')
md('''## 5 · 验收与下一次消融

你应能解释：为什么初始 B=0 时 A 可能没有梯度；为什么训练时右侧 padding、生成时左侧 padding；为什么只保存 adapter 不等于可以续训；为什么测试 24 条记录的有效语义样本数更少。

下一项研究应在新的开发/最终划分上比较 rank、target modules 与学习率，并加入可承受的全量微调对照。当前只验证一个配置、一个种子，不声称 LoRA 优于全量微调。合并模型将在第 18 课通过本地 HTTP 服务再做回归检查。

参考：[LoRA 论文](https://arxiv.org/abs/2106.09685)、[PEFT LoRA 指南](https://huggingface.co/docs/peft/developer_guides/lora)、[聊天模板](https://huggingface.co/docs/transformers/chat_templating)。''')
save('17_pretrained_lora.ipynb')

md('''# 第 18 课 · 本地流式服务与并发评估

模型在 Notebook 里答对，不代表服务交付正确。本课使用第 17 课合并模型，启动本机 vLLM OpenAI-compatible 接口，测量流式响应、并发、错误率与答案回归。无需外部 API 密钥，不上传数据，不开放互联网访问。

服务仅绑定 `127.0.0.1:18018`。端口被占用就停止，不连接或终止其他服务。本课启动的子进程在测试结束或异常时由 finally 清理。若强制杀死整个内核导致清理无法执行，可根据运行目录中的日志定位该进程；不要批量杀死其他 Ray/vLLM 作业。

## 1 · 固定评估协议

使用第 17 课的 24 条合成测试工单，temperature=0、max_tokens=12。加载 FP32 合并产物时显式转换为 BF16 以控制显存；这会引入额外数值差异，所以必须与 FP32 transformers 预测比较。并发为 1、2、4，每档运行三轮，轮换档位顺序；共 216 个计分请求，另有一个 warm-up 请求。重复请求用于测量服务行为，不增加独立语义样本数。

为减少重复前缀缓存造成的混淆，关闭 prefix cache；eager 模式与短上下文用于控制资源。这不是推荐的生产性能配置。模型加载与预热单独记录，不计入稳态吞吐。''')
code('''import importlib.util, json, time
from pathlib import Path
import pandas as pd
from IPython.display import display
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
lesson=module('routing17',Path('support/advanced17/experiment.py'))
serving=module('serving18',Path('support/advanced18/serving.py'))
run=Path(Path('support/experiment17_output/latest_run.txt').read_text().strip())
assert (run/'merged/config.json').exists(), '请先完成第 17 课的模型导出'
splits=json.loads((run/'dataset.json').read_text())
rows=splits['test']
out=Path('support/experiment18_output');out.mkdir(exist_ok=True)
server_logs=run/f'serving-{time.time_ns()}'
print('服务模型:',run/'merged')
print('计分请求数:',len(rows)*3*3)''')
md('''## 2 · 从流式 HTTP 响应测量延迟

TTFT 从客户端开始 HTTP 请求到收到第一个非空文本片段。空的角色声明不算首 token。端到端延迟到完整响应结束。每个请求保存 token usage、finish_reason、答案、成功状态和错误信息；失败请求留在分母中。

SSE 的一个 chunk 不一定等于一个 token。我们单独记录 chunk 数量与相邻 chunk 间隔，并用服务报告的 completion_tokens 估算 TPOT。由于 EOS 可能计入 token 数、多个 token 可能合并为一个 chunk，这个 TPOT 只是客户端近似值，不当作精确的逐 token 调度时间。

下面会启动并等待服务，完成压测后关闭。初始化可能需要几分钟，原始日志写入独立 Linux 运行目录。测量期间不要运行其他 GPU 实验。''')
code('''server=serving.LocalServer(run/'merged',server_logs)
started=time.perf_counter()
try:
    server.start()
    startup=time.perf_counter()-started
    print(f'本地服务已就绪，启动耗时 {startup:.1f}s。正在执行并发测试。',flush=True)
    summary=serving.benchmark(server,rows,lesson.messages,out,repeats=3)
    display(summary)
finally:
    server.stop()
    print('本课启动的服务已关闭。',flush=True)
(out/'startup.json').write_text(json.dumps(dict(startup_seconds=startup,server_stopped=True),indent=2))''')
md('''## 3 · 回归检查：服务换了，答案是否也变了

将每次 HTTP 答案与同一合并权重在 transformers 中的答案比较。两种后端的算子与数值路径不同，temperature=0 也不保证逐位一致；发生分歧时应保存原始回答、tokenizer 和配置，检查模板及停止条件，而不是删除异常请求。

这里的“回归集”已经用于第 17 课测试，适合发现部署偏差，不能再宣称是从未查看的新泛化测试。''')
code('''requests_frame=pd.read_csv(out/'requests.csv',keep_default_na=False)
reference=pd.read_csv(run/'merged_predictions.csv',keep_default_na=False)[['id','answer']].rename(columns={'answer':'transformers_answer'})
paired=requests_frame.merge(reference,on='id',how='left',validate='many_to_one')
assert paired.transformers_answer.notna().all()
paired['same_answer']=paired.answer==paired.transformers_answer
report=dict(requests=len(paired),errors=int((~paired.success).sum()),
    answer_disagreements=int((~paired.same_answer).sum()),
    accuracy_including_errors=float(paired.correct.mean()),
    note='Repeated short synthetic prompts; not independent samples or a production SLA.')
(out/'regression.json').write_text(json.dumps(report,indent=2))
paired.to_csv(out/'regression_predictions.csv',index=False)
display(pd.DataFrame([report]))
display(paired[~paired.same_answer][['id','answer','transformers_answer','error']].head(10))''')
code('''import matplotlib.pyplot as plt
summary=pd.read_csv(out/'summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,3))
axes[0].plot(summary.concurrency,summary.output_tokens_per_second,marker='o')
axes[0].set(xlabel='Concurrency',ylabel='Output tokens / second',title='Throughput')
axes[1].plot(summary.concurrency,summary.e2e_p95_ms,marker='o',label='End-to-end p95')
axes[1].plot(summary.concurrency,summary.ttft_p95_ms,marker='o',label='TTFT p95')
axes[1].set(xlabel='Concurrency',ylabel='Milliseconds',title='Latency cost');axes[1].legend()
plt.tight_layout();plt.show();plt.close(fig)''')
md('''## 4 · 怎样读懂 p95 与吞吐

本客户端采用闭环固定并发：最多 C 个 HTTP 请求同时执行，完成后才补下一个请求。每个请求的延迟从工作线程开始发请求计时，不包含它在客户端线程池排队的时间；吞吐的墙钟窗口包含整个批次。

因此，这个实验不能反映固定外部到达率下无限增长的排队延迟。生产压测还需要开环到达模型、长短请求混合、超时、取消、错误注入与持续运行。每档只有 72 个请求，p95 是描述性统计，不是稳定 SLA 保证。

吞吐定义为服务报告的输出 token 总数除以该档各轮测量窗口总时间，不是每请求 token/s 的简单平均。若提高并发后吞吐上升但 p95 变差，需要按业务延迟目标选配置，不能只追求峰值吞吐。

## 5 · 交付检查与扩展作业

把 base revision、adapter 配置、合并模型、tokenizer、chat template、解码参数、后端版本作为一组版本化产物。发布前固定回归集、失败率与回滚阈值；本课只运行本地临时服务，不是公网部署。

下一次实验只改变一个因素：打开 prefix cache、增加输出长度或提高并发。预先记录预期影响，再测质量与延迟。测试多卡、张量并行或量化时，需要重新验证数值与服务行为。

参考：[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)。所有性能结论都应注明硬件、模型、请求分布和测量窗口。''')
save('18_streaming_service_benchmark.ipynb')
