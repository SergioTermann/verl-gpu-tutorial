import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
md('''# 高级 11 · 真正运行 verl 的一轮 GRPO 训练

从这一课开始进入真实 verl 工程链路：**Ray 协调 → vLLM 生成 → 自定义奖励 → 组内优势 → FSDP actor 更新 → 检查点保存**。不再用自己写的 PyTorch 循环代替 verl trainer。

本课使用约 125 万参数的随机 Qwen2、4 个合成提示，每个提示生成 2 个回答。奖励是输出 UTF-8 字节和取模后的分数，**只用于产生不同奖励以测试工程链路，不衡量回答质量，也没有正确性/泛化提升结论**。

本机 8GB GPU 已实测。初始化开销大于实际更新，整次运行可能需要 1–3 分钟。先关闭其他训练内核，避免 GPU 和 WSL 主机内存争用。执行训练格前阅读开关说明。

## 1 · 理解进程与内存

Ray 负责协调工作进程；vLLM 负责 rollout；FSDP actor 负责梯度更新。单 GPU 时 FSDP 会使用 NO_SHARD，这是正常行为，不意味着没有训练。

**显存和主机内存是两回事。** 初次验证使用默认 8 个奖励 worker 时出现 Ray 主机内存压力；本课将 `reward.num_workers=1`，保留内存监控，不通过关闭保护绕过问题。

不启动独立 critic，因为 GRPO 的优势来自组内比较；actor KL loss 关闭，所以本次不验证 reference-policy 链路。''')
code('''import sys,json,os,subprocess,time,signal
from pathlib import Path
import pandas as pd
import torch
assert sys.executable.startswith('/opt/verl-venv/')
assert torch.cuda.is_available()
project=Path.cwd().parent
scripts=Path('advanced11').resolve()
print('GPU:',torch.cuda.get_device_name(0))
print('训练入口:',project/'verl/trainer/main_ppo.py')
print('完整脚本：')
print((scripts/'run.sh').read_text())''')
md('''## 2 · 数据、奖励与关键配置

`prepare.py` 从固定种子创建模型和词表，train/val 提示不同；本课不运行验证评估。每次训练建立独立 `/opt/verl-advanced11/runs/时间-进程号`，防止把旧检查点误认为新结果。WSL Linux 文件系统存模型，Notebook 存在 Windows 工作区。

| 设置 | 本课值 | 目的 |
|---|---|---|
| train batch / rollout.n | 4 / 2 | 共生成 8 个回答 |
| micro batch per GPU | 1 | 控制单次更新显存 |
| tensor parallel | 1 | 单 GPU |
| vLLM memory utilization | 0.2 | 限制生成后端预算，不代表整卡总占用 |
| reward workers | 1 | 降低主机内存压力 |
| total steps / save frequency | 1 / 1 | 更新一次就保存 |
| enforce eager | true | 避免 CUDA graph 的额外资源开销 |

这些设置仅证明本课微型模型可运行，不能直接推断 0.5B 或更大模型也放得下。''')
code('''print((scripts/'prepare.py').read_text())
print('奖励函数（工程测试用途）：')
print((scripts/'reward.py').read_text())''')
md('''## 3 · 启动训练并实时读取日志

为了打开教程就能检查已完成的实测，`RUN_TRAINING=False` 默认读取最近一轮日志，**不是新训练**。改成 True 后运行本格，会真正启动新训练；程序为每轮单独保存模型、日志和 checkpoint。

Notebook 只显示关键事件和指标，完整原始日志仍保存。中断本格会终止本次启动的进程组；不要同时运行多个副本。超时为 10 分钟，异常退出会明确报错。不会停止用户另行启动的 Ray 作业。

重跑只读模式不会改动权重；重跑训练模式会从随机初始化重新开始，而非续训。''')
code('''RUN_TRAINING=False
if RUN_TRAINING:
    process=subprocess.Popen(['bash',str(scripts/'run.sh')],cwd=project,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    import threading
    def terminate():
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    timer=threading.Timer(600,terminate);timer.start()
    try:
        for line in process.stdout:
            if any(word in line for word in ['Prepared','Started a local Ray','vLLMHttpServer','step:1','checkpoint','Error','Traceback','memory pressure']):
                print(line,end='',flush=True)
        assert process.wait()==0,'训练未正常结束，请查看完整 train.log'
    finally:
        timer.cancel();terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
run_root=Path((scripts/'latest_run.txt').read_text().strip())
assert (run_root/'train.log').exists()
print('本格模式：', '刚完成新训练' if RUN_TRAINING else '读取已完成实测，不重新训练')
print('运行目录:',run_root)
log=(run_root/'train.log').read_text()
for line in log.splitlines():
    if 'training/global_step:1' in line:print(line)
print('完整日志:',run_root/'train.log')''')
md('''## 4 · 检查证据：不能只看 loss 或退出码

本课检查真实 FSDP 权重与初始模型的最大绝对差、发生变化的张量数量、非零梯度范数、优化器状态和保存步数。

一次组内标准化更新的 `actor/pg_loss` 可能恰好为 0，但梯度并不一定为 0。不能只凭标量 loss 判断“没有学习”；权重变化才是直接证据之一。

注意 `perf/max_memory_allocated_gb` 是训练进程报告的 PyTorch 峰值，不是整卡全进程占用。耗时指标也不包括所有启动开销。''')
code('''import importlib.util
spec=importlib.util.spec_from_file_location('advanced11_verify',scripts/'verify.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.verify(run_root)
print('变化张量:',result['changed_tensors'],'/',result['total_tensors'])
print('最大权重差:',result['max_abs_weight_change'])
print('Ray 内存压力记录:',result['ray_memory_pressure'])
keys=['actor/grad_norm','actor/pg_loss','critic/rewards/mean','timing_s/gen','timing_s/update_actor','timing_s/save_checkpoint','perf/max_memory_allocated_gb']
display(pd.DataFrame([(key,result['metrics'].get(key)) for key in keys],columns=['metric','value']))
display(pd.DataFrame(result['checkpoint_files'],columns=['saved_file']))
assert not result['ray_memory_pressure'],'本轮仍存在主机内存压力，请先释放资源再运行'
print('PASS：真实生成/奖励/更新日志、权重变化和完整检查点均通过检查。')''')
md('''## 5 · 读取训练状态与下一步

只加载自己生成的可信 checkpoint。FSDP 分片包含模型和优化器状态，并不自动等价于完整 Hugging Face 导出；多 GPU 时还涉及分片合并。本课是 world_size=1。

练习：先记录本轮配置和结果，再单独尝试 `rollout.n=4`，观察生成 token 数与资源变化。不要直接增加模型、批量和序列长度三个因素。

下一课可在这条已验证的链路上引入有任务含义的奖励与固定评估，并验证训练恢复。当前实验只证明工程流程跑通，不证明训练提高了问答能力。''')
code('''summary_dir=Path('experiment11_output');summary_dir.mkdir(exist_ok=True)
(summary_dir/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(summary_dir/'run_script.sh').write_text((scripts/'run.sh').read_text(),encoding='utf-8')
print('Notebook 侧报告:',summary_dir.resolve())
print('原始训练日志与完整检查点:',run_root)
print('高级第一课完成：真实 verl 单步 GRPO 工程验证。')''')
for i,c in enumerate(cells):c['id']=f'advanced11-{i:02d}'
meta=dict(kernelspec=dict(display_name='Python (verl GPU)',language='python',name='verl-gpu'),language_info=dict(name='python',version='3.12'))
(ROOT/'11_real_verl_training.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=meta),ensure_ascii=False,indent=2),encoding='utf-8')
