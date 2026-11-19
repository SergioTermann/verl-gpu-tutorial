"""Source for the executable Chinese beginner notebook."""
import json
import statistics
import sys
import time


def emit(**values):
    print('@metric ' + json.dumps(values), flush=True)


def environment():
    import importlib.metadata as m
    import torch
    import verl
    for package in ('verl', 'torch', 'vllm', 'flash-attn'):
        print(f'{package}: {m.version(package)}', flush=True)
    print(f'Python: {sys.executable}\nverl: {verl.__file__}', flush=True)
    assert torch.cuda.is_available(), 'CUDA 不可用，请通过 start-notebook.ps1 启动 WSL 环境'
    print(f'GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}', flush=True)
    from flash_attn import flash_attn_func
    q = torch.randn(1, 32, 2, 64, device='cuda', dtype=torch.float16)
    assert torch.isfinite(flash_attn_func(q, q, q)).all()
    print('PASS：真实 CUDA 运算与 FlashAttention 均通过。', flush=True)


def benchmark():
    import torch
    torch.manual_seed(42)
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    times = {}
    reference = None
    print('FP32 / 2048×2048 / 预热 3 次 / 测量 10 次 / 不计数据传输', flush=True)
    for device in ('cpu', 'cuda'):
        x, y = a.to(device), b.to(device)
        for _ in range(3):
            z = x @ y
        if device == 'cuda':
            torch.cuda.synchronize()
        samples = []
        for i in range(10):
            start = time.perf_counter()
            z = x @ y
            if device == 'cuda':
                torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
            print(f'{device:4s} 第 {i+1:02d} 次：{samples[-1]:.3f} ms', flush=True)
        times[device] = statistics.median(samples)
        if device == 'cpu':
            reference = z
        else:
            torch.testing.assert_close(z.cpu(), reference, rtol=1e-3, atol=1e-3)
    print(f"中位耗时：CPU {times['cpu']:.2f} ms / GPU {times['cuda']:.2f} ms", flush=True)
    print(f"本次算子加速比：{times['cpu']/times['cuda']:.2f}×；不代表完整训练加速比。", flush=True)
    emit(cpu_ms=times['cpu'], gpu_ms=times['cuda'])


def reward():
    import torch
    from verl import DataProto
    batch = DataProto.from_dict(tensors={'input_ids': torch.tensor([[1, 2], [3, 4]])})
    print(f'真实 verl.DataProto 批次：{len(batch)} 条数据', flush=True)
    answers = ['4', '5', '4', '3']
    scores = torch.tensor([float(x == '4') for x in answers])
    advantages = (scores - scores.mean()) / (scores.std() + 1e-6)
    for answer, score, advantage in zip(answers, scores, advantages):
        print(f'问题 2+2=? | 回答 {answer} | 奖励 {score:.0f} | 组内标准化优势 {advantage:+.3f}', flush=True)
    print('优势 > 0：提高该回答概率；优势 < 0：降低该回答概率。', flush=True)
    print('这是组内优势的简化示意；完整 GRPO 还涉及 token 掩码、策略比率与裁剪等。', flush=True)


def train():
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    logits = torch.nn.Parameter(torch.zeros(4, device='cuda'))
    optimizer = torch.optim.Adam([logits], lr=0.08)
    print('GPU 上的四选一策略：2+2=?，候选答案 [3,4,5,6]。', flush=True)
    print('从 25% 正确概率开始。使用 REINFORCE；这不是 verl/PPO/GRPO 完整训练。', flush=True)
    for step in range(81):
        dist = torch.distributions.Categorical(logits=logits)
        probability = dist.probs[1].item()
        actions = dist.sample((128,))
        rewards = (actions == 1).float()
        if step % 2 == 0:
            emit(step=step, probability=probability, reward=rewards.mean().item())
            print(f'迭代 {step:02d} | 正确答案概率 {probability:.1%} | 本批平均奖励 {rewards.mean().item():.3f}', flush=True)
            time.sleep(0.08)
        if step == 80:
            break
        loss = -(dist.log_prob(actions) * rewards).mean()
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        optimizer.step()
    assert probability > 0.85, '本次策略未达到预期，请检查日志'
    print('PASS：真实采样 → 奖励 → 反向传播 → GPU 参数更新。', flush=True)
    print('这里只学会了一个固定问题，不表示模型获得了通用数学能力。', flush=True)


def config():
    import subprocess
    print('加载当前安装的 verl 训练配置，不启动大模型训练……', flush=True)
    process = subprocess.Popen([sys.executable, '-u', '-m', 'verl.trainer.main_ppo', '--cfg', 'job',
                                'algorithm.adv_estimator=grpo', 'trainer.n_gpus_per_node=1',
                                'trainer.nnodes=1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait() != 0:
            raise RuntimeError('verl 配置加载失败，请查看上方日志')
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()
    print('PASS：配置加载完成；配置成功不代表完整训练已验证。', flush=True)


LESSONS = {'environment': environment, 'benchmark': benchmark, 'reward': reward, 'train': train, 'config': config}

if __name__ == '__main__':
    LESSONS[sys.argv[1]]()
