# RTX 4060 上的 SGLang 修复与验证

已在 WSL Ubuntu 24.04、RTX 4060 Laptop 8GB 上完成 Qwen2.5-0.5B-Instruct 的 GPU 文本生成。使用 SGLang 0.5.3.post3、sgl-kernel 0.3.15、torchao 0.9.0、PyTorch 2.8.0+cu128。

## 修复原因

先前将 `Could not load any common_ops library` 判断为 SM89 不兼容是不准确的。开启 loader 调试日志后，真正的首个错误是缺少 `libnuma.so.1`。该版本加载器本来就会为非 SM90 GPU 选择 `sm100/common_ops` 兼容实现，不能仅凭目录名判断支持的架构。

修复包括安装 Ubuntu 的 `libnuma1`、设置 CUDA_HOME 指向实际包含 `lib/libcudart.so.12` 的目录、补齐 torchao，以及在 Engine 创建之后显式建立 asyncio 事件循环。

本机 `/opt/sglang-venv` 通过 `.pth` 读取 `/opt/verl-venv` 的基础依赖，并在自身目录安装 SGLang 包。这是依赖覆盖环境，不是完全隔离、锁定的环境；改变基础环境后需要重新验证。未在原 verl 环境升级 PyTorch 或 vLLM，也未验证 SGLang 的全部可选功能及完整 slime/Megatron 训练。此版本不等于当前 slime 主线的配套依赖版本。

## 重新验证

在已有兼容环境的 WSL 中，从仓库根目录执行：

```bash
export CUDA_HOME=/opt/verl-venv/lib/python3.12/site-packages/nvidia/cuda_runtime
/opt/sglang-venv/bin/python slime_tutorial/sglang_local/verify_gpu.py \
  --model "$HOME/.cache/verl-course/Qwen2.5-0.5B-Instruct" \
  --output slime_tutorial/results/sglang-verification.json
```

模型需要事先下载；脚本不下载权重。CUDA_HOME 示例仅适用于本机的 PyTorch wheel 依赖布局，其他机器应使用自己的实际 CUDA 路径。验证脚本会关闭它创建的 Engine。

启动仅监听本机的 HTTP 服务：

```bash
export SGLANG_MODEL="$HOME/.cache/verl-course/Qwen2.5-0.5B-Instruct"
bash slime_tutorial/sglang_local/serve.sh
```

Ctrl+C 停止服务。启动脚本采用与已通过 Engine 测试一致的低资源参数；HTTP 服务本身需单独检查健康状态和请求。该配置关闭 CUDA Graph，使用 torch_native attention 与 pytorch sampling，优先验证可运行性，不代表已完成性能优化。

历史测试成功返回了以 `Paris` 开头的 16-token 文本，后续文本包含不准确陈述且因长度截断。因此这里只判定 GPU 生成链路成功，不作为回答质量证据。模型权重与本机环境目录未提交。
