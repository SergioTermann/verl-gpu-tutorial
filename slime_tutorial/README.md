# slime 独立学习系列

这个系列专门讲 [THUDM/slime](https://github.com/THUDM/slime)，与上层 verl GPU 教程分开。课程固定上游 commit `79aa41a884ab0bf65222c94418f6357d3c31192b`，不把 slime 源码复制进本仓库；运行时从独立 checkout 导入。

slime 的核心是 Megatron 训练、SGLang rollout、Ray 资源编排和数据缓冲/传输路径。它提供自定义生成、奖励、过滤、loss 和训练钩子，适合从可验证奖励到多轮 agentic RL 的扩展。

## 课程

- [S00：系列首页](00_课程导航.ipynb)
- [S01：架构与源码](01_从训练主循环读懂slime架构.ipynb)
- [S02：Sample 与奖励](02_理解样本、奖励与词元掩码.ipynb)
- [S03：分离调试与验收](03_分离调试与训练验收.ipynb)
- [后续路线](ROADMAP.md)
- [验证记录](VALIDATION.md)
- [SGLang GPU 修复与本机启动](sglang_local/README.md)

S01 至 S03 只进行源码、接口和 CPU 夹具测试，不启动完整 Megatron/SGLang 训练。完整训练需要官方兼容的 Linux GPU 环境；官方快速入门当前主要覆盖 H100/H200，不能把本机 RTX 4060 的导入成功解释成训练可行。

## 固定源码

```bash
git clone https://github.com/THUDM/slime.git ~/.cache/slime-course/source
cd ~/.cache/slime-course/source
git checkout 79aa41a884ab0bf65222c94418f6357d3c31192b
export SLIME_SOURCE=$PWD
```

在本教程目录启动 JupyterLab，选择 `Python (verl GPU)` 内核：

```bash
jupyter lab --notebook-dir=slime_tutorial
```

如果源码放在别处，设置 `SLIME_SOURCE`。上游 checkout 必须干净，课程会检查 commit；不要从另一份 slime checkout 混入模块。

## 环境边界

官方 README 提供匹配依赖的 Docker 镜像和 conda/micromamba 路径。推荐先按官方安装，再执行 `pip install -e . --no-deps`。不要在现有 verl 环境中直接升级 CUDA、Megatron 或 SGLang 依赖。记录镜像 digest、驱动、CUDA、Megatron、SGLang、Ray 和模型 revision。

当前课程没有声称在 RTX 4060 上完成 slime 训练。模型权重、数据、Ray 日志和 checkpoint 放在教程目录外；本系列只提交小型源码报告，不提交第三方模型权重。

## 官方资料

- [slime 官方仓库](https://github.com/THUDM/slime)
- [Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)
- [Customization Guide](https://thudm.github.io/slime/get_started/customization.html)
- [Debugging Guide](https://thudm.github.io/slime/developer_guide/debug.html)
- [Reproducibility](https://thudm.github.io/slime/advanced/reproducibility.html)

源码和官方文档属于上游项目，遵循其许可证；本系列新增教程文字和夹具代码遵循本仓库 MIT 协议。执行结果与限制见 [VALIDATION.md](VALIDATION.md)。
