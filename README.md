# verl GPU 中文交互教程

12 节 Jupyter 课程，涵盖 GPU 入门、SFT、GRPO 风格原理、模型评估、训练复现及真实 verl 单卡训练与失败诊断。

[课程首页](tutorial/00_START_HERE.ipynb) · [课程目录](tutorial/README.md) · [高级课程](tutorial/ADVANCED.md)

## 运行方式

本仓库是独立教程资源包，不包含 verl 框架源码。以 [官方 verl v0.7.1](https://github.com/volcengine/verl/tree/v0.7.1) 为基础，先按官方说明配置 Linux/WSL2 CUDA 环境。

1. 克隆官方 verl v0.7.1 和本教程仓库。
2. 将本仓库的 `tutorial/` 与 `local_smoke/` 复制到官方 verl 项目根目录。
3. 在已安装 verl 的 Python 环境安装 `jupyterlab ipykernel ipywidgets matplotlib nbclient`。
4. 执行 `python -m ipykernel install --user --name verl-gpu --display-name "Python (verl GPU)"`。
5. 在 verl 根目录执行 `jupyter lab --notebook-dir=tutorial`，打开 `00_START_HERE.ipynb`。

按顺序运行课程，模型权重、数据和 checkpoint 将在本地生成。Notebook 输出已清空，避免携带本机日志与登录信息；`experimentXX_output/` 中的小型 JSON/CSV 是历史实测记录，不是跨设备性能保证。

高级脚本基于 WSL `/opt/verl-venv`，运行前按自己的环境调整激活路径和 `/opt` 输出目录。文档中的 `/path/to/verl` 是路径占位符。高级课程默认读取本地已完成运行，首次使用须先执行 `bash tutorial/advanced11/run.sh` 或 `bash tutorial/advanced12/run.sh`，再执行对应 Notebook。

## 结果边界

- 03/04 是教学版 PyTorch GRPO 风格循环，不等同于 verl trainer。
- 07 使用人工竖式算法组合学习到的局部规则。
- 11 实测完成真实 Ray/vLLM/FSDP 参数更新与 checkpoint 保存。
- 12 实测揭示稀疏奖励冷启动失败，保留负面结果，不宣称准确率改善。
- 训练与测试使用固定小规模数据；单次运行不代表统计显著性或通用能力。

## 参考环境

RTX 4060 Laptop 8GB、WSL2 Ubuntu 24.04、Python 3.12、PyTorch 2.8.0/CUDA 12.8、verl 0.7.1、vLLM 0.11.0、transformers 4.57.1。安装应遵循官方版本兼容要求。

## 许可与来源

以 Apache-2.0 发布，保留 LICENSE 与上游 Notice。教程配套代码基于 volcengine/verl 的接口与示例，不是官方教程，也不代表上游项目背书。
