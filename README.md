# Interactive verl GPU Tutorial

**English** | [简体中文](README.zh-CN.md)

The lessons are written in Chinese. Chinese-speaking learners can start with the [中文学习入口](README.zh-CN.md).

A 16-lesson Jupyter course covering GPU basics, supervised fine-tuning, GRPO, reproducibility, real single-GPU verl training, failure diagnosis, gradient checks, evaluation uncertainty, and memory/throughput measurement.

[Course home](tutorial/00_START_HERE.ipynb) · [Lesson index](tutorial/README.md) · [Advanced lessons](tutorial/ADVANCED.md)

For a research-to-engineering learning path, see the Chinese [engineering roadmap and assessment](tutorial/ENGINEERING_ROADMAP.md). Lessons 13–16 add executable exercises and distinguish micro-model evidence from larger-model, multi-GPU, and serving work still to be validated.

## Setup

This repository is a standalone tutorial package; it does not contain the verl framework source. Use it with [verl v0.7.1](https://github.com/volcengine/verl/tree/v0.7.1) and follow the official installation instructions for a Linux or WSL2 CUDA environment.

1. Clone the official verl v0.7.1 repository and this tutorial repository.
2. Copy this repository's `tutorial/` and `local_smoke/` directories into the root of the verl checkout.
3. In the Python environment where verl is installed, install `jupyterlab ipykernel ipywidgets matplotlib nbclient`.
4. Register the kernel:

   ```bash
   python -m ipykernel install --user --name verl-gpu --display-name "Python (verl GPU)"
   ```

5. From the verl root, start JupyterLab:

   ```bash
   jupyter lab --notebook-dir=tutorial
   ```

   Open `00_START_HERE.ipynb` and select the `Python (verl GPU)` kernel.

Run the lessons in order. Model weights, datasets, and checkpoints are generated locally. Notebook outputs in this repository have been cleared to avoid shipping machine-specific logs or credentials. Small JSON/CSV files under `experimentXX_output/` are recorded measurements from the reference environment, not cross-device performance guarantees.

The advanced scripts were developed in WSL with `/opt/verl-venv`; adjust the activation path and output directories for your environment. `/path/to/verl` in the documentation is a placeholder. Before opening the advanced notebooks for the first time, run `bash tutorial/advanced11/run.sh` or `bash tutorial/advanced12/run.sh` as appropriate.

## Scope and limitations

- Lessons 03 and 04 use an educational PyTorch GRPO-style loop; they are not equivalent to the verl trainer.
- Lesson 07 combines a learned local digit rule with a hand-written column-addition algorithm.
- Lesson 11 verifies a real Ray/vLLM/FSDP parameter update and checkpoint save.
- Lesson 12 records a sparse-reward cold-start failure and makes no accuracy-improvement claim.
- The experiments use small fixed datasets and single runs; they do not establish statistical significance or general capability.

## Reference environment

RTX 4060 Laptop 8GB, WSL2 Ubuntu 24.04, Python 3.12, PyTorch 2.8.0/CUDA 12.8, verl 0.7.1, vLLM 0.11.0, and transformers 4.57.1. Follow the official compatibility guidance when installing dependencies.

## License and attribution

This tutorial and its companion scripts are released under the [MIT License](LICENSE). The verl framework is installed separately and retains its own license. This is an independent tutorial and is not endorsed by the upstream project.
