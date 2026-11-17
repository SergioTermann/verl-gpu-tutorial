#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/verl-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=2
python local_smoke/prepare.py
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/opt/verl-smoke/train.parquet \
  data.val_files=/opt/verl-smoke/val.parquet \
  data.train_batch_size=4 data.val_batch_size=2 \
  data.max_prompt_length=32 data.max_response_length=16 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=/opt/verl-smoke/model \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.max_model_len=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=64 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  custom_reward_function.path=$PWD/local_smoke/reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger='[console]' \
  trainer.project_name=local_smoke trainer.experiment_name=tiny_qwen2_grpo \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.save_freq=1 \
  trainer.default_local_dir=/opt/verl-smoke/checkpoints \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.num_cpus=4 \
  2>&1 | tee local_smoke/train.log
