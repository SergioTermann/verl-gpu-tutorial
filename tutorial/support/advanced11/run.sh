#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export VERL_ADV11_ROOT="${VERL_ADV11_ROOT:-/opt/verl-advanced11/runs/$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$VERL_ADV11_ROOT"
printf "%s" "$VERL_ADV11_ROOT" > tutorial/support/advanced11/latest_run.txt
source /opt/verl-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=2
python tutorial/support/advanced11/prepare.py
python -m verl.trainer.main_ppo \
  reward.num_workers=1 \
  algorithm.adv_estimator=grpo \
  data.train_files=$VERL_ADV11_ROOT/train.parquet \
  data.val_files=$VERL_ADV11_ROOT/val.parquet \
  data.train_batch_size=4 data.val_batch_size=2 \
  data.max_prompt_length=32 data.max_response_length=16 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$VERL_ADV11_ROOT/model \
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
  custom_reward_function.path=$PWD/tutorial/support/advanced11/reward.py \
  custom_reward_function.name=compute_score \
  trainer.logger='[console]' \
  trainer.project_name=local_smoke trainer.experiment_name=tiny_qwen2_grpo \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.save_freq=1 \
  trainer.default_local_dir=$VERL_ADV11_ROOT/checkpoints \
  trainer.resume_mode=disable \
  ray_kwargs.ray_init.num_cpus=4 \
  2>&1 | tee "$VERL_ADV11_ROOT/train.log" tutorial/support/advanced11/train.log
