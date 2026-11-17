#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VERL_VENV:-/opt/verl-venv}/bin/activate"
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 WANDB_MODE=disabled OMP_NUM_THREADS=2
artifact="${VERL_ADV14_ROOT:?Set VERL_ADV14_ROOT to the export directory printed by lesson 14}"
test -f "$artifact/manifest.json"
run="$artifact/grpo-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$run"
python -m verl.trainer.main_ppo \
 reward.num_workers=1 algorithm.adv_estimator=grpo \
 data.train_files="$artifact/train.parquet" data.val_files="$artifact/validation.parquet" \
 data.train_batch_size=16 data.val_batch_size=80 data.max_prompt_length=20 data.max_response_length=2 data.dataloader_num_workers=0 \
 actor_rollout_ref.model.path="$artifact/model" actor_rollout_ref.model.use_remove_padding=true \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.optim.lr=5e-5 \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.agent.num_workers=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.max_model_len=24 \
 actor_rollout_ref.rollout.max_num_batched_tokens=64 actor_rollout_ref.rollout.max_num_seqs=64 \
 actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.enable_chunked_prefill=false \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 custom_reward_function.path="$PWD/tutorial/advanced14/experiment.py" custom_reward_function.name=compute_score \
 trainer.logger='[console]' trainer.project_name=verl_advanced trainer.experiment_name=warm_parity_grpo \
 trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.val_before_train=true trainer.test_freq=5 \
 trainer.total_epochs=10 trainer.total_training_steps=10 trainer.save_freq=10 \
 trainer.default_local_dir="$run/checkpoints" trainer.resume_mode=disable ray_kwargs.ray_init.num_cpus=4 \
 2>&1 | tee "$run/train.log"
printf '%s\n' "$run" > "$artifact/latest_successful_grpo.txt"
printf 'Completed run: %s\n' "$run"
