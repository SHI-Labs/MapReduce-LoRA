#!/bin/bash

set -exo pipefail

export LD_LIBRARY_PATH=/usr/local/lib/:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

# Set environment variables
export NCCL_DEBUG=INFO

# ================================
# MULTI-NODE CONFIGURATION
# ================================
# Multi-node training configuration (must be set early in script)
if [ -n "$MASTER_PORT" ]; then
  NUM_GPUS=$RUNAI_NUM_OF_GPUS
  NUMS_NODES=${WORLD_SIZE}
  export NODE_RANK=$RANK
else
  MASTER_ADDR="127.0.0.1"
  MASTER_PORT=9998
  if [ -n "$RUNAI_NUM_OF_GPUS" ]; then
    NUM_GPUS=$RUNAI_NUM_OF_GPUS
  else
    NUM_GPUS=8
  fi
  NUMS_NODES=1
  export NODE_RANK=0  # used by MDS
fi
export LOCAL_WORLD_SIZE=$NUM_GPUS  # used by MDS

echo "Multi-node configuration:"
echo "  NUMS_NODES: $NUMS_NODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"

echo ""

# ================================
# REPOSITORY SETUP
# ================================
# Check HF cache directory
hf_hub_cache_dir() {
  if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    printf '%s\n' "$HUGGINGFACE_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then
    printf '%s\n' "$HF_HOME/hub"
  elif [ -n "${XDG_CACHE_HOME:-}" ]; then
    printf '%s\n' "$XDG_CACHE_HOME/huggingface/hub"
  else
    printf '%s\n' "$HOME/.cache/huggingface/hub"
  fi
}

# receive the actual directory of HF cache
HF_HUB_CACHE_DIR="$(hf_hub_cache_dir)"
echo "$HF_HUB_CACHE_DIR"

# Use the relocated lightweight launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_SERVER_ROOT="$(cd "${REPO_ROOT}/../reward-server" && pwd -P)"

# ================================
# Training
# ================================

# --------------------------------
# open the GenEval server
# --------------------------------

echo "Creating reward server tmux session..."
tmux new-session -d -s geneval-reward
tmux send-keys -t geneval-reward "source '$(conda info --base)/etc/profile.d/conda.sh'" Enter
tmux send-keys -t geneval-reward "conda activate reward_server" Enter
tmux send-keys -t geneval-reward "cd ${REWARD_SERVER_ROOT}" Enter
tmux send-keys -t geneval-reward "NUM_DEVICES=$NUM_GPUS gunicorn 'app_geneval:create_app()'" Enter

# --------------------------------
# open the GRPO training script
# --------------------------------

echo "Creating FLUX.1-dev GRPO tmux session..."

# Calculate training GPU configuration
# IMPORTANT: All nodes must use the same number of GPUs for torchrun to work
# Each node uses N GPUs for training
TRAIN_GPUS=$((NUM_GPUS))
echo "Node $NODE_RANK: $TRAIN_GPUS GPUs for training"


# --------------------------------
# Find the latest checkpoint for resuming
# --------------------------------
# CHECKPOINT_BASE_PATH="/MapReduce-LoRA/logs/ocr_1.0/FLUX.1-dev-grpo-32gpus/beta0.04_lr0.0003_bz9x32gpus/wandbID-a4ltbyqr"
RESUME_FROM=""

if [ -d "$CHECKPOINT_BASE_PATH" ]; then
    echo "Searching for latest checkpoint in: $CHECKPOINT_BASE_PATH"
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_BASE_PATH" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        RESUME_FROM="$LATEST_CHECKPOINT"
        echo "Found latest checkpoint: $RESUME_FROM"
    else
        echo "No checkpoint found in $CHECKPOINT_BASE_PATH"
    fi
else
    echo "Checkpoint base path does not exist: $CHECKPOINT_BASE_PATH"
fi


tmux new-session -d -s flux_1_dev_grpo
tmux send-keys -t flux_1_dev_grpo "source '$(conda info --base)/etc/profile.d/conda.sh'" Enter
tmux send-keys -t flux_1_dev_grpo "conda activate mapreduce-lora" Enter
tmux send-keys -t flux_1_dev_grpo "cd ${REPO_ROOT}" Enter
# tmux send-keys -t flux_1_dev_grpo "export HF_HOME=/mnt/localssd/hf" Enter
# tmux send-keys -t flux_1_dev_grpo "export HUGGINGFACE_HUB_CACHE=/mnt/localssd/hf" Enter
# tmux send-keys -t flux_1_dev_grpo "export DIFFUSERS_CACHE=/mnt/localssd/hf" Enter


# Build the accelerate command with optional resume_from
# GenEval
ACCEL_CMD="accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
                            --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
                            --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
                            scripts/train_flux.py \
                            --config config/grpo.py:geneval_flux --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'"

# PickScore
# ACCEL_CMD="accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#                             --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
#                             --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#                             scripts/train_flux.py \
#                             --config config/grpo.py:pickscore_flux --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'"

# OCR
# ACCEL_CMD="accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
#                             --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
#                             --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#                             scripts/train_flux.py \
#                             --config config/grpo.py:general_ocr_flux --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'"


# Add resume_from if checkpoint was found
if [ -n "$RESUME_FROM" ]; then
    ACCEL_CMD="$ACCEL_CMD --config.resume_from='$RESUME_FROM'"
fi

tmux send-keys -t flux_1_dev_grpo "$ACCEL_CMD" Enter


# create a tmux conf file
if [ ! -f ~/.tmux.conf ] || ! grep -q "set -g mouse on" ~/.tmux.conf; then
    echo "set -g mouse on" >> ~/.tmux.conf
    tmux source-file ~/.tmux.conf
fi

echo "Start training!"

sleep infinity