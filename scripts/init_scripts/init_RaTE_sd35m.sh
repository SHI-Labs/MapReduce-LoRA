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

echo "Creating SD3.5M GRPO tmux session..."

# Calculate training GPU configuration
# IMPORTANT: All nodes must use the same number of GPUs for torchrun to work
# Each node uses N GPUs for training
TRAIN_GPUS=$((NUM_GPUS))
echo "Node $NODE_RANK: $TRAIN_GPUS GPUs for training"

tmux new-session -d -s sd35M_grpo
tmux send-keys -t sd35M_grpo "source '$(conda info --base)/etc/profile.d/conda.sh'" Enter
tmux send-keys -t sd35M_grpo "conda activate mapreduca-lora" Enter
tmux send-keys -t sd35M_grpo "cd ${REPO_ROOT}" Enter
# tmux send-keys -t sd35M_grpo "export HF_HOME=/mnt/localssd/hf" Enter
# tmux send-keys -t sd35M_grpo "export HUGGINGFACE_HUB_CACHE=/mnt/localssd/hf" Enter
# tmux send-keys -t sd35M_grpo "export DIFFUSERS_CACHE=/mnt/localssd/hf" Enter
# GenEval
tmux send-keys -t sd35M_grpo "accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
                            --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
                            --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
                            scripts/train_sd3_RaTE.py \
                            --config config/sft_ti.py:geneval_sd3 --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80'" Enter

# Picks
# tmux send-keys -t sd35M_grpo "accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
#                             --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
#                             --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#                             scripts/train_sd3_RaTE.py \
#                             --config config/sft_ti.py:pickscore_sd3 --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80'" Enter

# OCR
# tmux send-keys -t sd35M_grpo "accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
#                             --num_machines ${NUMS_NODES} --num_processes $((NUMS_NODES*TRAIN_GPUS)) \
#                             --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#                             scripts/train_sd3_RaTE.py \
#                             --config config/sft_ti.py:general_ocr_sd3 \
#                             --config.pretrained.model='${HF_HUB_CACHE_DIR}/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80'" Enter

# create a tmux conf file
if [ ! -f ~/.tmux.conf ] || ! grep -q "set -g mouse on" ~/.tmux.conf; then
    echo "set -g mouse on" >> ~/.tmux.conf
    tmux source-file ~/.tmux.conf
fi

echo "Start training!"

sleep infinity