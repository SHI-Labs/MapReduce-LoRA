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

# vLLM configuration - each node runs its own vLLM server
# export VLLM_HOST="localhost"
# export VLLM_PORT="8000"

echo "Multi-node configuration:"
echo "  NUMS_NODES: $NUMS_NODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
# echo "  vLLM: Each node runs its own vLLM server on ${VLLM_HOST}:${VLLM_PORT}"

echo ""

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
MERGE_SCRIPT="${REPO_ROOT}/scripts/merge_scripts/merge_lora.py"


# Orchestrate sequential cycles of (geneval, pickscore, ocr) training with periodic merges.
# - After each cycle (3 tasks), merges LoRAs (1:1:1) and uses merged as next resume_from.
# - Optional: run a one-time environment setup before launching (ENV_SETUP=1).
#
# Environment/Args (override as needed):
#   CYCLES: number of cycles (default: 80)
#   MERGE_STEPS: global steps per task before merge (default: 100)
#   WEIGHTS: merge weights (default: "1 1 1")
#   NODES: number of nodes (default: 4)
#   GPUS_PER_NODE: GPUs per node (default: 8)
#   MASTER_ADDR: primary node IP/hostname (default: 127.0.0.1)
#   MASTER_PORT: primary port (default: 9998)
#   LOG_DIR: log directory (default: ./logs)
#   MODEL_PATH: optional local base model snapshot path (for merging)
#   OUT_ROOT: base output for merged adapters (default: ./logs/merge_lora_auto)
#   ENV_SETUP: if "1", run setup_env_once.sh before training (default: 0)
#

CYCLES="${CYCLES:-80}"
MERGE_STEPS="${MERGE_STEPS:-100}"
WEIGHTS="${WEIGHTS:-1 1 1}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-9998}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
MODEL_PATH="${MODEL_PATH:-}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/merge_lora_auto}"
ENV_SETUP="${ENV_SETUP:-0}"
# Optional: start the GenEval reward server before training the geneval task
START_GENEVAL_REWARD="${START_GENEVAL_REWARD:-1}"
# Optional: stop the GenEval reward server right after geneval finishes
STOP_GENEVAL_REWARD_AFTER="${STOP_GENEVAL_REWARD_AFTER:-1}"
# Allow overriding the pretrained model snapshot path used by training
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${HF_HUB_CACHE_DIR}/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"


timestamp() { date +"%Y%m%d_%H%M%S"; }

# Per-run timestamped root for cycles (fixed at script start)
RUN_TS="$(timestamp)"
RUN_ROOT="${OUT_ROOT}/${RUN_TS}"
mkdir -p "${RUN_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"

setup_environment() {

  if ! conda env list | grep -q "^mapreduce-lora"; then
    rm -rf /opt/conda/pkgs/icu-75.1-he02047a_0
    conda install --force-reinstall icu -y
    echo "Creating and installing mapreduce-lora environment"
    conda create -n mapreduce-lora python=3.12 -y
    conda activate mapreduce-lora
    pip install diffusers==0.33.1
    pip install torch==2.6.0
    pip install transformers==4.54.0
    pip install protobuf==5.29.5
    pip install sentencepiece==0.2.0
    pip install accelerate==1.9.0
    pip install --no-cache-dir -U packaging ninja==1.11.1.4
    pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir
    pip install xformers==0.0.31.post1
    pip install absl-py==2.3.1
    pip install ml_collections==1.1.0
    pip install wandb==0.18.7
    pip install peft==0.10.0
    # NOTE: for deepspeed
    pip install deepspeed==0.17.2
    # NOTE: for paddleocr
    pip install paddlepaddle-gpu==2.6.2
    pip install paddleocr==2.9.1
    pip install python-Levenshtein==0.27.1

    # NOTE: for loading DiffusionDB
    # pip install pandas==2.2.3
    # pip install pyarrow==21.0.0

    # NOTE: for s3 
    # pip install boto3==1.40.63
  else
    echo "mapreduce-lora environment already exists. It means we have already re-init the node. Skipping."
  fi

  :
}

if [ "${ENV_SETUP}" = "1" ]; then
  echo "Running one-time environment setup..."
  setup_environment
  echo "Environment setup completed."
fi

build_accel_cmd() {
  local task="$1"
  local resume_from="$2"
  local step_limit="$3"
  local nodes="$4"
  local gpus_per_node="$5"
  local master_addr="$6"
  local master_port="$7"

  local config_preset
  case "$task" in
    geneval)   config_preset="config/grpo.py:geneval_sd3_4gpu" ;;
    pickscore) config_preset="config/grpo.py:pickscore_sd3_4gpu" ;;
    ocr)       config_preset="config/grpo.py:general_ocr_sd3_4gpu" ;;
    *) echo "Unknown TASK: $task (expected: geneval|pickscore|ocr)" >&2; return 1 ;;
  esac

  local num_procs=$((nodes * gpus_per_node))
  local cmd="accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines ${nodes} --num_processes ${num_procs} \
    --machine_rank \${NODE_RANK:-0} --main_process_ip ${master_addr} --main_process_port ${master_port} \
    scripts/train_sd3.py \
    --config ${config_preset} \
    --config.pretrained.model='${PRETRAINED_MODEL_PATH}'"

  if [ -n "${resume_from}" ]; then
    cmd="${cmd} --config.resume_from='${resume_from}'"
  fi
  if [ -n "${step_limit}" ] && [ "${step_limit}" != "0" ] && [ "${step_limit}" != "-1" ]; then
    cmd="${cmd} --config.merge_freq=${step_limit}"
  fi
  echo "${cmd}"
}

ensure_geneval_reward_server() {
  if tmux has-session -t geneval-reward 2>/dev/null; then
    echo "GenEval reward server already running (tmux session: geneval-reward)" >&2
    return 0
  fi
  echo "Starting GenEval reward server in tmux (session: geneval-reward)" >&2
  tmux new-session -d -s geneval-reward
  tmux send-keys -t geneval-reward "source '$(conda info --base)/etc/profile.d/conda.sh'" Enter
  tmux send-keys -t geneval-reward "conda activate reward_server" Enter
  tmux send-keys -t geneval-reward "cd ${REWARD_SERVER_ROOT}" Enter
  tmux send-keys -t geneval-reward "NUM_DEVICES=$NUM_GPUS gunicorn 'app_geneval:create_app()'" Enter
}

stop_geneval_reward_server() {
  if tmux has-session -t geneval-reward 2>/dev/null; then
    echo "Stopping GenEval reward server (tmux session: geneval-reward)" >&2
    tmux kill-session -t geneval-reward || true
  else
    echo "GenEval reward server is not running." >&2
  fi
}

wait_tmux_exit() {
  local session="$1"
  # Wait until tmux reports the session no longer exists
  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep 5
  done
}

extract_save_dir_from_log() {
  local logfile="$1"
  # Prefer 'Final save_dir:' (wandb init) then fallback to 'Auto-constructed save_dir:'
  local path_line
  path_line=$(grep -E "Final save_dir:|Auto-constructed save_dir:" "${logfile}" | tail -n1 || true)
  # Extract everything after the colon+space
  echo "${path_line}" | sed -E 's/.*save_dir:\s*//'
}

latest_checkpoint_dir() {
  local save_dir="$1"
  # Return newest checkpoint-*/ directory
  ls -dt "${save_dir}"/checkpoint-*/ 2>/dev/null | head -n1
}

run_task() {
  local task="$1"
  local resume_from="$2"
  local step_limit="$3"
  local cycle="$4"
  local task_ts; task_ts=$(timestamp)
  local logfile="${LOG_DIR}/sd35m_${task}_${task_ts}.log"
  local save_dir="${RUN_ROOT}/cycle_${cycle}/${task}"
  mkdir -p "${save_dir}"

  echo "Launching task=${task} resume_from='${resume_from}' steps=${step_limit}" >&2

  local session_name="sd35M_${task}_$(date +%s)"
  local accel_cmd
  if [ "${task}" = "geneval" ] && [ "${START_GENEVAL_REWARD}" = "1" ]; then
    ensure_geneval_reward_server
  fi
  accel_cmd="$(build_accel_cmd "${task}" "${resume_from}" "${step_limit}" "${NODES}" "${GPUS_PER_NODE}" "${MASTER_ADDR}" "${MASTER_PORT}")"

  # start tmux: activate env, cd to MapReduce-LoRA, set cache, execute and tee to log
  tmux new-session -d -s "${session_name}" "bash -lc 'source '$(conda info --base)/etc/profile.d/conda.sh'; conda activate mapreduce-lora; cd ${REPO_ROOT}; export EXTERNAL_SAVE_DIR=\"${save_dir}\"; ${accel_cmd} |& tee -a \"${logfile}\"'"

  echo "Session: ${session_name}" >&2

  wait_tmux_exit "${session_name}"
  echo "Task ${task} finished. Parsing logs: ${logfile}" >&2

  local save_dir
  save_dir="$(extract_save_dir_from_log "${logfile}")"
  if [ -z "${save_dir}" ]; then
    echo "ERROR: Could not find save_dir in log: ${logfile}" >&2
    exit 1
  fi

  local ckpt_dir
  ckpt_dir="$(latest_checkpoint_dir "${save_dir}")"
  if [ -z "${ckpt_dir}" ]; then
    echo "ERROR: Could not find any checkpoint-* under save_dir=${save_dir}" >&2
    exit 1
  fi

  ckpt_dir="${ckpt_dir%/}"
  if [ "${task}" = "geneval" ] && [ "${START_GENEVAL_REWARD}" = "1" ] && [ "${STOP_GENEVAL_REWARD_AFTER}" = "1" ]; then
    stop_geneval_reward_server
  fi
  echo "save_dir: ${ckpt_dir}" >&2
  echo "${ckpt_dir}"
}

echo "Starting orchestration: CYCLES=${CYCLES}, MERGE_STEPS=${MERGE_STEPS}, WEIGHTS=${WEIGHTS}"

RESUME_PARENT=""  # parent dir of a checkpoint (contains 'lora'), or empty for first cycle

for (( cycle=1; cycle<=CYCLES; cycle++ )); do
  echo "=== Cycle ${cycle}/${CYCLES} ==="

  GENEVAl_CKPT="$(run_task geneval "${RESUME_PARENT}" "${MERGE_STEPS}" "${cycle}")"
  PICK_CKPT="$(run_task pickscore "${RESUME_PARENT}" "${MERGE_STEPS}" "${cycle}")"
  OCR_CKPT="$(run_task ocr "${RESUME_PARENT}" "${MERGE_STEPS}" "${cycle}")"

  GEN_LORA="${GENEVAl_CKPT}/lora"
  PICK_LORA="${PICK_CKPT}/lora"
  OCR_LORA="${OCR_CKPT}/lora"

  MERGE_PARENT="${RUN_ROOT}/cycle_${cycle}/checkpoint-merged"
  MERGE_OUT="${MERGE_PARENT}/lora"
  mkdir -p "${MERGE_OUT}"

  echo " ---- source lora ---- "
  echo "GenEval: ${GEN_LORA}"
  echo "Pickscore: ${PICK_LORA}"
  echo "OCR: ${OCR_LORA}"
  echo " ---- merged lora ---- "
  echo "Merging LoRAs into ${MERGE_OUT}"
  conda activate mapreduce-lora
  python "${MERGE_SCRIPT}" \
    --model_name sd35m \
    $( [ -n "${MODEL_PATH}" ] && echo --model_path "${MODEL_PATH}" ) \
    --lora_paths "${GEN_LORA}" "${PICK_LORA}" "${OCR_LORA}" \
    --weights ${WEIGHTS} \
    --output_dir "${MERGE_OUT}"

  # Next cycle resumes from the parent (so train_sd3.py can find MERGE_PARENT/lora)
  RESUME_PARENT="${MERGE_PARENT}"
  # Update a symlink for convenience
  ln -sfn "${MERGE_PARENT}" "${OUT_ROOT}/LATEST_MERGED"
  echo "Cycle ${cycle} complete. Next resume_from: ${RESUME_PARENT}"
done

echo "All cycles complete."


