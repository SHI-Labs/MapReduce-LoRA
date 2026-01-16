#!/bin/bash

set -euo pipefail

# export LD_LIBRARY_PATH=/usr/local/lib/:/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

# Set environment variables
export NCCL_DEBUG=INFO

# ================================
# MULTI-NODE CONFIGURATION
# ================================
# Multi-node training configuration (must be set early in script)
if [ -n "${MASTER_PORT:-}" ]; then
  NUM_GPUS=${RUNAI_NUM_OF_GPUS:-${NUM_GPUS:-8}}
  NUMS_NODES=${WORLD_SIZE:-${NUMS_NODES:-1}}
  export NODE_RANK=${RANK:-0}
else
  MASTER_ADDR="127.0.0.1"
  MASTER_PORT=9998
  if [ -n "${RUNAI_NUM_OF_GPUS:-}" ]; then
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
echo "  MASTER_PORT (base): $MASTER_PORT"

echo ""

# Orchestrate parallel cycles of (geneval, pickscore, ocr) training with periodic merges.
# - Each cycle launches 3 trainings in parallel (each using NODES x GPUS_PER_NODE).
# - Waits for all three to reach MERGE_STEPS, then merges LoRAs (1:1:1) on one node.
# - Uses merged adapter as resume_from for next cycle.
# - Optional: run a one-time environment setup before launching (ENV_SETUP=1).
#
# Environment/Args (override as needed):
#   CYCLES: number of cycles (default: 1)
#   MERGE_STEPS: global steps per task before merge (default: 100)
#   WEIGHTS: merge weights (default: "1 1 1")
#   NODES_PER_TASK: number of nodes per training (default: 4)
#   GPUS_PER_NODE: GPUs per node (default: 8)
#   MASTER_ADDR: primary node IP/hostname (default: 127.0.0.1)
#   MASTER_PORT: base port; parallel tasks use BASE, BASE+1, BASE+2 (default: 9998)
#   LOG_DIR: log directory (default: ./logs)
#   MODEL_PATH: optional local base model snapshot path (for merging)
#   OUT_ROOT: base output for merged adapters (default: ./logs/merge_lora_auto)
#   ENV_SETUP: if "1", preparing environment before training (default: 0)
#   START_GENEVAL_REWARD: if "1", start reward server before geneval (default: 0)
#   STOP_GENEVAL_REWARD_AFTER: if "1", stop reward server after geneval finishes (default: 0)
#   PRETRAINED_MODEL_PATH: pretrained model snapshot path used at training start
#   START_AT_CYCLE: resume-or-start from this cycle index (default: 1)
#   RESUME_RUN_TS: reuse an existing RUN_TS (directory under OUT_ROOT) to resume a killed run
#   RESUME_GENEVAL_CKPT|RESUME_PICK_CKPT|RESUME_OCR_CKPT: optional checkpoint dirs (checkpoint-*) to resume each task for START_AT_CYCLE
#   SKIP_COMPLETED_TASKS: if "1", do not relaunch tasks that already wrote a DONE flag (default: 1)
#   AUTO_RESUME: if "1", auto-detect latest RUN_TS and latest cycle to resume (default: 0)
#   AUTO_RESUME_DIR: directory to search RUN_TS folders in (default: OUT_ROOT)
#

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

CYCLES="${CYCLES:-80}"
MERGE_STEPS="${MERGE_STEPS:-100}"
WEIGHTS="${WEIGHTS:-1 1 1}"
NODES_PER_TASK="${NODES_PER_TASK:-4}"  # nodes used by each parallel task (geneval/pickscore/ocr)
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-9998}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
MODEL_PATH="${MODEL_PATH:-}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/merge_lora_auto/sd35m}"
ENV_SETUP="${ENV_SETUP:-1}"
# Optional: start the GenEval reward server before training the geneval task
START_GENEVAL_REWARD="${START_GENEVAL_REWARD:-1}"
# Optional: stop the GenEval reward server right after geneval finishes (for parallel training, we don't need to stop it)
STOP_GENEVAL_REWARD_AFTER="${STOP_GENEVAL_REWARD_AFTER:-0}"
# Allow overriding the pretrained model snapshot path used by training
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${HF_HUB_CACHE_DIR}/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}"
# Resume controls
START_AT_CYCLE="${START_AT_CYCLE:-1}"
RESUME_RUN_TS="${RESUME_RUN_TS:-}" #20251219_112111
RESUME_GENEVAL_CKPT="${RESUME_GENEVAL_CKPT:-}"
RESUME_PICK_CKPT="${RESUME_PICK_CKPT:-}"
RESUME_OCR_CKPT="${RESUME_OCR_CKPT:-}"
SKIP_COMPLETED_TASKS="${SKIP_COMPLETED_TASKS:-0}"
AUTO_RESUME="${AUTO_RESUME:-0}"
AUTO_RESUME_DIR="${AUTO_RESUME_DIR:-${OUT_ROOT}}"
MERGE_COORD_RANK="${MERGE_COORD_RANK:-0}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

timestamp() { date +"%Y%m%d_%H%M%S"; }

# If AUTO_RESUME is enabled and RESUME_RUN_TS not provided, pick the latest RUN_TS under AUTO_RESUME_DIR
if [ "${AUTO_RESUME}" = "1" ] && [ -z "${RESUME_RUN_TS}" ]; then
  latest_ts_dir="$(ls -1d "${AUTO_RESUME_DIR}"/2* 2>/dev/null | sort | tail -n1 || true)"
  if [ -n "${latest_ts_dir}" ]; then
    RESUME_RUN_TS="$(basename "${latest_ts_dir}")"
    echo "AUTO_RESUME: selected latest RUN_TS=${RESUME_RUN_TS} under ${AUTO_RESUME_DIR}"
  else
    echo "AUTO_RESUME: No RUN_TS directories found under ${AUTO_RESUME_DIR}" >&2
  fi
fi

# Per-run timestamped root for cycles (fixed at script start)
RUN_TS_FILE="${OUT_ROOT}/RUN_TS.txt"
# Cleanup the RUN_TS barrier file before starting a new run (coordinator only)
if [ -z "${RESUME_RUN_TS}" ] && [ "${NODE_RANK}" = "0" ]; then
  rm -f "${RUN_TS_FILE}"
fi
if [ -n "${RESUME_RUN_TS}" ]; then
  # Reuse an existing run timestamp (resume a killed run)
  RUN_TS="${RESUME_RUN_TS}"
  if [ "${NODE_RANK}" = "0" ]; then
    echo "${RUN_TS}" > "${RUN_TS_FILE}"
  else
    echo "Resuming with provided RUN_TS=${RUN_TS}"
  fi
else
  if [ "${NODE_RANK}" = "0" ]; then
    RUN_TS="$(timestamp)"
    echo "${RUN_TS}" > "${RUN_TS_FILE}"
  else
    echo "Waiting for RUN_TS at ${RUN_TS_FILE}"
    while [ ! -f "${RUN_TS_FILE}" ]; do
      sleep 1
    done
    RUN_TS="$(cat "${RUN_TS_FILE}")"
  fi
fi
RUN_ROOT="${OUT_ROOT}/${RUN_TS}"
mkdir -p "${RUN_ROOT}"

# If auto-resuming, clear any stale decision markers to avoid adopting old values
if [ "${AUTO_RESUME}" = "1" ]; then
  echo "AUTO_RESUME: cleaning stale decision markers under ${RUN_ROOT}"
  rm -f "${RUN_ROOT}/START_AT_CYCLE" \
        "${RUN_ROOT}/RESUME_GENEVAL_CKPT" \
        "${RUN_ROOT}/RESUME_PICK_CKPT" \
        "${RUN_ROOT}/RESUME_OCR_CKPT"
fi

# One-time environment setup (only runs if ENV_SETUP=1)
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
  local machine_rank="$8"

  local config_preset
  case "$task" in
    geneval)   config_preset="config/grpo.py:geneval_sd3" ;;
    pickscore) config_preset="config/grpo.py:pickscore_sd3" ;;
    ocr)       config_preset="config/grpo.py:general_ocr_sd3" ;;
    *) echo "Unknown TASK: $task (expected: geneval|pickscore|ocr)" >&2; return 1 ;;
  esac

  local num_procs=$((nodes * gpus_per_node))
  local cmd="accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines ${nodes} --num_processes ${num_procs} \
    --machine_rank ${machine_rank} --main_process_ip ${master_addr} --main_process_port ${master_port} \
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

# Helper: get node IP by global rank (requires NODE_IPS="ip0 ip1 ip2 ..."; falls back to MASTER_ADDR)
get_node_ip_by_rank() {
  local rank="$1"
  if [ -z "${NODE_IPS:-}" ]; then
    # Fallback: try to derive host by replacing a trailing "-<num>" suffix with requested rank
    # e.g., pluto-prod-foo-0 -> pluto-prod-foo-${rank}
    if [[ "${MASTER_ADDR}" =~ ^(.+)-([0-9]+)$ ]]; then
      local prefix="${BASH_REMATCH[1]}"
      echo "${prefix}-${rank}"
      return 0
    else
      echo "${MASTER_ADDR}"
      return 0
    fi
  fi
  # shellcheck disable=SC2206
  local ips_arr=( ${NODE_IPS} )
  if [ -z "${ips_arr[$rank]:-}" ]; then
    echo "${MASTER_ADDR}"
  else
    echo "${ips_arr[$rank]}"
  fi
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

extract_save_dir_from_log() {
  local logfile="$1"
  # Prefer 'Final save_dir:' (wandb init) then fallback to 'Auto-constructed save_dir:'
  local path_line
  path_line=$(grep -E "Final save_dir:|Auto-constructed save_dir:" "${logfile}" | tail -n1 || true)
  # Extract everything after the colon+ space
  echo "${path_line}" | sed -E 's/.*save_dir:\s*//'
}

latest_checkpoint_dir() {
  local save_dir="$1"
  # Return best checkpoint directory under save_dir, matching common layout:
  #   save_dir/<task>/wandbID-*/checkpoint-<step>
  # Strategy:
  # 1) Prefer checkpoint-* under wandbID-*; choose by highest numeric <step>
  # 2) Fallback: any checkpoint-* within 3 levels by newest mtime
  # 3) If none, return empty string

  # Prefer: numeric step ranking under wandbID-*
  local best=""
  best="$({ find "${save_dir}" -mindepth 1 -maxdepth 3 -type d -path '*/wandbID-*/*' -name 'checkpoint-*' -print 2>/dev/null \
    | awk -F/ '{
        name=$NF;
        step=name; sub(/^checkpoint-/, "", step);
        if (step ~ /^[0-9]+$/) { print step, $0 } else { print -1, $0 }
      }' \
    | sort -k1,1nr -k2,2r \
    | head -n1 \
    | cut -d" " -f2-; } || true)"
  if [ -n "${best}" ]; then
    echo "${best%/}"
    return 0
  fi

  # Fallback: newest by mtime anywhere within 3 levels
  local mtime_best=""
  mtime_best="$({ find "${save_dir}" -mindepth 1 -maxdepth 3 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n1 \
    | awk '{$1=""; sub(/^ /,""); print}'; } || true)"
  if [ -n "${mtime_best}" ]; then
    echo "${mtime_best%/}"
    return 0
  fi

  echo ""
}

resolve_ckpt_from_log() {
  local logfile="$1"
  local save_dir ckpt_dir
  save_dir="$(extract_save_dir_from_log "${logfile}")"
  if [ -z "${save_dir}" ]; then
    echo "ERROR: Could not find save_dir in log: ${logfile}" >&2
    return 1
  fi
  ckpt_dir="$(latest_checkpoint_dir "${save_dir}")"
  if [ -z "${ckpt_dir}" ]; then
    echo "ERROR: Could not find any checkpoint-* under save_dir=${save_dir}" >&2
    return 1
  fi
  ckpt_dir="${ckpt_dir%/}"
  echo "${ckpt_dir}"
}

launch_task_tmux() {
  local task="$1"
  local resume_from="$2"
  local step_limit="$3"
  local cycle="$4"
  local nodes="$5"
  local gpus_per_node="$6"
  local master_addr="$7"
  local master_port="$8"
  local machine_rank="$9"
  local logfile="${10}"
  local save_dir="${11}"

  echo "Launching task=${task} resume_from='${resume_from}' steps=${step_limit} (port: ${master_port})" >&2

  local session_name="sd35M_${task}_$(date +%s)"
  local accel_cmd
  if [ "${task}" = "geneval" ] && [ "${START_GENEVAL_REWARD}" = "1" ]; then
    ensure_geneval_reward_server
  fi
  accel_cmd="$(build_accel_cmd "${task}" "${resume_from}" "${step_limit}" "${nodes}" "${gpus_per_node}" "${master_addr}" "${master_port}" "${machine_rank}")"

  # start tmux: activate env, cd to MapReduce-LoRA, set cache, execute and tee to log
  # Ensure log file exists so errors before command start are captured
  mkdir -p "$(dirname "${logfile}")"
  touch "${logfile}"
  # Launch tmux; if tmux itself fails to create the session, capture the error and mark EXIT_CODE
  if ! tmux new-session -d -s "${session_name}" "bash -lc 'set -o pipefail; source '$(conda info --base)/etc/profile.d/conda.sh'; conda activate mapreduce-lora; cd ${REPO_ROOT}; export EXTERNAL_SAVE_DIR=\"${save_dir}\"; export TASK_LOCAL_RANK=\"${machine_rank}\"; export MASTER_ADDR=\"${master_addr}\"; export MASTER_PORT=\"${master_port}\"; export GLOO_SOCKET_IFNAME=eth0; export NCCL_SOCKET_IFNAME=eth0; export PYTHONUNBUFFERED=1; exec > >(tee -a \"${logfile}\") 2>&1; echo \"[LAUNCH] task=${task} cycle=${cycle} machine_rank=${machine_rank} nodes=${nodes} gpus_per_node=${gpus_per_node}\"; echo \"[RZV] MASTER_ADDR=${master_addr} MASTER_PORT=${master_port}\"; echo \"[CMD] ${accel_cmd}\"; stdbuf -oL -eL ${accel_cmd}; code=\$?; echo \"\${code}\" > \"${save_dir}/EXIT_CODE.${machine_rank}\"; touch \"${save_dir}/DONE.${machine_rank}\"'"; then
    {
      echo "[TMUX_ERROR] Failed to create tmux session '${session_name}' at $(date '+%F %T')"
    } >> "${logfile}" 2>&1
    echo "-99" > "${save_dir}/EXIT_CODE.${machine_rank}"
    return 1
  fi

  echo "${session_name}"
}

## Auto-detect latest cycle and task ckpts when AUTO_RESUME is enabled
if [ "${AUTO_RESUME}" = "1" ]; then
  if [ "${NODE_RANK}" = "${MERGE_COORD_RANK}" ]; then
    latest_cycle_dir="$(ls -1d "${RUN_ROOT}"/cycle_* 2>/dev/null | sort -V | tail -n1 || true)"
    if [ -n "${latest_cycle_dir}" ]; then
      latest_cycle_num="$(basename "${latest_cycle_dir}" | sed -E 's/cycle_([0-9]+)/\1/')"
      if [ -f "${RUN_ROOT}/cycle_${latest_cycle_num}/MERGE_DONE" ]; then
        START_AT_CYCLE="$((latest_cycle_num + 1))"
        echo "AUTO_RESUME: last completed cycle=${latest_cycle_num}; starting at next cycle=${START_AT_CYCLE}"
      else
        START_AT_CYCLE="${latest_cycle_num}"
        g_save="${RUN_ROOT}/cycle_${START_AT_CYCLE}/geneval"
        p_save="${RUN_ROOT}/cycle_${START_AT_CYCLE}/pickscore"
        o_save="${RUN_ROOT}/cycle_${START_AT_CYCLE}/ocr"
        g_ckpt="$(latest_checkpoint_dir "${g_save}")"
        p_ckpt="$(latest_checkpoint_dir "${p_save}")"
        o_ckpt="$(latest_checkpoint_dir "${o_save}")"
        if [ -n "${g_ckpt}" ]; then
          RESUME_GENEVAL_CKPT="${g_ckpt%/}"
        else
          echo "AUTO_RESUME: does not find geneval checkpoint in ${g_save}, will start from scratch"
        fi
        if [ -n "${p_ckpt}" ]; then
          RESUME_PICK_CKPT="${p_ckpt%/}"
        else
          echo "AUTO_RESUME: does not find pickscore checkpoint in ${p_save}, will start from scratch"
        fi
        if [ -n "${o_ckpt}" ]; then
          RESUME_OCR_CKPT="${o_ckpt%/}"
        else
          echo "AUTO_RESUME: does not find ocr checkpoint in ${o_save}, will start from scratch"
        fi
        echo "AUTO_RESUME: cycle=${START_AT_CYCLE} geneval=${RESUME_GENEVAL_CKPT:-N/A} pickscore=${RESUME_PICK_CKPT:-N/A} ocr=${RESUME_OCR_CKPT:-N/A}"
      fi
    else
      echo "AUTO_RESUME: No cycle_* directories found under ${RUN_ROOT}" >&2
    fi
    # Publish coordinator decisions for other nodes
    echo "${START_AT_CYCLE}" > "${RUN_ROOT}/START_AT_CYCLE"
    if [ -n "${RESUME_GENEVAL_CKPT:-}" ]; then echo "${RESUME_GENEVAL_CKPT}" > "${RUN_ROOT}/RESUME_GENEVAL_CKPT"; fi
    if [ -n "${RESUME_PICK_CKPT:-}" ]; then echo "${RESUME_PICK_CKPT}" > "${RUN_ROOT}/RESUME_PICK_CKPT"; fi
    if [ -n "${RESUME_OCR_CKPT:-}" ]; then echo "${RESUME_OCR_CKPT}" > "${RUN_ROOT}/RESUME_OCR_CKPT"; fi
  else
    echo "AUTO_RESUME: waiting for coordinator decision files..."
    while [ ! -f "${RUN_ROOT}/START_AT_CYCLE" ]; do
      sleep 2
    done
    START_AT_CYCLE="$(cat "${RUN_ROOT}/START_AT_CYCLE")"
    if [ -f "${RUN_ROOT}/RESUME_GENEVAL_CKPT" ]; then RESUME_GENEVAL_CKPT="$(cat "${RUN_ROOT}/RESUME_GENEVAL_CKPT")"; fi
    if [ -f "${RUN_ROOT}/RESUME_PICK_CKPT" ]; then RESUME_PICK_CKPT="$(cat "${RUN_ROOT}/RESUME_PICK_CKPT")"; fi
    if [ -f "${RUN_ROOT}/RESUME_OCR_CKPT" ]; then RESUME_OCR_CKPT="$(cat "${RUN_ROOT}/RESUME_OCR_CKPT")"; fi
    echo "AUTO_RESUME: adopted START_AT_CYCLE=${START_AT_CYCLE} geneval=${RESUME_GENEVAL_CKPT:-N/A} pickscore=${RESUME_PICK_CKPT:-N/A} ocr=${RESUME_OCR_CKPT:-N/A}"
  fi
fi

echo "Starting parallel orchestration: CYCLES=${CYCLES}, MERGE_STEPS=${MERGE_STEPS}, WEIGHTS=${WEIGHTS}"

RESUME_PARENT=""  # parent dir of a checkpoint (contains 'lora'), or empty for first cycle

for (( cycle=START_AT_CYCLE; cycle<=CYCLES; cycle++ )); do
  echo "=== Cycle ${cycle}/${CYCLES} (parallel) ==="

  # For cycles > 1, read resume_from from shared marker written by coordinator
  if [ "${cycle}" -gt 1 ]; then
    RESUME_FILE="${RUN_ROOT}/RESUME_FROM_NEXT"
    PREV_DONE="${RUN_ROOT}/cycle_$((cycle - 1))/MERGE_DONE"
    echo "Waiting for previous cycle barrier: ${PREV_DONE}"
    while [ ! -f "${PREV_DONE}" ]; do
      sleep 5
    done
    RESUME_PARENT="$(cat "${RESUME_FILE}")"
    echo "Using resume_from: ${RESUME_PARENT}"
  fi

  # Distinct master ports for each parallel task
  MASTER_PORT_GENEVAL="${MASTER_PORT}"
  MASTER_PORT_PICK=$((MASTER_PORT + 1))
  MASTER_PORT_OCR=$((MASTER_PORT + 2))

  # Prepare logs and save dirs (shared paths for cross-node coordination)
  G_SAVE="${RUN_ROOT}/cycle_${cycle}/geneval";   mkdir -p "${G_SAVE}"
  P_SAVE="${RUN_ROOT}/cycle_${cycle}/pickscore"; mkdir -p "${P_SAVE}"
  O_SAVE="${RUN_ROOT}/cycle_${cycle}/ocr";       mkdir -p "${O_SAVE}"
  G_LOG="${G_SAVE}/train.log"
  P_LOG="${P_SAVE}/train.log"
  O_LOG="${O_SAVE}/train.log"

  # Determine this node's assigned task group
  local_nodes_per_task="${NODES_PER_TASK}"
  task_index=$(( NODE_RANK / local_nodes_per_task ))   # 0: geneval, 1: pickscore, 2: ocr
  task_local_rank=$(( NODE_RANK - task_index * local_nodes_per_task ))

  # Compute group master address based on base rank of the group
  group_base_rank=$(( task_index * local_nodes_per_task ))
  if [ "${local_nodes_per_task}" -eq 1 ]; then
    GROUP_MASTER_ADDR="127.0.0.1"
  else
    GROUP_MASTER_ADDR="$(get_node_ip_by_rank "${group_base_rank}")"
  fi

  # Per-task resume overrides for the starting cycle (optional)
  G_RESUME="${RESUME_PARENT}"
  P_RESUME="${RESUME_PARENT}"
  O_RESUME="${RESUME_PARENT}"
  if [ "${cycle}" -eq "${START_AT_CYCLE}" ]; then
    if [ -n "${RESUME_GENEVAL_CKPT}" ]; then G_RESUME="${RESUME_GENEVAL_CKPT%/}/lora"; fi
    if [ -n "${RESUME_PICK_CKPT}" ]; then P_RESUME="${RESUME_PICK_CKPT%/}/lora"; fi
    if [ -n "${RESUME_OCR_CKPT}" ]; then O_RESUME="${RESUME_OCR_CKPT%/}/lora"; fi
  fi

  # Launch exactly one training per node, according to its assigned task group
  G_SESSION=""; P_SESSION=""; O_SESSION=""
  if [ "${task_index}" -eq 0 ]; then
    if [ "${SKIP_COMPLETED_TASKS}" = "1" ] && [ -f "${G_SAVE}/DONE" ]; then
      echo "geneval already DONE for cycle ${cycle}, skipping relaunch."
    else
      G_SESSION="$(launch_task_tmux geneval   "${G_RESUME}" "${MERGE_STEPS}" "${cycle}" "${local_nodes_per_task}" "${GPUS_PER_NODE}" "${GROUP_MASTER_ADDR}" "${MASTER_PORT_GENEVAL}" "${task_local_rank}" "${G_LOG}" "${G_SAVE}")"
    fi
  elif [ "${task_index}" -eq 1 ]; then
    if [ "${SKIP_COMPLETED_TASKS}" = "1" ] && [ -f "${P_SAVE}/DONE" ]; then
      echo "pickscore already DONE for cycle ${cycle}, skipping relaunch."
    else
      P_SESSION="$(launch_task_tmux pickscore "${P_RESUME}" "${MERGE_STEPS}" "${cycle}" "${local_nodes_per_task}" "${GPUS_PER_NODE}" "${GROUP_MASTER_ADDR}" "${MASTER_PORT_PICK}"    "${task_local_rank}" "${P_LOG}" "${P_SAVE}")"
    fi
  elif [ "${task_index}" -eq 2 ]; then
    if [ "${SKIP_COMPLETED_TASKS}" = "1" ] && [ -f "${O_SAVE}/DONE" ]; then
      echo "ocr already DONE for cycle ${cycle}, skipping relaunch."
    else
      O_SESSION="$(launch_task_tmux ocr       "${O_RESUME}" "${MERGE_STEPS}" "${cycle}" "${local_nodes_per_task}" "${GPUS_PER_NODE}" "${GROUP_MASTER_ADDR}" "${MASTER_PORT_OCR}"     "${task_local_rank}" "${O_LOG}" "${O_SAVE}")"
    fi
  else
    echo "NODE_RANK=${NODE_RANK} not assigned to any task for local_nodes_per_task=${local_nodes_per_task}; idling this cycle."
  fi

  # Coordinator responsibilities (default rank 0): wait for all tasks to finish, merge, and publish resume for next cycle
  MERGE_COORD_RANK="${MERGE_COORD_RANK:-0}"
  if [ "${NODE_RANK}" = "${MERGE_COORD_RANK}" ]; then
    echo "Coordinator (NODE_RANK=${NODE_RANK}) waiting for task completion flags..."
    for save in "${G_SAVE}" "${P_SAVE}" "${O_SAVE}"; do
      for ((r=0; r<${NODES_PER_TASK}; r++)); do
        while [ ! -f "${save}/DONE.${r}" ]; do
          sleep 2
        done
      done
    done

  echo "All tasks finished. Parsing logs to resolve checkpoints..."
  GENEVAl_CKPT="$(resolve_ckpt_from_log "${G_LOG}")"
  PICK_CKPT="$(resolve_ckpt_from_log "${P_LOG}")"
  OCR_CKPT="$(resolve_ckpt_from_log "${O_LOG}")"

  if [ -z "${GENEVAl_CKPT}" ] || [ -z "${PICK_CKPT}" ] || [ -z "${OCR_CKPT}" ]; then
    echo "ERROR: Failed to resolve one or more checkpoints. Aborting." >&2
    exit 1
  fi

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

  # Merge on coordinator
  conda activate mapreduce-lora
  python "${MERGE_SCRIPT}" \
    --model_name sd35m \
    $( [ -n "${MODEL_PATH}" ] && echo --model_path "${MODEL_PATH}" ) \
    --lora_paths "${GEN_LORA}" "${PICK_LORA}" "${OCR_LORA}" \
    --weights ${WEIGHTS} \
    --output_dir "${MERGE_OUT}"
  # Update a symlink for convenience
  ln -sfn "${MERGE_PARENT}" "${OUT_ROOT}/LATEST_MERGED"

  # If reward server is managed here, optionally stop it after geneval
  if [ "${START_GENEVAL_REWARD}" = "1" ] && [ "${STOP_GENEVAL_REWARD_AFTER}" = "1" ]; then
    stop_geneval_reward_server
  fi

  # Publish resume path for next cycle and release barrier
  echo "${MERGE_PARENT}" > "${RUN_ROOT}/RESUME_FROM_NEXT"
  touch "${RUN_ROOT}/cycle_${cycle}/MERGE_DONE"
  echo "Cycle ${cycle} complete. Next resume_from: ${MERGE_PARENT}"
  else
    # Non-coordinator: wait until merge completes before proceeding to next cycle
    echo "Node ${NODE_RANK} waiting for coordinator merge barrier..."
    while [ ! -f "${RUN_ROOT}/cycle_${cycle}/MERGE_DONE" ]; do
      sleep 5
    done
  fi
done

echo "All cycles complete."

# Cleanup the RUN_TS barrier file after run completes (coordinator only)
if [ "${NODE_RANK}" = "0" ]; then
  rm -f "${RUN_TS_FILE}"
fi



