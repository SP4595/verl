#!/usr/bin/env bash
# =============================================================================
# Mem-In-Action Memory-OPD + LoRA | Qwen3 | vLLM rollout | FSDP actor
#
# 训练输入必须是 MemoryOPDEpisodeCollector 已展开的 step trace：
#
#   LoCoMo/其他 episode
#     -> MemoryOPDEpisodeCollector
#     -> train_steps.json 或 train_steps.jsonl
#     -> MemoryOPDStepDataset
#     -> MemoryOPDStepAgentLoop
#     -> privileged teacher log-prob
#     -> RayPrivilegeOPDTrainer 更新 student LoRA
#
# 本脚本不会执行 episode，也不能直接读取 LoCoMo 原始 JSON。每条训练样本必须包含：
#
#   {"memory_step": {...}, "action": "...", "status": "..."}
#
# 默认 GPU 布局（单节点共需 8 GPU）：
#
#   global_pool  : 4 GPU，student LoRA actor + vLLM rollout
#   teacher_pool : 4 GPU，privileged teacher vLLM
#
# 使用示例：
#
#   TRAIN_STEP_TRACE=/data/memory_opd/train_steps.jsonl \
#   STUDENT_MODEL=Qwen/Qwen3-8B \
#   TEACHER_MODEL=Qwen/Qwen3-32B \
#   bash examples/meminaction/run_qwen3_8b_lora_opd_fsdp.sh
#
# 只检查并展开 Hydra 配置，不启动 Ray/GPU：
#
#   TRAIN_STEP_TRACE=/data/memory_opd/train_steps.jsonl DRY_RUN=1 \
#   bash examples/meminaction/run_qwen3_8b_lora_opd_fsdp.sh
# =============================================================================

set -euo pipefail

# ------------------------------ 项目路径 ------------------------------

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROJECT_ROOT=$(cd -- "${VERL_ROOT}/../.." && pwd)

export PYTHONPATH="${PROJECT_ROOT}/src:${VERL_ROOT}:${PYTHONPATH:-}"

# 优先使用项目虚拟环境；没有虚拟环境时再回退到 PATH 中的 python3。
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
    DEFAULT_PYTHON_BIN=python3
fi
PYTHON_BIN=${PYTHON_BIN:-"${DEFAULT_PYTHON_BIN}"}

STEP_DATASET_PATH="${VERL_ROOT}/verl/trainer/meminaction/RLDatasets/memory_opd_step_dataset.py"
PROMPT_PATH="${PROJECT_ROOT}/src/prompts/chat.txt"
LEGACY_PROMPT_PATH="${PROJECT_ROOT}/src/prompts/chat_legacy.txt"
UPDATE_POLICY_PATH="${PROJECT_ROOT}/src/prompts/chat_update_policy.txt"
ANSWER_POLICY_PATH="${PROJECT_ROOT}/src/prompts/chat_answer_policy.txt"

# ------------------------------ 数据路径 ------------------------------

# 训练数据必须是 Collector 输出的 step trace。validation 在纯 OPD Trainer 中被关闭，
# 但 VeRL main_ppo 仍会实例化 val Dataset，因此默认复用 train trace。
TRAIN_STEP_TRACE=${TRAIN_STEP_TRACE:-"${PROJECT_ROOT}/data/memory_opd/train_steps.jsonl"}
VAL_STEP_TRACE=${VAL_STEP_TRACE:-"${TRAIN_STEP_TRACE}"}

# ------------------------------ 模型配置 ------------------------------

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-32B}

# LoRA 仅应用于 student。teacher 固定参数，只负责在 privileged prompt 下计算 log-prob。
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}

# ------------------------------ GPU 拓扑 ------------------------------

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-4}

ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-4}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}

# ------------------------------ Batch/长度 ------------------------------

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}

# Student prompt 只包含当前输入和 Cache；teacher prompt 额外包含完整长期 memory。
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-8}

STUDENT_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

# ------------------------------ 优化/蒸馏 ------------------------------

ACTOR_LR=${ACTOR_LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-100}

# k1 只需要 teacher 对 student 已采样 token 的 log-prob，通信量低，适合 OPD。
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}

# Memory Controller 动作预算。这些是全局 prompt/rollout 配置，不属于 Dataset 样本。
QUERY_TOP_N=${QUERY_TOP_N:-5}
UPDATE_QUERY_TOP_N=${UPDATE_QUERY_TOP_N:-5}
MAX_QUERIES_PER_ACTION=${MAX_QUERIES_PER_ACTION:-3}
MAX_QUERY_ROUNDS=${MAX_QUERY_ROUNDS:-3}
MAX_MEMORY_STEPS=${MAX_MEMORY_STEPS:-6}
SEED_QUERY_TOP_N=${SEED_QUERY_TOP_N:-5}
ANSWER_SEED_QUERY_TOP_N=${ANSWER_SEED_QUERY_TOP_N:-5}
UPDATE_PROTOCOL=${UPDATE_PROTOCOL:-replace-cache}

# ------------------------------ 日志/checkpoint ------------------------------

PROJECT_NAME=${PROJECT_NAME:-mem_in_action_lora_opd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_lora_privileged_opd}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
RESUME_MODE=${RESUME_MODE:-auto}

DRY_RUN=${DRY_RUN:-0}

# ------------------------------ 启动前校验 ------------------------------

for required_file in \
    "${TRAIN_STEP_TRACE}" \
    "${VAL_STEP_TRACE}" \
    "${STEP_DATASET_PATH}" \
    "${PROMPT_PATH}" \
    "${LEGACY_PROMPT_PATH}" \
    "${UPDATE_POLICY_PATH}" \
    "${ANSWER_POLICY_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file does not exist: ${required_file}" >&2
        exit 1
    fi
done

if ((LORA_RANK <= 0)); then
    echo "LORA_RANK must be greater than 0 for this LoRA OPD script." >&2
    exit 1
fi

if ((NGPUS_PER_NODE % ROLLOUT_TP != 0)); then
    echo "NGPUS_PER_NODE (${NGPUS_PER_NODE}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})." >&2
    exit 1
fi

if (((TEACHER_GPUS_PER_NODE * NNODES) % TEACHER_TP != 0)); then
    echo "Teacher pool size must be divisible by TEACHER_TP." >&2
    exit 1
fi

if ((TEACHER_MAX_MODEL_LEN < STUDENT_MAX_MODEL_LEN)); then
    echo "TEACHER_MAX_MODEL_LEN must be >= student prompt + response + 1." >&2
    exit 1
fi

# ------------------------------ Hydra overrides ------------------------------

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    critic.enable=False
    reward.reward_model.enable=False

    data.train_files="${TRAIN_STEP_TRACE}"
    data.val_files="${VAL_STEP_TRACE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=error
    data.shuffle=True
    data.dataloader_num_workers=0

    data.custom_cls.path="${STEP_DATASET_PATH}"
    data.custom_cls.name=MemoryOPDStepDataset
    +data.default_data_source=memory_opd
    +data.default_agent_name=memory_opd_step
    +data.validate_custom_sample=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    MODEL+=(actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}")
fi

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.max_model_len=${STUDENT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}

    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS}
    actor_rollout_ref.rollout.agent.default_agent_loop=memory_opd_step
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.trainer.meminaction.agentic_loop.PrivilegeOPDAgentLoopManager

    +actor_rollout_ref.rollout.agent.memory_opd_prompt.prompt_path="${PROMPT_PATH}"
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.legacy_prompt_path="${LEGACY_PROMPT_PATH}"
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.update_policy_path="${UPDATE_POLICY_PATH}"
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.answer_policy_path="${ANSWER_POLICY_PATH}"
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.controller_mode=separated
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.update_protocol="${UPDATE_PROTOCOL}"
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.query_top_n=${QUERY_TOP_N}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.update_query_top_n=${UPDATE_QUERY_TOP_N}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.max_queries_per_action=${MAX_QUERIES_PER_ACTION}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.max_query_rounds=${MAX_QUERY_ROUNDS}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.max_steps=${MAX_MEMORY_STEPS}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.seed_query_top_n=${SEED_QUERY_TOP_N}
    +actor_rollout_ref.rollout.agent.memory_opd_prompt.answer_seed_query_top_n=${ANSWER_SEED_QUERY_TOP_N}
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_models.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.enforce_eager=True
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True

    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

TRAINER=(
    trainer.trainer_class=verl.trainer.meminaction.ray_privilege_opd_trainer.RayPrivilegeOPDTrainer
    trainer.balance_batch=True
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.val_only=False
    trainer.test_freq=-1
    trainer.save_freq=${SAVE_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode="${RESUME_MODE}"
)

COMMAND=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${DISTILLATION[@]}"
    "${TRAINER[@]}"
)

cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Hydra dry-run command:\n'
    printf ' %q' "${COMMAND[@]}" --cfg job --resolve "$@"
    printf '\n'
    exec "${COMMAND[@]}" --cfg job --resolve "$@"
fi

printf 'Starting Memory-OPD LoRA training.\n'
printf '  train step trace       : %s\n' "${TRAIN_STEP_TRACE}"
printf '  student / teacher      : %s / %s\n' "${STUDENT_MODEL}" "${TEACHER_MODEL}"
printf '  student / teacher GPUs : %s / %s per node\n' "${NGPUS_PER_NODE}" "${TEACHER_GPUS_PER_NODE}"
printf '  LoRA rank / alpha      : %s / %s\n' "${LORA_RANK}" "${LORA_ALPHA}"
printf '  checkpoint dir         : %s\n' "${CHECKPOINT_DIR}"

exec "${COMMAND[@]}" "$@"
