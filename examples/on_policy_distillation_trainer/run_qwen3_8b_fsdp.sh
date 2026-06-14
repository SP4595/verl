#!/usr/bin/env bash
# =============================================================================
# On-Policy Distillation 训练启动脚本
# 场景: 文本任务 | vLLM 生成 rollout | FSDP 训练后端 | NVIDIA GPU
#
# 整体流程:
#   1. 定义超参数（环境变量优先，否则使用默认值）
#   2. 组装 Hydra 配置参数数组（DATA / MODEL / ACTOR / ROLLOUT / TRAINER / EXTRA）
#   3. 调用 python3 -m verl.trainer.main_ppo 启动分布式训练
#
# 关键组件关系:
#   ┌─────────────────────────────────────────────────────────────┐
#   │  8 × GPU（学生模型 actor/rollout/ref，使用 FSDP）            │
#   │  4 × GPU（教师模型 vLLM 推理，用于生成 KL 蒸馏信号）          │
#   │  共用 1 个 Ray 集群，通过 ResourcePoolManager 隔离 GPU 池    │
#   └─────────────────────────────────────────────────────────────┘
# =============================================================================

set -xeuo pipefail
# set -x  : 执行前打印每条命令，便于调试
# set -e  : 任何命令失败时立即退出
# set -u  : 引用未定义变量时报错
# set -o pipefail : 管道中任意命令失败时整体失败

# ======================== 用户可调参数 ========================

# 学生模型路径（本地路径或 HuggingFace model id），将被 FSDP 训练
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-8B}
# 教师模型路径，仅用于推理（生成 token 级别的 log-prob 蒸馏目标），不参与梯度更新
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-32B}

# 分布式拓扑
NNODES=${NNODES:-1}              # 节点数（单机默认为 1）
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}          # 每节点 GPU 数（学生模型占用）
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}  # 教师模型所需 GPU 总数（在独立资源池中）
# 注意：TEACHER_WORLD_SIZE 必须等于 teacher_tp × teacher_dp × teacher_pp
# 本脚本 teacher_tp=2，teacher_dp=2（默认），故 TEACHER_WORLD_SIZE=4

# 蒸馏损失配置
# loss_mode=k1 : 使用单样本 KL 估计器（负 KL 作为 reward，梯度效率高）
# 其他可选值: kl, abs, mse, k2, k3, forward_kl_topk（详见 distillation/losses.py）
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
# use_policy_gradient=True : 将蒸馏损失的负值作为 reward 送入策略梯度（OPD 方案）
# use_policy_gradient=False: 直接以监督方式反向传播蒸馏损失（SFT 蒸馏方案）
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
# topk : 只对 teacher 概率最高的 top-k token 计算 KL，节省显存和通信量
# 仅在 loss_mode=forward_kl_topk 时生效；k1/k3 等 estimator 模式下会忽略此参数
distillation_topk=${DISTILLATION_TOPK:-64}

# 批量大小
train_batch_size=${TRAIN_BATCH_SIZE:-128}        # 全局 rollout 批量大小（prompt 数）
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}  # PPO 更新时的 mini-batch 大小
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}     # prompt 最大 token 数（超长时过滤）
max_response_length=${MAX_RESPONSE_LENGTH:-2048} # 生成响应最大 token 数

# 动态批量上限（use_dynamic_bsz=True 时生效）
# 每块 GPU 在一次前向中允许的最大 token 数，用于自动调节 micro-batch size
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

# 优化器
actor_lr=${ACTOR_LR:-1e-6}   # 学生模型 actor 的学习率

# vLLM 学生 rollout 配置
rollout_tp=${ROLLOUT_TP:-2}                       # rollout 阶段 Tensor Parallel 大小
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4} # vLLM 显存占用比例（为 FSDP 留余量）

# vLLM 教师模型推理配置
teacher_tp=${TEACHER_TP:-2}                       # 教师模型 Tensor Parallel 大小
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4} # 教师模型 vLLM 显存占用比例

# 训练调度
total_epochs=${TOTAL_EPOCHS:-15}    # 总训练轮数（遍历训练集次数）
save_freq=${SAVE_FREQ:-200}         # 每隔多少步保存一次 checkpoint
test_freq=${TEST_FREQ:-5}           # 每隔多少步跑一次验证集评测

# 实验追踪（WandB）
project_name=${PROJECT_NAME:-verl_distill_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_vllm_fsdp}

# ======================== 数据文件 ========================
# 训练/验证数据以 Parquet 格式存储，包含字段: prompt, data_source, reward_model 等
# 由 verl/utils/dataset/rl_dataset.py 的 RLHFDataset 类读取
gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

# Hydra/OmegaConf 接受 Python list 字符串格式，支持多数据集混合
train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"

# 模型最大序列长度 = prompt + response + 1 (EOS token)
# 教师模型推理时同样使用此上限（max_model_len）
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# ======================== Hydra 参数数组 ========================
# 以下各数组对应 verl/trainer/config/ppo_trainer.yaml 中的配置字段。
# 数组元素格式为 key=value，由 python3 -m verl.trainer.main_ppo 通过 Hydra 解析。

# ---------- DATA: 数据与算法配置 ----------
DATA=(
    # adv_estimator=grpo: 使用 GRPO 优势估计（group-relative，无需 critic）
    algorithm.adv_estimator=grpo
    # 关闭奖励中的 KL 惩罚项（蒸馏实验中 KL 通过 distillation loss 管控）
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # 过滤掉 prompt 超过 max_prompt_length 的样本，避免截断引入噪声
    data.filter_overlong_prompts=True
    # truncation=error: 响应超长时报错而非静默截断，确保数据质量
    data.truncation='error'
    # shuffle=False: 按顺序采样，配合 SequentialSampler 保证 checkpoint 可恢复
    data.shuffle=False
)

# ---------- MODEL: 学生模型配置（作用于 actor/rollout/ref 三合一 worker）----------
MODEL=(
    # 学生模型路径，由 ActorRolloutRefWorker 加载
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    # 移除 padding token 的激活值，减少计算量（需要 flash-attention）
    actor_rollout_ref.model.use_remove_padding=True
    # 启用梯度检查点以节省显存（以重计算换显存）
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# ---------- ACTOR: 学生模型训练配置 ----------
ACTOR=(
    # 使用 torch.compile 加速训练前向/反向（需 PyTorch >= 2.0）
    actor_rollout_ref.actor.use_torch_compile=True
    # Actor 学习率
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    # PPO mini-batch 大小（每次参数更新所用的样本数）
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    # 启用动态批量大小，根据 token 数自动确定 micro-batch 大小
    actor_rollout_ref.actor.use_dynamic_bsz=True
    # 每 GPU 最大 token 数上限（控制显存峰值）
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # 将模型参数卸载到 CPU，节省 GPU 显存（以吞吐量换显存）
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    # 将优化器状态（Adam m/v/master weight）卸载到 CPU，大幅降低显存占用
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

# ---------- ROLLOUT: 学生模型 vLLM 推理配置 ----------
# rollout 阶段：学生模型使用 vLLM 高效生成 n 条响应
ROLLOUT=(
    # 使用 vLLM 作为 rollout 引擎（替代 HuggingFace generate）
    actor_rollout_ref.rollout.name=vllm
    # Tensor Parallel 大小（每个 vLLM 实例横跨 rollout_tp 块 GPU）
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    # vLLM KV cache 占用 GPU 显存的比例（需与 FSDP 训练共享 GPU，故设较低值）
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    # 每条 prompt 只生成 1 条响应（GRPO 用，非 Best-of-N）
    actor_rollout_ref.rollout.n=1
    # vLLM 模型序列最大长度（prompt + response）
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    # log_prob 重计算阶段也使用动态批量大小
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # log_prob 重计算每 GPU 最大 token 数
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

# ---------- TRAINER: 全局训练控制配置 ----------
TRAINER=(
    # balance_batch=True: 在多 GPU 间均衡 token 数，防止某 GPU 过载
    trainer.balance_batch=True
    # 同时输出到控制台和 WandB
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    # 训练前不先跑验证（加快启动速度）
    trainer.val_before_train=False
    # checkpoint 保存频率（按训练步数）
    trainer.save_freq=${save_freq}
    # 验证集评测频率（按训练步数）
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# ---------- EXTRA: 蒸馏专属配置 ----------
# 对应 verl/workers/config/distillation.py 中的 DistillationConfig 数据类
# 教师模型在独立的 "teacher_pool" GPU 资源池中运行，与学生模型完全隔离
EXTRA=(
    # 开启 on-policy 蒸馏，在 RayPPOTrainer 中激活教师模型 worker 组
    distillation.enabled=True
    # 教师资源池：TEACHER_WORLD_SIZE 块 GPU（= teacher_tp × teacher_dp_replicas）
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    # 教师模型路径（由 DistillationTeacherModelConfig 加载）
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    # 教师 vLLM 推理的 Tensor Parallel 大小
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    # 教师推理引擎类型（vLLM）
    distillation.teacher_models.teacher_model.inference.name=vllm
    # 教师 vLLM KV cache 显存占用比例
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    # 教师模型最大序列长度（等于学生 prompt+response，因为需要完整上下文）
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    # 蒸馏损失模式：k1 = 单样本 KL 估计（负 KL 即 log p_t/p_s，可正可负）
    # 对应 verl/trainer/distillation/losses.py 中注册为 "k1" 的损失函数
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    # 仅在 forward_kl_topk 模式下生效；k1 模式中此参数被忽略
    distillation.distillation_loss.topk=${distillation_topk}
    # 不使用任务奖励（只用蒸馏信号），即纯蒸馏模式
    distillation.distillation_loss.use_task_rewards=False
    # 使用策略梯度方式：将 -KL 作为 advantage 送入 PPO，而非直接监督
    # 对应 distillation/losses.py:distillation_loss() 中的 policy_loss_fn 分支
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    # 对蒸馏损失进行双向截断，防止梯度爆炸（最大 10，最小 -10）
    distillation.distillation_loss.loss_max_clamp=10.0
    # 对 log-prob 进行下界截断，防止数值不稳定（极小概率 token 的 log-prob → -∞）
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

# ======================== 启动训练 ========================
# 入口: verl/trainer/main_ppo.py
#   1. 通过 Hydra 解析上述参数数组（覆盖 config/ppo_trainer.yaml 默认值）
#   2. 初始化 Ray 集群，分配 global_pool（学生）和 teacher_pool（教师）
#   3. 实例化 RayPPOTrainer，调用 trainer.init_workers() 和 trainer.fit()
# "$@" 允许在命令行追加额外的 Hydra override，例如:
#   NNODES=2 ./run_qwen3_8b_fsdp.sh trainer.save_path=/checkpoints/exp1
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
