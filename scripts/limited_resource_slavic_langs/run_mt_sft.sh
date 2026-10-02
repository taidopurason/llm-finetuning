#!/bin/bash
#SBATCH --job-name=train-packed-mt-sft-tokfix-4e-lr5e5
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=# TODO: Fill this
#SBATCH --output=logs/%x_%j.out
#SBATCH --exclude=nid005138,nid006369,nid005796,nid007382


LUMI_PROJECT_ID=# TODO: Fill this
export EBU_USER_PREFIX=/scratch/${LUMI_PROJECT_ID}/EasyBuild
module load LUMI
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20241125
RUN_NAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}

export SINGULARITYENV_NCCL_DEBUG=WARN

# Set environment for the app
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export HF_TOKEN=# TODO: Fill this
export HF_HOME=/scratch/${LUMI_PROJECT_ID}/hf_cache
export HF_DATASETS_CACHE=/scratch/${LUMI_PROJECT_ID}/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/${LUMI_PROJECT_ID}/hf_cache/hub

echo "Exporting variables"
EXPERIMENT_NAME=Qwen2.5-3B_${SLURM_JOB_NAME}
RUN_NAME=${EXPERIMENT_NAME}_${SLURM_JOB_ID}
WORKING_DIR=$PWD
export WANDB_NAME=${RUN_NAME}
export WANDB_PROJECT=slavic_llms

GPUS_PER_NODE=8
LR_START=5e-5
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION=2
MAX_SEQ_LEN=4096
TOTAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GPUS_PER_NODE * GRADIENT_ACCUMULATION * SLURM_NNODES))
DECAY_STEPS=768
WARMUP_STEPS=256
echo "Train samples: ${TRAIN_SAMPLES}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "Warmup steps ${WARMUP_STEPS}"
# Training and validation datasets can be HuggingFace datasets
# with the following format: <dataset_name_1>:<config_1>,<dataset_name_2>:<config_2>,...
# loads from "train" split automatically
TRAIN_PATH=your_hf_name/parallel-data-chat:combined

# loads from "validation" split automatically
VALID_PATH=de-hsb:your_hf_name/parallel-data-chat:de-hsb,de-dsb:your_hf_name/parallel-data-chat:de-dsb
OUTPUT_DIR=/scratch/${LUMI_PROJECT_ID}/sllm_experiments/checkpoints_2/${RUN_NAME}

mkdir -p ${OUTPUT_DIR}
PRE_TRAINED_MODEL="tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"
TOKENIZER_NAME="tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"

export LAUNCHER="accelerate launch \
  --config_file ${WORKING_DIR}/fsdp_train_config_gradop.yaml \
  --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
  --num_machines ${SLURM_NNODES} \
  --rdzv_backend c10d \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision bf16 \
"

export PYTHON_FILE="src/finetuning.py"
export ARGS="--model_name ${PRE_TRAINED_MODEL} \
    --run_name ${RUN_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --low_cpu_mem_usage \
    --train_split_name train \
    --valid_split_name validation \
    --train_path ${TRAIN_PATH} \
    --valid_path ${VALID_PATH} \
    --report_to wandb \
    --seed 42 \
    --max_seq_len ${MAX_SEQ_LEN} \
    --num_train_epochs 4 \
    --warmup_steps ${WARMUP_STEPS} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_final_model \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LR_START} \
    --lr_scheduler_type warmup_stable_decay \
    --lr_scheduler_kwargs {\"decay_type\":\"1-sqrt\",\"num_decay_steps\":${DECAY_STEPS}} \
    --bf16 True \
    --weight_decay 0.1 \
    --torch_dtype bfloat16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --pad_token <|vision_pad|> \
    --use_flash_attention_2"

export CMD="$LAUNCHER $PYTHON_FILE $ARGS"

echo "Running $CMD"


echo "Running script"
srun \
 singularity exec $SIF ${WORKING_DIR}/runscript.sh ${CMD}

