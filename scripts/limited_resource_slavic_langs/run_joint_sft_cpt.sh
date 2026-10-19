#!/bin/bash
#SBATCH --job-name=train-packed-hsb-dsb-inst-para2-wsd-4e
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=
#SBATCH --output=logs/%x_%j.out
#SBATCH --exclude=nid005138,nid006369,nid005796,nid007382


# TODO: Add your LUMI project ID here
LUMI_PROJECT_ID=
export EBU_USER_PREFIX=/scratch/${LUMI_PROJECT_ID}/EasyBuild
module load LUMI
module load PyTorch/2.5.1-rocm-6.2.3-python-3.12-singularity-20241125
RUN_NAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}

export SINGULARITYENV_NCCL_DEBUG=WARN

# Set environment for the app
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

#TODO: Add your Hugging Face token here if you want to use private models or datasets
export HF_TOKEN=""
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
LR_START=1e-4
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION=1
MAX_SEQ_LEN=4096
EVAL_FREQ=256
SAVE_FREQ=1024
DECAY_STEPS=768
WARMUP_STEPS=256
TOTAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GPUS_PER_NODE * GRADIENT_ACCUMULATION * SLURM_NNODES))
TOTAL_BATCH_SIZE_TOKENS=$((TOTAL_BATCH_SIZE * MAX_SEQ_LEN))
TRAIN_SAMPLES=292449
echo "Train samples: ${TRAIN_SAMPLES}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "Total batch size (tokens): ${TOTAL_BATCH_SIZE_TOKENS}"
echo "Warmup steps ${WARMUP_STEPS}"


DATA_PATH="/data/packed${MAX_SEQ_LEN}"
dsb_mono_path="${DATA_PATH}/sorbian__combined.mono_docs.dsb/train"
hsb_mono_path="${DATA_PATH}/sorbian__combined.mono_docs.hsb/train"

# instruction tuned
inst_path="${DATA_PATH}/magpie_aya_euroblocks_oasst_flan_lf/train"

# x -> sorbian
lr_para_path_v3_tgt="${DATA_PATH}/trinst-v3/lr-parallel-chat-100-target-v3/train"

# sorbian -> x
lr_para_path_v3_src="${DATA_PATH}/trinst-v3/lr-parallel-chat-100-source-v3/train"

TRAIN_PATH="${dsb_mono_path},${hsb_mono_path},${inst_path},${lr_para_path_v3_tgt},${lr_para_path_v3_src}"
TRAIN_WEIGHTS='0.04149783381033958,0.3567801565401147,0.2937572021104534,0.25402035910534826,0.053944448433744005'
echo "Training path: ${TRAIN_PATH}"
echo "Training weights: ${TRAIN_WEIGHTS}"

VALID_PATH="dsb_Latn:your_hf_name/sorbian-mono-data:combined.dsb.heldout,hsb_Latn:your_hf_name/sorbian-mono-data:combined.hsb.heldout,deu_Latn:HuggingFaceFW/fineweb-2:deu_Latn"
OUTPUT_DIR=/scratch/${LUMI_PROJECT_ID}/sllm_experiments/checkpoints_2/${RUN_NAME}

mkdir -p ${OUTPUT_DIR}
PRE_TRAINED_MODEL="Qwen/Qwen2.5-3B-Instruct"

export LAUNCHER="accelerate launch \
  --config_file ${WORKING_DIR}/fsdp_train_config_gradop.yaml \
  --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
  --num_machines ${SLURM_NNODES} \
  --rdzv_backend c10d \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision bf16 \
"

TOKENIZER_NAME="tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"
export PYTHON_FILE="src/continued_pretraining.py"
export ARGS="--model_name ${PRE_TRAINED_MODEL} \
    --run_name ${RUN_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --low_cpu_mem_usage \
    --valid_split_limit 10000 \
    --stream_valid_dataset True \
    --train_split_name train \
    --valid_split_name test \
    --eval_steps ${EVAL_FREQ} \
    --save_steps ${SAVE_FREQ} \
    --train_path ${TRAIN_PATH} \
    --valid_path ${VALID_PATH} \
    --train_weights ${TRAIN_WEIGHTS} \
    --train_reproducible_shuffle \
    --disable_dataloader_shuffle \
    --train_dataset_num_samples ${TRAIN_SAMPLES} \
    --train_dataset_type huggingface_local_packed \
    --valid_dataset_type huggingface \
    --eval_packing True \
    --dataset_text_field text \
    --report_to wandb \
    --seed 42 \
    --max_seq_len ${MAX_SEQ_LEN} \
    --num_train_epochs 1 \
    --warmup_steps ${WARMUP_STEPS} \
    --evaluation_strategy steps \
    --save_strategy steps \
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

