# TODO: fill these out
SCRIPT_PATH=""
OUT_PATH=""
# dataset to process
config_name=""
ds_name=""


model_name="tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb"
seq_len=4096
TOKENIZED_OUT="${OUT_PATH}/${model_name}/sorbian__${config_name}"
PACKED_OUT="${OUT_PATH}/${model_name}/pack${seq_len}/sorbian__${config_name}"
mkdir -p ${OUT_PATH} ${TOKENIZED_OUT} ${PACKED_OUT}

python -u ${SCRIPT_PATH}/tokenize_dataset.py \
    --tokenizer_name="${model_name}" \
    --dataset_name="${ds_name}" \
    --config_name=${config_name} \
    --output_dir=${TOKENIZED_OUT}/train \
    --workers=32 \
    --seed=42 \
    --max_in_memory_size=24_000_000_000


python -u ${SCRIPT_PATH}/pack_dataset.py \
    --tokenizer_name="${model_name}" \
    --dataset_name=${TOKENIZED_OUT}/train \
    --output_dir=${PACKED_OUT}/train \
    --seq_length=${seq_len} \
    --local_dataset=True \
    --shuffle=True \
    --fast_packing=True \
    --seed=42 \
    --workers=32 \
    --max_in_memory_size=24_000_000_000

