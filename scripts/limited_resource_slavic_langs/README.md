# Finetuning on your own data

`src/finetuning.py` takes a Hugging Face dataset as input (`dataset:configuration`). It reads each row's `messages`, applies the tokenizer's chat template, and creates token IDs and labels for assistant-response training. The language fields are optional metadata.

## Create finetuning data

Each example needs a `messages` list with `role` and `content`. Example:

```json
{
  "messages": [
    {
      "content": "You are a helpful translator. Translate the following German sentences into Lower Sorbian. Answer with only the Lower Sorbian text.",
      "role": "system"
    },
    {
      "content": "Sein Wissen über die Trachten und das richtige Ankleiden möchte er auch in den Workshops des neu gegründeten Forums weitergeben.",
      "role": "user"
    },
    {
      "content": "Swóju wědu wó drastwach a pšawem woblekanju co teke we workshopach nowo załožonego foruma dalej dawaś.",
      "role": "assistant"
    }
  ]
}
```

## Upload to Hugging Face

Install `datasets` and log in with `hf auth login`. Build separate lists of training and validation examples using the format above:

```python
from datasets import Dataset, DatasetDict

# Pseudo examples: replace the placeholder text with real sentence pairs.
train_examples = [
    {"messages": [
        {"role": "system", "content": "You are a helpful translator. Translate the following German sentences into Lower Sorbian. Answer with only the Lower Sorbian text."},
        {"role": "user", "content": "<German sentence 1>"},
        {"role": "assistant", "content": "<Lower Sorbian translation 1>"},
    ], "src_lang": "de", "tgt_lang": "dsb"},
    {"messages": [
        {"role": "system", "content": "You are a helpful translator. Translate the following German sentences into Lower Sorbian. Answer with only the Lower Sorbian text."},
        {"role": "user", "content": "<German sentence 2>"},
        {"role": "assistant", "content": "<Lower Sorbian translation 2>"},
    ], "src_lang": "de", "tgt_lang": "dsb"},
]
# Use separate, held-out sentence pairs for validation in the same format.
validation_examples = [
    {"messages": [
        {"role": "system", "content": "You are a helpful translator. Translate the following German sentences into Lower Sorbian. Answer with only the Lower Sorbian text."},
        {"role": "user", "content": "<Held-out German sentence>"},
        {"role": "assistant", "content": "<Held-out Lower Sorbian translation>"},
    ], "src_lang": "de", "tgt_lang": "dsb"},
]

dataset = DatasetDict({
    "train": Dataset.from_list(train_examples),
    "validation": Dataset.from_list(validation_examples),
})
dataset.push_to_hub(
    "your_hf_name/your_dataset",
    config_name="combined",  # e.g. "de-hsb" for a language pair
    private=True,
)
```

## Run machine translation finetuning

This script performs machine translation finetuning. It does not reproduce the training of the final [tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb model](https://huggingface.co/tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb), which used continued pretraining with Sorbian monolingual/parallel data and general instruction data.

Install the training dependencies from [requirements.txt](../../requirements.txt), from the repository root:

```sh
pip install -r requirements.txt
```

1. Upload your dataset as above.
2. In `run_mt_sft.sh`, set `TRAIN_PATH=your_hf_name/your_dataset:combined` and `VALID_PATH=your_hf_name/your_dataset:combined`; replace `combined` with your configuration name. Set the model/tokenizer and training settings.
3. Fill in the LUMI account/project and authentication settings; update the environment path in `runscript.sh`.
4. Run `sbatch scripts/limited_resource_slavic_langs/run_mt_sft.sh` from the repository root. Set the launcher's `runscript.sh` path to `${WORKING_DIR}/scripts/limited_resource_slavic_langs/runscript.sh` first.

These launch scripts are for the LUMI supercomputer (Slurm). You may need to modify the resource settings, modules/container, paths, and launch command for your environment. They require the referenced Accelerate configuration and environment. Use a model with a chat template compatible with the script's Qwen-style user/assistant markers.

## Joint instruction tuning and continued pretraining

[run_joint_sft_cpt.sh](run_joint_sft_cpt.sh) runs `src/continued_pretraining.py` on a mixture of locally tokenized and packed datasets.

Instruction data (including translation SFT) uses the same `messages` format shown above. Pretraining data uses a `text` field, as in [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2):

```json
{"text": "A document in the target language..."}
```

1. Fill in `SCRIPT_PATH` (the repository's `src` directory), `OUT_PATH`, `ds_name`, and `config_name` in the preprocessing scripts. Use the same tokenizer and sequence length for all datasets and training.
2. Run [preprocess_instruct.sh](data_preprocessing/preprocess_instruct.sh) for each instruction/translation dataset: tokenize with prompt masking, then pack while preserving labels. Run [preprocess_mono.sh](data_preprocessing/preprocess_mono.sh) for each text dataset: tokenize, then pack. Keep each dataset separate; mixing happens during training.
3. Set `TRAIN_PATH` in `run_joint_sft_cpt.sh` to the resulting packed dataset directories. Set `TRAIN_WEIGHTS` in the same order, and configure `VALID_PATH` for held-out Hugging Face data (`test` split by default).
4. Choose the training budget using packed sample counts: for dataset `i` with `N_i` packed samples and desired `E_i` passes, use `B_i = N_i * E_i`, `TRAIN_SAMPLES = sum(B_i)`, and weight `i = B_i / TRAIN_SAMPLES`. With `--num_train_epochs 1`, approximate optimizer steps are `ceil(TRAIN_SAMPLES / TOTAL_BATCH_SIZE)`. Adjust warmup/decay steps to match this budget.
5. Set the LUMI account/project, environment, and paths, including `PYTHON_FILE`, the Accelerate configuration, and the container wrapper (`runscript_sm.sh`). From the repository root, submit:

```sh
sbatch scripts/limited_resource_slavic_langs/run_joint_sft_cpt.sh
```

Install the dependencies from `requirements.txt` as above. This launcher also targets LUMI and may need changes for your environment.

## Citation

From the [model card](https://huggingface.co/tartuNLP/Qwen2.5-3B-Instruct-hsb-dsb#citation-info) (abbreviated BibTeX):

```bibtex
@inproceedings{purason-fishel-2025-tartunlp,
  title = "{T}artu{NLP} at {WMT}25 {LLM}s with Limited Resources for {S}lavic Languages Shared Task",
  author = "Purason, Taido and Fishel, Mark",
  year = "2025",
  url = "https://aclanthology.org/2025.wmt-1.88/",
  doi = "10.18653/v1/2025.wmt-1.88"
}
```
