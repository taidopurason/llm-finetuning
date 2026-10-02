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
  ],
  "src_lang": "de",
  "tgt_lang": "dsb"
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
