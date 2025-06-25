# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py
# and https://github.com/facebookresearch/llama-recipes
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import datasets
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizer, PreTrainedModel, is_datasets_available

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers.trainer_utils import seed_worker
from trl import SFTTrainer, SFTConfig

HF_DATASET_TYPE = "huggingface"
LOCAL_PACKED_HF_DATASET_TYPE = "huggingface_local_packed"
DATASET_TYPES = [HF_DATASET_TYPE, LOCAL_PACKED_HF_DATASET_TYPE]


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data path"})
    valid_path: Optional[str] = field(metadata={"help": "Validation data path"}, default=None)
    train_dataset_type: str = field(
        default=HF_DATASET_TYPE,
        metadata={"help": "Training dataset type", "choices": DATASET_TYPES}
    )
    train_split_name: str = field(default="train")
    valid_split_name: str = field(default="validation")
    valid_split_limit: Optional[int] = field(default=None)
    train_weights: Optional[str] = field(default=None, metadata={"help": "Training data weights"})
    train_reproducible_shuffle: bool = field(
        default=False,
        metadata={"help": "Use same shuffle regardless of the number of training samples "
                          "works with huggingface_local_packed"}
    )
    train_dataset_num_samples: Optional[int] = field(default=None, metadata={
        "help": "Number of training samples (for reproducible shuffle)"})
    train_dataset_shuffle_frequency: Optional[int] = field(
        default=None, metadata={"help": "Shuffle frequency for replicable shuffle (with combined dataset)"}
    )
    valid_dataset_type: str = field(
        default=HF_DATASET_TYPE,
        metadata={"help": "Training dataset type", "choices": DATASET_TYPES}
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)
    save_final_model: bool = field(default=False)
    calculate_chars_per_token: bool = field(default=False)
    stream_train_dataset: bool = field(default=False)
    stream_valid_dataset: bool = field(default=False)
    add_position_ids: bool = field(default=False)
    allow_empty_pad_token: bool = field(default=False)
    position_ids_bos_token_id: Optional[int] = field(default=None)
    position_ids_eos_token_id: Optional[int] = field(default=None)
    pad_token: Optional[str] = field(default=None)



@dataclass
class CustomTrainingArguments(SFTConfig):
    disable_dataloader_shuffle: bool = False


# https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/utils.py#L116C1-L125C43
def get_chars_per_token(dataset: Dataset, tokenizer: PreTrainedTokenizer, data_column: str, nb_examples: int = 500):
    """
    Estimate the average number of characters per token in the dataset.
    """
    logging.info("Estimating average number of characters per token in the dataset...")
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer.encode(example[data_column], truncation=False, padding=False))

    return total_characters / total_tokens


def create_position_ids(
        input_ids,
        bos_token_id=None,
        eos_token_id=None,
):
    input_ids = np.asarray(input_ids)
    n = input_ids.shape[0]
    idx = np.arange(n, dtype=np.int32)

    if bos_token_id is None:
        is_bos = np.zeros(n, dtype=bool)
    else:
        is_bos = (input_ids == bos_token_id)

    if eos_token_id is None:
        is_eos = np.zeros(n, dtype=bool)
    else:
        shifted = np.empty(n, dtype=input_ids.dtype)
        shifted[0] = -100
        shifted[1:] = input_ids[:-1]
        is_eos = (shifted == eos_token_id)

    is_reset = is_bos | is_eos
    is_reset[0] = True

    reset_idx = np.where(is_reset, idx, -1)
    last_reset = np.maximum.accumulate(reset_idx)

    pos_ids = idx - last_reset
    return pos_ids


class HFLocalPackedDataset(Dataset):
    def __init__(
            self,
            path,
            add_position_ids: bool = False,
            bos_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None
    ):
        self.dataset = load_from_disk(path)
        self.add_position_ids = add_position_ids
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if self.add_position_ids:
            if self.bos_token_id is None:
                logging.warning("BOS token ID is not set for automatic positional_id calculation.")
            if self.eos_token_id is None:
                logging.warning("EOS token ID is not set for automatic positional_id calculation.")
            if len(self.dataset) > 0:
                if "attention_mask" in self.dataset[0]:
                    logging.warning("Attention mask is present in the dataset, but positional ids will be added.")
                if "positional_ids" in self.dataset[0]:
                    logging.warning("Positional ids are already present in the dataset, new ids won't be created")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds_object = self.dataset[idx]
        example = {
            "input_ids": torch.LongTensor(ds_object["input_ids"]),
            "labels": torch.LongTensor(ds_object.get("labels", ds_object["input_ids"])),
            **{k: torch.LongTensor(ds_object[k]) for k in ["attention_mask", "position_ids"] if k in ds_object}
        }
        if self.add_position_ids and "position_ids" not in ds_object:
            example["position_ids"] = torch.LongTensor(
                create_position_ids(
                    ds_object["input_ids"], bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
                )
            )
        return example


def create_dataset(
        path: str,
        tokenizer: PreTrainedTokenizer,
        args: ScriptArguments,
        training_args: CustomTrainingArguments,
        split: str = "train",
        streaming: bool = False,
        dataset_type: str = HF_DATASET_TYPE,
):
    logging.info(f"Loading dataset from: {path}")
    if dataset_type == HF_DATASET_TYPE:
        if ":" in path:
            ds_name, lang = path.split(":")
            return load_dataset(ds_name, lang, streaming=streaming, split=split)
        return load_dataset(path, streaming=streaming, split=split)
    if dataset_type == LOCAL_PACKED_HF_DATASET_TYPE:
        from training_datasets import DatasetWrapper, CombinedDatasetWrapper
        training_paths = path.split(",")
        if args.train_reproducible_shuffle and not training_args.disable_dataloader_shuffle:
            raise ValueError("Replicable shuffle is enabled but dataloader shuffle is not disabled "
                             "use --disable_dataloader_shuffle")

        if args.position_ids_bos_token_id is None:
            bos_token_id = tokenizer.bos_token_id
        else:
            bos_token_id = args.position_ids_bos_token_id

        if args.position_ids_eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        else:
            eos_token_id = args.position_ids_bos_token_id

        if len(training_paths) == 1:
            ds = HFLocalPackedDataset(
                training_paths[0],
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                add_position_ids=args.add_position_ids,
            )
            if not args.train_reproducible_shuffle or split != "train":
                return ds

            return DatasetWrapper(
                ds,
                n_samples=args.train_dataset_num_samples,
                seed=training_args.seed,
            )

        dss = [
            HFLocalPackedDataset(
                p,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                add_position_ids=args.add_position_ids,
            )
            for p in training_paths
        ]
        weights = [float(w) for w in args.train_weights.split(",")] if args.train_weights is not None else None
        return CombinedDatasetWrapper(
            datasets=dss,
            weights=weights,
            n_samples=args.train_dataset_num_samples,
            seed=training_args.seed,
            shuffle_frequency=args.train_dataset_shuffle_frequency,
        )

    raise ValueError("Unsupported dataset type")


def create_train_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: CustomTrainingArguments):
    return create_dataset(
        tokenizer=tokenizer, args=args, training_args=training_args, path=args.train_path, split=args.train_split_name,
        streaming=args.stream_train_dataset, dataset_type=args.train_dataset_type
    )


def data_generator(iterator):
    yield from iterator


def _create_valid_dataset(
        path: str,
        tokenizer,
        args: ScriptArguments,
        training_args: CustomTrainingArguments,
):
    with PartialState().local_main_process_first():
        valid_ds = create_dataset(
            tokenizer=tokenizer, args=args, training_args=training_args, path=path, split=args.valid_split_name,
            streaming=args.stream_valid_dataset, dataset_type=args.valid_dataset_type
        )
        if args.valid_split_limit is not None:
            valid_ds = valid_ds.take(args.valid_split_limit)
        if args.stream_valid_dataset:
            valid_ds = datasets.Dataset.from_generator(
                data_generator, gen_kwargs={"iterator": valid_ds}, features=valid_ds.features
            )
        return valid_ds


def create_valid_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: CustomTrainingArguments):
    logging.info(f"Creating validation dataset from: {args.valid_path}")
    if args.valid_path is None:
        return None

    if len(args.valid_path.split(",")) == 1:
        return _create_valid_dataset(args.valid_path, tokenizer, args, training_args)

    valid_datasets = {}
    for path in args.valid_path.split(","):
        valid_name, *valid_path = path.split(":")
        valid_path = ":".join(valid_path)
        valid_datasets[valid_name] = _create_valid_dataset(valid_path, tokenizer, args, training_args)
    return valid_datasets


def create_and_prepare_model(
        args: ScriptArguments, training_args: CustomTrainingArguments
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if args.torch_dtype is None or args.torch_dtype == "auto":
        torch_dtype = args.torch_dtype
    else:
        torch_dtype = getattr(torch, args.torch_dtype)

    model_kwargs = {}

    if args.use_flash_attention_2:
        logging.info("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        use_cache=not training_args.gradient_checkpointing,
        trust_remote_code=True,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if args.tokenizer_name is None else args.tokenizer_name,
        model_max_length=training_args.max_seq_length,
        padding_side="right",
    )

    if args.pad_token is not None:
        logging.info(f"Setting pad token to {args.pad_token}")
        tokenizer.pad_token = args.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.padding_idx = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id

    if tokenizer.pad_token is None and not args.allow_empty_pad_token:
        logging.warning("Pad token is not set, using EOS token as pad token.")
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Using pad token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    logging.info(f"Using eos token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    return model, tokenizer


class CustomSFTTrainer(SFTTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if isinstance(self.args, CustomTrainingArguments) and self.args.disable_dataloader_shuffle:
            logging.info("Disabling training dataset shuffling")
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(script_args: ScriptArguments, training_args: CustomTrainingArguments):
    torch.cuda.manual_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer = create_and_prepare_model(
        script_args, training_args
    )
    model.config.use_cache = False

    train_dataset = create_train_dataset(tokenizer, script_args, training_args)
    eval_dataset = create_valid_dataset(tokenizer, script_args, training_args)

    if script_args.calculate_chars_per_token:
        chars_per_token = get_chars_per_token(train_dataset, tokenizer, "text")
        training_args.chars_per_token = chars_per_token
        logging.info(f"Estimated chars per token: {chars_per_token}")

    logging.info(f"Max sequence length: {tokenizer.model_max_length}")
    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.accelerator.print(f"{trainer.model}")
    print_trainable_parameters(trainer.model)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if script_args.save_final_model:
        final_out = os.path.join(training_args.output_dir, "last_checkpoint")
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
            trainer.save_model(final_out)
        else:
            trainer.model.save_pretrained(final_out)
        tokenizer.save_pretrained(final_out)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    parser = HfArgumentParser([ScriptArguments, CustomTrainingArguments])
    args, training_args = parser.parse_args_into_dataclasses()
    main(
        script_args=args,
        training_args=training_args,
    )
