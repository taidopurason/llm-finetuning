# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py
# and https://github.com/facebookresearch/llama-recipes
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple

import datasets
import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizer, get_polynomial_decay_schedule_with_warmup, \
    PreTrainedModel, is_datasets_available

from torch.utils.data import Dataset, DataLoader
from transformers.trainer_utils import seed_worker
from trl import SFTTrainer, SFTConfig

HF_DATASET_TYPE = "huggingface"
LOCAL_PACKED_HF_DATASET_TYPE = "huggingface_local_packed"
MEGATRON_DATASET_TYPE = "megatron"
DATASET_TYPES = [HF_DATASET_TYPE, MEGATRON_DATASET_TYPE, LOCAL_PACKED_HF_DATASET_TYPE]


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
    megatron_path_to_cache: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(SFTConfig):
    scheduler_lr_end: float = None
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


class HFLocalPackedDataset(Dataset):
    def __init__(self, path):
        self.dataset = load_from_disk(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds_object = self.dataset[idx]
        return {
            "input_ids": torch.LongTensor(ds_object["input_ids"]),
            "labels": torch.LongTensor(ds_object.get("labels", ds_object["input_ids"])),
        }

def create_dataset(
        path: str,
        tokenizer,
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
        return HFLocalPackedDataset(path)
    if dataset_type == MEGATRON_DATASET_TYPE:
        from megatron_dataset import load_megatron_dataset, MegatronDatasetWrapperHF
        tokenizer_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name
        max_seq_length = training_args.max_seq_length
        with PartialState().local_main_process_first():
            ds = MegatronDatasetWrapperHF(load_megatron_dataset(
                path,
                tokenizer,
                tokenizer_name,
                max_seq_length,
                seed=training_args.seed,
                is_built_on_rank=lambda: True,
                path_to_cache=args.megatron_path_to_cache
            ))
            return ds

    raise ValueError("Unsupported dataset type")


def create_train_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: CustomTrainingArguments):
    return create_dataset(
        tokenizer=tokenizer, args=args, training_args=training_args, path=args.train_path, split=args.train_split_name,
        streaming=args.stream_train_dataset, dataset_type=args.train_dataset_type
    )


def create_valid_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, training_args: CustomTrainingArguments):
    if args.valid_path is None:
        return None

    return create_dataset(
        tokenizer=tokenizer, args=args, training_args=training_args, path=args.valid_path, split=args.valid_split_name,
        streaming=False, dataset_type=args.valid_dataset_type
    )


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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, lr_init: float = 1,
        lr_end: float = 0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    relative_lr_end = lr_end / lr_init
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (
            1 - relative_lr_end) + relative_lr_end


def get_cosine_schedule_with_warmup_end_lr(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        lr_end: float = 0,
):
    lr_init = optimizer.defaults["lr"]
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        lr_init=lr_init,
        lr_end=lr_end,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomSFTTrainer(SFTTrainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if (
                self.lr_scheduler is None and
                isinstance(self.args, CustomTrainingArguments) and
                self.args.scheduler_lr_end is not None
        ):
            logging.info(
                f"Using {self.args.lr_scheduler_type} with learning rate with end lr {self.args.scheduler_lr_end}"
            )
            if self.args.lr_scheduler_type == "polynomial":
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            elif self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup_end_lr(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            else:
                raise ValueError(f"lr scheduler {self.args.lr_scheduler_type} not supported with scheduler_lr_end")
            self._created_lr_scheduler = True
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if self.args.disable_dataloader_shuffle:
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


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
