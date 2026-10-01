import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import warnings
import numpy as np
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizer, PreTrainedModel, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from torch.utils.data import Dataset
from transformers.utils import PaddingStrategy


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data path"})
    valid_path: Optional[str] = field(metadata={"help": "Validation data path"}, default=None)
    train_split_name: str = field(default="train")
    valid_split_name: str = field(default="validation")
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
    )
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)
    save_final_model: bool = field(default=False)
    allow_empty_pad_token: bool = field(default=False)
    pad_token: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = None
    dataset_text_field: str = field(default="text")


class ChatTemplateDataset(Dataset):
    def __init__(
            self,
            dataset: list,
            tokenizer: PreTrainedTokenizer,
            instruction_template: str = "<|im_start|>user\n",
            response_template: str = "<|im_start|>assistant\n",
            return_list: bool = False,
            remove_starting_newline: bool = True
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template
        self.ignore_index = -100
        self.return_list = return_list
        self.remove_starting_newline = remove_starting_newline

    def __len__(self) -> int:
        return len(self.dataset)

    def process_example(self, example: dict) -> str:
        if self.remove_starting_newline:
            messages = [
                {"role": x["role"], "content": x["content"].lstrip("\n")}
                for x in example["messages"]
            ]
        else:
            messages = example["messages"]
        return self.tokenizer.apply_chat_template(messages, tokenize=True)

    def __getitem__(self, index: int) -> dict:
        input_ids = torch.LongTensor(self.process_example(self.dataset[index]))
        labels = input_ids.clone()

        if self.instruction_template is None:
            response_token_ids_start_idx = None

            for idx in np.where(labels == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                        self.response_token_ids
                        == labels[idx: idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the following instance: "
                    f"{self.tokenizer.decode(input_ids)}. This instance will be ignored in loss "
                    "calculation. Note, if this happens often, consider increasing the `max_length`.",
                    UserWarning,
                )
                labels[:] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                labels[:response_token_ids_end_idx] = self.ignore_index
        else:
            response_token_ids_idxs = []
            human_token_ids_idxs = []
            for assistant_idx in np.where(labels == self.response_token_ids[0])[0]:
                # find the indexes of the start of a response.
                if (
                        self.response_token_ids
                        == labels[assistant_idx: assistant_idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

            if len(response_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the following instance: "
                    f"{self.tokenizer.decode(input_ids)}. This instance will be ignored in loss "
                    "calculation. Note, if this happens often, consider increasing the `max_length`.",
                    UserWarning,
                )
                labels[:] = self.ignore_index

            human_token_ids = self.instruction_token_ids
            for human_idx in np.where(labels == human_token_ids[0])[0]:
                # find the indexes of the start of a human answer.
                if human_token_ids == labels[human_idx: human_idx + len(human_token_ids)].tolist():
                    human_token_ids_idxs.append(human_idx)

            if len(human_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                    f"{self.tokenizer.decode(input_ids)}. This instance will be ignored in loss "
                    "calculation. Note, if this happens often, consider increasing the `max_length`.",
                    UserWarning,
                )
                labels[:] = self.ignore_index
            if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
            ):
                human_token_ids_idxs = [0] + human_token_ids_idxs

            for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                # Make pytorch loss function ignore all non response tokens
                if idx != 0:
                    labels[start:end] = self.ignore_index
                else:
                    labels[:end] = self.ignore_index

            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                labels[human_token_ids_idxs[-1]:] = self.ignore_index

        attn_mask = torch.ones_like(input_ids, dtype=torch.long)

        if self.return_list:
            return {
                "input_ids": input_ids.tolist(),
                "labels": labels.tolist(),
                "attention_mask": attn_mask.tolist(),
            }

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
        }


def create_dataset(
        path: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        streaming: bool = False,
):
    logging.info(f"Loading dataset from: {path}")
    if ":" in path:
        ds_name, lang = path.split(":")
        ds = load_dataset(ds_name, lang, streaming=streaming, split=split)
    else:
        ds = load_dataset(path, streaming=streaming, split=split)

    return ChatTemplateDataset(
        ds,
        tokenizer,
        return_list=True,
        remove_starting_newline=True
    )


def create_train_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments):
    return create_dataset(
        tokenizer=tokenizer, path=args.train_path, split=args.train_split_name,
    )


def create_valid_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments):
    logging.info(f"Creating validation dataset from: {args.valid_path}")
    if args.valid_path is None:
        return None

    if len(args.valid_path.split(",")) == 1:
        return create_dataset(
            tokenizer=tokenizer, path=args.valid_path, split=args.valid_split_name,
        )

    valid_datasets = {}
    for path in args.valid_path.split(","):
        valid_name, *valid_path = path.split(":")
        valid_path = ":".join(valid_path)
        valid_datasets[valid_name] = create_dataset(
            tokenizer=tokenizer, path=valid_path, split=args.valid_split_name,
        )
    return valid_datasets


def create_and_prepare_model(
        args: ScriptArguments, training_args: TrainingArguments
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
    model.config.use_cache = False

    tokenizer_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name
    logging.info(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length=args.max_seq_length,
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


def main(script_args: ScriptArguments, training_args: TrainingArguments):
    torch.cuda.manual_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer = create_and_prepare_model(
        script_args, training_args
    )

    train_dataset = create_train_dataset(tokenizer, script_args)
    eval_dataset = create_valid_dataset(tokenizer, script_args)

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=PaddingStrategy.LONGEST,
        max_length=args.max_seq_length
    )

    logging.info(f"Max sequence length: {tokenizer.model_max_length}")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
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

    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    args, training_args = parser.parse_args_into_dataclasses()
    main(
        script_args=args,
        training_args=training_args,
    )
