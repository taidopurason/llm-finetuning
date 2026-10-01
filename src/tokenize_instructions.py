import logging
import os
import sys
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


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

    def process_example(self, example: dict) -> List[int]:
        if len(example["messages"]) == 0:
            raise ValueError("Encountered an empty example")

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

        if self.return_list:
            return {
                "input_ids": input_ids.tolist(),
                "labels": labels.tolist(),
            }

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def data_generator(iterator):
    yield from iterator


def tokenize_instructions(
        tokenizer_name: str,
        dataset_name: str,
        output_dir: str,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
        max_in_memory_size: Optional[int] = None,
        local_dataset: bool = True,
        config_name: Optional[str] = None,
        dataset_split: str = "train",
        max_length: Optional[int] = None,
):
    if max_in_memory_size is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = max_in_memory_size

    print(f"Loading dataset from {dataset_name}")
    if local_dataset:
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(
            dataset_name, name=config_name, split=dataset_split
        )

    print(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit is not None:
        print(f"Limiting dataset to {limit} examples")
        ds = ds.take(limit)

    chat_ds = ChatTemplateDataset(
        ds,
        tokenizer,
        return_list=True,
        remove_starting_newline=True
    )

    # Preview
    if len(chat_ds) > 0:
        example = chat_ds[0]
        toks = []
        for input_id, label in zip(example["input_ids"], example["labels"]):
            toks.append(tokenizer.decode(input_id))
            if label == -100:
                toks.append("[M]")
        print("### Preview example ([M] shows masked token):")
        print("".join(toks))

    print(f"Tokenizing dataset with {len(chat_ds)} examples")
    tokenized_ds = datasets.Dataset.from_generator(
        data_generator, gen_kwargs={"iterator": chat_ds}
    )

    print(f"Number of examples: {len(tokenized_ds)}")
    if max_length is not None:
        tokenized_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) < max_length)
        print(f"Number of examples after removing too long examples: {len(tokenized_ds)}")

    tokenized_ds.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(tokenize_instructions)
