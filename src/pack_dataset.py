import logging
import os
import random
import sys
import warnings
from typing import List, Optional

import psutil
import numpy as np
from torch.utils.data import IterableDataset
import datasets
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset, Features, Sequence, Value

# Packing implementation adapted from TRL
# https://github.com/huggingface/trl/blob/4871c82b0cd1caae72522182f9171ea069481250/trl/trainer/utils.py#L546
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The processor used for processing the data.
        dataset (`dataset.Dataset`):
            Dataset with text files.
        dataset_text_field (`str` or `None`, *optional*, defaults to `None`):
            Name of the field in the dataset that contains the text. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        formatting_func (`Callable`, *optional*):
            Function that formats the text before tokenization. Usually it is recommended to follow a certain
            pattern such as `"### Question: {question} ### Answer: {answer}"`. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        infinite (`bool`, *optional*, defaults to `False`):
            If True the iterator is reset after dataset reaches end else stops.
        seq_length (`int`, *optional*, defaults to `1024`):
            Length of token sequences to return.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of token sequences to keep in buffer.
        chars_per_token (`int`, *optional*, defaults to `3.6`):
            Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id (`int`, *optional*, defaults to `0`):
            Id of the end of sequence token if the passed tokenizer does not have an EOS token.
        shuffle (`bool`, *optional*, defaults to `True`)
            Shuffle the examples before they are returned
        append_concat_token (`bool`, *optional*, defaults to `True`)
            If true, appends `eos_token_id` at the end of each sample being packed.
        add_special_tokens (`bool`, *optional*, defaults to `True`)
            If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
            self,
            tokenizer,
            dataset,
            dataset_text_field=None,
            formatting_func=None,
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            eos_token_id=0,
            shuffle=True,
            append_concat_token=True,
            add_special_tokens=True,
            add_position_ids=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        self.add_position_ids = add_position_ids

        if dataset_text_field is not None and formatting_func is not None:
            warnings.warn(
                "Only one of `dataset_text_field` and `formatting_func` should be provided. "
                "Ignoring `dataset_text_field` and using `formatting_func`.",
                UserWarning,
            )

        if formatting_func is not None:
            self.formatting_func = formatting_func
        elif dataset_text_field is not None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:  # neither is provided
            raise ValueError("Either `dataset_text_field` or `formatting_func` should be provided.")

        self.pretokenized = False
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names is not None and "input_ids" in column_names:
            self.pretokenized = True
            # since the dataset is tokenized, the unit of buffer size should be tokens
            self.max_buffer_size = seq_length * num_of_sequences

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            if self.pretokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(
                    buffer, add_special_tokens=self.add_special_tokens, truncation=False
                )["input_ids"]

            if self.add_position_ids:
                yield from self._pack_with_position_ids(tokenized_inputs)
            else:
                yield from self._pack(tokenized_inputs)

    def _pack(self, tokenized_inputs):
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            if self.append_concat_token:
                tokenized_input = tokenized_input + [self.concat_token_id]
            all_token_ids.extend(tokenized_input)
        examples = []
        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i: i + self.seq_length]
            if len(input_ids) == self.seq_length:
                examples.append({"input_ids": input_ids})
        if self.shuffle:
            random.shuffle(examples)
        for example in examples:
            self.current_size += 1
            yield example

    def _pack_with_position_ids(self, tokenized_inputs):
        input_position_ids = [list(range(len(x))) for x in tokenized_inputs]
        all_token_ids = []
        all_position_ids = []
        for tokenized_input, position_ids in zip(tokenized_inputs, input_position_ids):
            if self.append_concat_token:
                tokenized_input = tokenized_input + [self.concat_token_id]
                position_ids = position_ids + [len(position_ids)]
            all_token_ids.extend(tokenized_input)
            all_position_ids.extend(position_ids)
        examples = []
        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i: i + self.seq_length]
            input_pos_ids = all_position_ids[i: i + self.seq_length]
            if len(input_ids) == self.seq_length:
                examples.append({"input_ids": input_ids, "position_ids": input_pos_ids})
        if self.shuffle:
            random.shuffle(examples)
        for example in examples:
            self.current_size += 1
            yield example


def data_generator(iterator, features=("input_ids", "labels")):
    for x in iterator:
        yield {feature: x[feature] for feature in features}


def group_texts(
        input_ids: List[List[int]],
        labels: Optional[List[List[int]]] = None,
        sequence_length: int = 2048,
        concat_token_id: Optional[int] = None,
        add_position_ids: bool = False,
) -> dict:
    position_ids = None
    if concat_token_id is None:
        concatenated_input_ids = np.concatenate(input_ids)
        if labels is not None:
            concatenated_labels = np.concatenate(labels)
        if add_position_ids:
            position_ids = np.concatenate([list(range(len(x))) for x in input_ids])
    else:
        concatenated_input_ids = np.concatenate([x + [concat_token_id] for x in input_ids])
        if labels is not None:
            concatenated_labels = np.concatenate([x + [concat_token_id] for x in labels])
        if add_position_ids:
            position_ids = np.concatenate([list(range(len(x) + 1)) for x in input_ids])
    total_length = len(concatenated_input_ids)

    extra_fields = {}
    if add_position_ids:
        assert position_ids is not None
        extra_fields["position_ids"] = [
            position_ids[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ]

    if labels is not None:
        extra_fields["labels"] = [
            concatenated_labels[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ]

    return {
        "input_ids": [
            concatenated_input_ids[i: i + sequence_length] for i in
            range(0, total_length - sequence_length + 1, sequence_length)
        ],
        **extra_fields,
    }


def pack_dataset_fast(ds, tokenizer, seq_length, num_proc=-1, append_concat_token=True, add_position_ids=False,
                      add_labels=False):
    if num_proc == -1:
        num_proc = psutil.cpu_count()

    concat_token_id = getattr(tokenizer, 'eos_token_id', None)
    if concat_token_id is None and append_concat_token:
        raise ValueError("concat_token_id is not present")
    if not append_concat_token:
        logging.warning("Concat token will not be added")
        concat_token_id = None

    extra_features = {}
    if add_position_ids:
        extra_features["position_ids"] = Sequence(feature=Value(dtype="int64"), length=seq_length)

    extra_fields = []
    if add_labels:
        extra_fields.append("labels")
        extra_features["labels"] = Sequence(feature=Value(dtype="int64"), length=seq_length)

    return ds.map(
        lambda x, *y: group_texts(x, *y, concat_token_id=concat_token_id, sequence_length=seq_length,
                                  add_position_ids=add_position_ids),
        input_columns=["input_ids", *extra_fields],
        remove_columns=ds.column_names,
        batched=True,
        features=Features({
            "input_ids": Sequence(feature=Value(dtype="int64"), length=seq_length),
            **extra_features
        }),
        num_proc=num_proc,
    )


def main(
        tokenizer_name: str,
        dataset_name: str,
        output_dir: str,
        seq_length: int,
        local_dataset: bool = True,
        dataset_split: str = "train",
        append_concat_token: bool = True,
        max_in_memory_size: Optional[int] = None,
        shuffle: bool = False,
        limit_examples: Optional[int] = None,
        seed: int = 42,
        fast_packing: bool = False,
        workers: int = -1,
        add_position_ids: bool = False,
        add_labels: bool = False,
):
    if max_in_memory_size is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = max_in_memory_size
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if local_dataset:
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(dataset_name, split=dataset_split)

    logging.info(f"Loaded dataset {dataset_name} with {len(ds)} examples.")

    if shuffle:
        logging.info(f"Shuffling the dataset.")
        ds = ds.shuffle(seed=seed)

    if limit_examples is not None:
        ds = ds.take(limit_examples)

    logging.info(f"Starting packing dataset with {len(ds)} examples saving to {output_dir}.")

    if fast_packing:
        logging.warning("Using fast packing which might lose some examples.")
        packed_dataset = pack_dataset_fast(
            ds, tokenizer, seq_length,
            append_concat_token=append_concat_token, num_proc=workers, add_position_ids=add_position_ids,
            add_labels=add_labels,
        )
    else:
        if workers != -1:
            logging.warning("Num workers will have no effect on slow packing")
        ds_const = ConstantLengthDataset(
            tokenizer=tokenizer,
            dataset=ds,
            formatting_func=lambda x: x["input_ids"],
            shuffle=False,
            seq_length=seq_length,
            append_concat_token=append_concat_token,
            add_position_ids=add_position_ids,
        )

        extra_features = []
        if add_position_ids:
            extra_features.append("position_ids")

        packed_dataset = datasets.Dataset.from_generator(
            data_generator, gen_kwargs={"iterator": ds_const, "features": ("input_ids", *extra_features)}
        )
        logging.info(f"Finished packing the dataset with {len(packed_dataset)} packed examples.")
    logging.info(f"Example: {packed_dataset[0]}")
    packed_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    fire.Fire(main)
