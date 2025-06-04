import logging
import math
from random import random
from typing import Optional, List, Tuple
import gc
from torch.utils.data import Dataset
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def build_document_index(
        n_samples: int, weights: np.ndarray
) -> np.ndarray:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights

    Adapted from Nanotron
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):
        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index


def shuffle_document_indices(doc_indices: np.ndarray, chunk_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled_indices = doc_indices.copy()

    for i in range(0, len(doc_indices), chunk_size):
        end = min(i + chunk_size, len(doc_indices))
        rng.shuffle(shuffled_indices[i:end])

    return shuffled_indices


def create_dataset_indices(n_samples: int, dataset_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_epochs = math.ceil(n_samples / dataset_size)
    indices = np.empty(dataset_size * n_epochs, dtype=np.int64)
    for i in range(n_epochs):
        epoch_indices = np.arange(dataset_size, dtype=np.int64)
        rng.shuffle(epoch_indices)
        indices[i * dataset_size:(i + 1) * dataset_size] = epoch_indices
    indices = indices[:n_samples]
    return indices


@jit(nopython=True, cache=True)
def build_sample_index(
        doc_index: np.ndarray,
        dataset_indices: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    # Keep track of the current index for each dataset
    dataset_counters = {i: 0 for i in range(len(dataset_indices))}

    # Construct the output array
    result = np.empty(len(doc_index), dtype="long")

    for i, dataset_id in enumerate(doc_index):
        # Get the index from the respective dataset array
        result[i] = dataset_indices[dataset_id][dataset_counters[dataset_id]]

        # Increment the counter for the used dataset
        dataset_counters[dataset_id] += 1

    return result


def count_document_indices(doc_indices: np.array, num_datasets: int) -> np.ndarray:
    unique, _counts = np.unique(doc_indices, return_counts=True)
    counts = np.zeros(num_datasets, dtype="long")
    for i, count in zip(unique, _counts):
        counts[i] = count
    return counts


def build_reproducible_document_index(
        n_samples: int, weights: np.ndarray, seed: int = 42, shuffle_freq: Optional[int] = None
) -> np.ndarray:
    if shuffle_freq is None:
        shuffle_freq = n_samples
    n_shuffles = math.ceil(n_samples / shuffle_freq)
    doc_index = build_document_index(n_shuffles * shuffle_freq, weights)
    return shuffle_document_indices(doc_index, shuffle_freq, seed=seed)[:n_samples]


def build_reproducible_document_index_simple(
        n_samples: int, weights: np.ndarray, seed: int = 42, **kwargs
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(range(len(weights)), p=np.array(weights) / sum(weights), size=n_samples)


class DatasetWrapper(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            n_samples: Optional[int] = None,
            seed: int = 1234
    ):
        self.dataset = dataset
        self.n_samples = len(self.dataset) if n_samples is None else n_samples
        self.index = create_dataset_indices(self.n_samples, len(self.dataset), seed)
        assert n_samples == len(self.index)
        logging.info(
            f"Creating Dataset Wrapper with {len(self.index)} samples ({len(self.index) / len(self.dataset)} epochs)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.dataset[int(self.index[idx])]


class CombinedDatasetWrapper(Dataset):
    def __init__(
            self,
            datasets: List[Dataset],
            weights: List[float],
            n_samples: Optional[int] = None,
            seed: int = 1234,
            shuffle_frequency: Optional[int] = None,
    ):
        self.seed = seed
        self.datasets = datasets

        if weights is None:
            weights = [1.0] * len(datasets)
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()
        self.weights = weights

        self.dataset_sizes = np.array([len(ds) for ds in self.datasets])
        self.n_samples = int(self.dataset_sizes.sum()) if n_samples is None else n_samples
        logging.info(f"Creating Combined Dataset with {self.n_samples} samples")
        logging.info(f"Dataset sizes: {self.dataset_sizes}")
        self.shuffle_frequency = self.n_samples if shuffle_frequency is None else shuffle_frequency
        logging.info(f"Shuffle frequency is set to {self.shuffle_frequency}")

        self.doc_index = build_reproducible_document_index(
            self.n_samples, self.weights, seed=self.seed, shuffle_freq=self.shuffle_frequency
        )
        self.samples_per_dataset = count_document_indices(self.doc_index, len(weights))
        dataset_indices = [
            create_dataset_indices(n_samples=samples, dataset_size=size, seed=self.seed)
            for size, samples in zip(self.dataset_sizes, self.samples_per_dataset)
        ]
        self.sample_idx = build_sample_index(self.doc_index, dataset_indices)
        calculated_samples = ((self.n_samples + 1) * weights).astype("long")
        actual_weights = self.samples_per_dataset / self.samples_per_dataset.sum()

        logging.info(f"Samples per datasets: {self.samples_per_dataset}")
        logging.info(f"Dataset epochs: {self.samples_per_dataset / self.dataset_sizes}")
        logging.info(f"Given weights: {self.weights}")
        logging.info(f"Actual weights: {actual_weights}")
        logging.info(f"Error in weights: {actual_weights - self.weights}")
        logging.info(f"Error in samples: {self.samples_per_dataset - calculated_samples}")
        assert self.n_samples == len(self.sample_idx)

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        doc_idx = int(self.doc_index[idx])
        sample_idx = int(self.sample_idx[idx])
        return self.datasets[doc_idx][sample_idx]
