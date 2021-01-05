from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data.dataset import Dataset
from random import gauss

# class PartitionType(Enum):
#     HOMOGENEOUS = 1
#     HETEROGENEOUS = 2

# def create_partitioning(train_data: Dataset, test_data: Dataset, partitionType: PartitionType, n_partitions: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
#     partitioning = {
#         PartitionType.HOMOGENEOUS: create_partitioning_homogeneous,
#         PartitionType.HETEROGENEOUS: create_partitioning_heterogeneous
#     }.get(partitionType)

#     return partitioning(train_data, test_data, n_partitions)


# def create_partitioning_homogeneous(train_data: Dataset, test_data: Dataset, n_partitions: int):
#     train_partitioning = create_partitioning_even(train_data, n_partitions)
#     test_partitioning = create_partitioning_even(test_data, n_partitions)
#     return train_partitioning, test_partitioning


# def create_partitioning_heterogeneous(
#         train_data: Dataset, test_data: Dataset, n_classes: int, n_partitions: int,
#         is_balanced: bool, dirichlet_alpha: float = 0.5):

#     train_partitioning = create_partitioning_uneven(train_data, n_classes, n_partitions, is_balanced, dirichlet_alpha)
#     test_partitioning = create_partitioning_uneven(test_data, n_classes, n_partitions, is_balanced, dirichlet_alpha)

#     return train_partitioning, test_partitioning


def create_partitioning_iid(
        data: Dataset, n_partitions: int,
        is_balanced: bool, unbalanced_min_part_min_size: float, unbalanced_min_part_max_size: float,
        dirichlet_alpha: float = 0.5) -> Dict[int, List[int]]:
    
    # Based on code from: https://github.com/IBM/probabilistic-federated-neural-matching
    n = len(data)
    idxs = np.random.permutation(n) # Shuffle data
    partition_dataidxs_map: Dict[int, List[int]] = {}
    min_part_size = 0
    min_part_low = round(unbalanced_min_part_min_size * n)
    min_part_high = round(unbalanced_min_part_max_size * n)

    if is_balanced:
        # Split evenly
        batch_idxs = np.array_split(idxs, n_partitions)
        partition_dataidxs_map = {i: batch_idxs[i] for i in range(n_partitions)}
    else:
    # Try to partition until each partition has at least some samples
        while min_part_size < min_part_low or min_part_size > min_part_high:
            # Split unevenly
            batch_idxs = list(divide(idxs, min_part_low, n_partitions))

            min_part_size = min([len(p) for p in batch_idxs])
            partition_dataidxs_map = {i: batch_idxs[i] for i in range(n_partitions)}
            
    return partition_dataidxs_map

def divide(lst, min_size, split_size):
    # https://stackoverflow.com/questions/14427531
    it = iter(lst)
    from itertools import islice
    size = len(lst)
    for i in range(split_size - 1, 0, -1):
        s = np.random.randint(min_size, size - min_size * i)
        yield list(islice(it, 0, s))
        size -= s
    yield list(it)

def create_partitioning_non_iid(
        data: Dataset, n_classes: int, n_partitions: int,
        is_balanced: bool, unbalanced_min_part_min_size: float, unbalanced_min_part_max_size: float,
        dirichlet_alpha: float = 0.5) -> Dict[int, List[int]]:
    # Based on code from: https://github.com/IBM/probabilistic-federated-neural-matching
    X = data[:][0]
    y = data[:][1]

    n = len(data)
    partition_dataidxs_map: Dict[int, List[int]] = {}
    min_part_size = 0
    part_sizes = np.empty((1, n_partitions))
    avg_part_size = n / n_partitions
    n_partitions_close_to_average = 0
    threshold = 10
    min_part_low = round(unbalanced_min_part_min_size * n)
    min_part_high = round(unbalanced_min_part_max_size * n)

    # Try to partition until each partition has at least some samples
    # while min_part_size < min_part_low or (min_part_size > min_part_high and not is_balanced) or (n_partitions_close_to_average < n_partitions and is_balanced):
    while min_part_size < min_part_low or (min_part_size > min_part_high and not is_balanced):

        parts = [list()] * n_partitions
        
        # Partition each class randomly
        for k in range(n_classes):
            # Select indexes of class k
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)

            # Partition using dirichlet distribution. Returned values sum to 1.
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, n_partitions))

            if is_balanced:
                # Find all partitions that currently have lower than average partition sizes
                proportions = np.array([p * (len(part) < avg_part_size) for p, part in zip(proportions, parts)])

                # Normalize the proportions of these partitions
                norm_proportions = proportions / proportions.sum()

                # Convert proportions to indexes
                proportion_idxs = (np.cumsum(norm_proportions) * len(idx_k)).astype(int)[:-1]
            else:
                proportion_idxs = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # Add indexes to partitions
            parts = [idx_j + idx.tolist() for idx_j, idx in zip(parts, np.split(idx_k, proportion_idxs))]

            part_sizes = np.array([len(p) for p in parts])
            min_part_size = min(part_sizes)
            n_partitions_close_to_average = sum(abs(part_sizes - avg_part_size) < threshold)
            
        for i in range(n_partitions):
            np.random.shuffle(parts[i])
            partition_dataidxs_map[i] = parts[i]

    if is_balanced:
        size_above_mean = [len(partition_dataidxs_map[i]) - avg_part_size for i in range(n_partitions)]
        for i in range(n_partitions):
            if size_above_mean[i] > 0:
                samples_above_mean, partition_dataidxs_map[i] = np.split(partition_dataidxs_map[i], [int(size_above_mean[i])])
                
                for j, value in enumerate(size_above_mean):
                    if value < 0 and len(samples_above_mean) > 0:
                        diff = len(samples_above_mean) + value
                        if diff < 0:
                            partition_dataidxs_map[j].extend(samples_above_mean)
                        else:
                            new_samples, new_samples_above_mean = np.split(samples_above_mean, [abs(int(value))])
                            partition_dataidxs_map[j].extend(new_samples)
                            samples_above_mean = new_samples_above_mean
                        size_above_mean = [len(partition_dataidxs_map[i]) - avg_part_size for i in range(n_partitions)]

    return partition_dataidxs_map

def light_shuffle(data, k = 1000, orderliness = 2):
    # https://stackoverflow.com/questions/62436299
    return sorted(data, key=lambda i: gauss(i[0] * orderliness, 1))

def get_data_partitioning_stats(y_train, partition_dataidxs_map):
    # Based on code from: https://github.com/IBM/probabilistic-federated-neural-matching
	partition_class_counts = {}

	for partition_idx, data_idx in partition_dataidxs_map.items():
        # Get the label count of each partition
		unique_labels, unique_label_count = np.unique(y_train[data_idx], return_counts=True)

        # Store counts in dict
		tmp = {unique_labels[i]: unique_label_count[i] for i in range(len(unique_labels))}
		partition_class_counts[partition_idx] = tmp

	return partition_class_counts