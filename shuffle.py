import datasets
import math
from config import LABEL_NAME
# Returns device datasets and cluster labels after shuffling the data
# Assume that datasets is a list of numpy arrays
# Implements the transformation described by Equation 1 from ShuffleFL
def shuffle_data(datasets, clusters, distributions, transition_matrices):
    n_devices = len(distributions)
    n_clusters = len(distributions[0])
    n_transferred_samples = 0
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        distribution = distributions[d]
        for i in range(n_clusters):
            for j in range(n_devices):
                if d != j:
                    # Sample class i and send to device j
                    n_samples = math.floor(transition_matrix[i][j] * distribution[i])
                    datasets[d], samples = _extract(datasets[d], clusters[d], n_samples)

                    # Add the samples to the dataset of device j
                    datasets[j] = _insert(datasets[j], samples)

                    n_transferred_samples += n_samples
    
    return datasets, n_transferred_samples

# Extract an amount of samples of the target cluster from the dataset
def _extract(dataset, target_cluster, n_samples):
    dataset_cluster = dataset.filter(lambda x: cluster(x) == target_cluster)
    # Remove the samples from the dataset
    samples = dataset_cluster.shuffle().select(range(min(n_samples, len(dataset_cluster))))
    dataset = dataset.filter(lambda x: x not in samples)
    return dataset, samples

# Insert samples into the dataset
def _insert(dataset, samples):
    return datasets.concatenate_datasets([dataset, samples])

# TODO: Implement cluster function
def cluster(x):
    return x[LABEL_NAME]