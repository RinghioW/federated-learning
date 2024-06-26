import datasets
import math
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
                    datasets[d], clusters[d], samples = _extract(datasets[d], clusters[d], i, n_samples)

                    # Add the samples to the dataset of device j
                    datasets[j], clusters[j] = _insert(datasets[j], clusters[j], samples, i)

                    n_transferred_samples += n_samples
    
    return datasets, n_transferred_samples

# Extract an amount of samples of the target cluster from the dataset
def _extract(dataset, cluster_labels, target_cluster, n_samples):
    if n_samples == 0:
        return dataset, cluster_labels, []
    dataset = list(dataset)
    idx_matches = []
    matches = []

    for idx, cluster in enumerate(cluster_labels):
        if cluster == target_cluster and len(matches) < n_samples:
            idx_matches.append(idx)
            matches.append(dataset[idx])

    # Remove the samples from the dataset and cluster labels
    dataset = [sample for idx, sample in enumerate(dataset) if idx not in idx_matches]
    cluster_labels = [cluster for idx, cluster in enumerate(cluster_labels) if idx not in idx_matches]
    return datasets.Dataset.from_list(dataset), cluster_labels, matches

# Insert samples of cluster label into the dataset
def _insert(dataset, cluster_labels, samples, target_label):
    if len(samples) == 0:
        return dataset, cluster_labels
    dataset = list(dataset)
    for sample in samples:
        dataset.append(sample)
        cluster_labels.append(target_label)
    return datasets.Dataset.from_list(dataset), cluster_labels
