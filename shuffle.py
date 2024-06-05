import datasets
# Returns device datasets and cluster labels after shuffling the data
# Assume that datasets is a list of numpy arrays
# Implements the transformation described by Equation 1 from ShuffleFL
def shuffle_data(datasets, clusters, distributions, transition_matrices):
    n_devices = len(transition_matrices)
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        distribution = distributions[d]
        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[0])):
                if d != j:
                    # Sample class i and send to device j
                    num_samples = transition_matrix[i][j] * distribution[i]
                    datasets[d], clusters[d], samples = _extract(datasets[d], clusters[d], i, num_samples)

                    # Add the samples to the dataset of device j
                    datasets[j], clusters[j] = _insert(datasets[j], clusters[j], samples, i)
    
    return datasets, clusters

# Extract an amount of samples of the target cluster from the dataset
def _extract(dataset, cluster_labels, target_cluster, amount):
    dataset = list(dataset)
    
    matches = []
    while len(matches) < amount:
        for idx, cluster in enumerate(cluster_labels):
            if cluster == target_cluster:
                matches.append(dataset.pop(idx))
                cluster_labels.pop(idx)
                break
        break
    return datasets.Dataset.from_list(dataset), cluster_labels, matches

# Insert samples of cluster label into the dataset
# TODO : dataset is a Dataset object and should return so
def _insert(dataset, cluster_labels, samples, target_label):
    dataset = list(dataset)
    for sample in samples:
        dataset.append(sample)
        cluster_labels.append(target_label)
    return datasets.Dataset.from_list(dataset), cluster_labels
