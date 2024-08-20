import datasets
import math

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
                    if samples is not None and len(samples) > 0:
                        datasets[j], clusters[j] = _insert(datasets[j], clusters[j], samples, i)

                    n_transferred_samples += n_samples
    
    return datasets, n_transferred_samples

def _extract(dataset, cluster_labels, target_cluster, n_samples):
    if n_samples == 0:
        return dataset, cluster_labels, None
    idx_matches = []
    matches = []

    for idx, cluster in enumerate(cluster_labels):
        if cluster == target_cluster:
            idx_matches.append(idx)

    # Dataset filter the rows on idx_matches
    idx_matches = idx_matches[:n_samples]
    matches = dataset.select(idx_matches)
    # Remove the samples from the dataset
    dataset = dataset.select([idx for idx in range(len(dataset)) if idx not in idx_matches])
    
    cluster_labels = [cluster for idx, cluster in enumerate(cluster_labels) if idx not in idx_matches]
    assert len(dataset) == len(cluster_labels)
    return dataset, cluster_labels, matches

def _insert(dataset, cluster_labels, samples, target_label):
    if samples is None or len(samples) == 0:
        return dataset, cluster_labels
    cluster_labels = cluster_labels + [target_label] * len(samples)
    dataset = datasets.concatenate_datasets([dataset, samples])
    assert len(dataset) == len(cluster_labels)
    return dataset, cluster_labels