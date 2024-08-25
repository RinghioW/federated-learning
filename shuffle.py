import datasets
import math

def shuffle_data(ds: datasets.Dataset, clusters, distributions, transition_matrices, devices):
    n_devices = len(distributions)
    n_clusters = len(distributions[0])
    latency = 0
    kd_dataset = None
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        distribution = distributions[d]
        for i in range(n_clusters):
            for j in range(n_devices):
                if d != j:
                    pass
                    # Sample class i and send to device j
                    # n_samples = math.floor(transition_matrix[i][j] * distribution[i])
                    # if n_samples > 0:
                    #     ds[d], clusters[d], samples = extract(ds[d], clusters[d], i, n_samples)
                    #     # Add the samples to the dataset of device j
                    #     ds[j], clusters[j] = insert(ds[j], clusters[j], samples, i)
                
                else:
                    n_samples = math.floor(transition_matrix[i][j] * distribution[i])
                    latency += n_samples * devices[d].uplink
                    if kd_dataset is None:
                        kd_dataset = filter(ds[d], clusters[d], i, n_samples)
                    else:
                        kd_dataset = datasets.concatenate_datasets([kd_dataset, filter(ds[d], clusters[d], i, n_samples)])
                        
    return ds, kd_dataset, latency

# Function to remove data from a dataset
def extract(dataset, clusters, target, n_samples):
    # Find the indices of the samples that belong to the target cluster
    indices = [i for i in range(len(clusters)) if clusters[i] == target]
    indices = indices[:n_samples]
    # Select the samples from the dataset
    samples = dataset.select(indices)
    # Remove the samples from the dataset
    dataset = dataset.select([i for i in range(len(dataset)) if i not in indices])
    clusters = [clusters[i] for i in range(len(clusters)) if i not in indices]
    return dataset, clusters, samples

# Function to add data to a dataset
def insert(dataset, clusters, samples, cluster):
    # Add the samples to the dataset
    dataset = datasets.concatenate_datasets([dataset,samples])
    # Add the cluster labels
    clusters.extend([cluster for i in range(len(samples))])
    return dataset, clusters

def filter(dataset, clusters, target, n_samples):
    # Find the indices of the samples that belong to the target cluster
    indices = [i for i in range(len(clusters)) if clusters[i] == target]
    indices = indices[:n_samples]
    # Select the samples from the dataset
    samples = dataset.select(indices)
    # Remove the samples from the dataset
    return samples