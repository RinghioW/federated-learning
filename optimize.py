import numpy as np
import math
from scipy.spatial.distance import jensenshannon

# A dataset cluster is an array of sample classes
# It's needed to compute data imbalance and optimize the data shuffling

def shuffle_result(configs, transition_matrices, dataset_clusters):
    # Compute the resulting dataset clusters after the data shuffling
    dataset_clusters_post_shuffle = _shuffle_data(transition_matrices, dataset_clusters)

    # Compute the latencies
    latencies = _latencies(configs, transition_matrices, dataset_clusters, dataset_clusters_post_shuffle)

    # Compute the data imbalances
    data_imbalances = _data_imbalances(dataset_clusters_post_shuffle)

    return latencies, data_imbalances
    
# Returns a list of the dataset clusters after shuffling the data
def _shuffle_data(transition_matrices, dataset_clusters):
    n_devices = len(dataset_clusters)
    dataset_clusters_post_shuffle = dataset_clusters.copy()
    
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]

        for i in range(len(transition_matrices)):
            num_samples = dataset_clusters[d][i]
            for j in range(len(transition_matrices[0])):
                if d != j:
                    transmitted_samples = math.floor(transition_matrix[i][j] * num_samples)
                    dataset_clusters_post_shuffle[d][i] -= transmitted_samples
                    dataset_clusters_post_shuffle[j][i] += transmitted_samples
        
    return dataset_clusters_post_shuffle

# Returns a list of the data imbalance for each device
def _data_imbalances(dataset_clusters):
    # In case the data distribution is empty, return 0.
    n_devices = len(dataset_clusters)
    data_imbalances = []
    for d in range(n_devices):
        dataset_cluster = dataset_clusters[d]
        n_classes = len(dataset_cluster)
        n_samples = sum(dataset_cluster)
        reference_distribution = [math.floor(n_samples/n_classes)] * n_classes

        js_divergence = jensenshannon(np.array(reference_distribution), np.array(dataset_cluster)) ** 2
        data_imbalances.append(js_divergence)
    return data_imbalances

# Returns a list of latencies for each device
def _latencies(configs, transition_matrices, dataset_clusters_pre_shuffle, dataset_clusters_post_shuffle):
    n_devices = len(configs)
    # Communication is computed before the shuffle
    t_comm = _t_communication(configs, dataset_clusters_pre_shuffle, transition_matrices)
    
    # Computation time is measured on shuffled data
    t_comp = _t_computation(configs, dataset_clusters_post_shuffle)
    
    latencies = [t_comm[i] + t_comp[i] for i in range(n_devices)]
    return latencies

# Returns a list of latencies for each device
def _t_computation(configs, dataset_clusters):
    n_devices = len(configs)
    t_computation = []
    for d in range(n_devices):
        compute = configs[d]["compute"]
        n_samples = sum(dataset_clusters[d])
        t = 3 * n_samples * compute
        t_computation.append(t)
    return t_computation

# Returns a list of latencies for each device
def _t_communication(configs, dataset_clusters, transition_matrices):
    n_devices = len(configs)
    t_transmission = [0.] * n_devices
    t_reception = [0.] * n_devices

    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        dataset_cluster = dataset_clusters[d]
        
        uplink = configs[d]["uplink_rate"]

        for i in range(len(transition_matrix)):
            num_samples = dataset_cluster[i]
            for j in range(len(transition_matrix[0])):
                if d != j:
                    downlink = configs[j]["downlink_rate"]
                    transferred_samples = math.floor(transition_matrix[i][j] * num_samples)
                    t_transmission[d] +=  uplink * transferred_samples
                    t_reception[j] += downlink * transferred_samples
    
    t_communication = [t_transmission[i] + t_reception[i] for i in range(n_devices)]
    return t_communication