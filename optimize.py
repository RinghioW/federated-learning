import numpy as np
from scipy.optimize import minimize
from plots import plot_optimization
from scipy.spatial.distance import jensenshannon
from typing import List
# It's needed to compute data imbalance and optimize the data shuffling
DI_SCALING = 10e4
# Function for optimizing equation 7 from ShuffleFL
def optimize_transmission_matrices(adaptive_scaling_factor: float,
                                   cluster_distributions: List[int],
                                   uplinks: List[float],
                                   downlinks: List[float],
                                   computes: List[float]) -> np.ndarray:

    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])

    objective_function_history: List[float] = []
    latency_history: List[float] = []
    data_imbalance_history: List[float] = []

    def objective(x: np.ndarray) -> float:
        transition_matrices = _tms_from_flat_unnormalized(x, n_devices, n_clusters)

        # Compute latencies and data imbalances
        cluster_distributions_post_shuffle = _shuffle_clusters(transition_matrices, cluster_distributions)

        data_imbalances = np.array([_data_imbalance(c) for c in cluster_distributions_post_shuffle])
        latencies = np.array(_latencies(uplinks, downlinks, computes, transition_matrices, cluster_distributions, cluster_distributions_post_shuffle))


        data_imbalance = np.mean(data_imbalances) * DI_SCALING
        system_latency = np.amax(latencies)
        obj_func = system_latency + data_imbalance
        objective_function_history.append(obj_func)
        latency_history.append(system_latency)
        data_imbalance_history.append(data_imbalance)
        return obj_func

    
    result = minimize(objective,
                        x0=np.random.rand(n_devices, n_clusters, n_devices).flatten(),
                        method='SLSQP',
                        bounds=[(0.,1.)] * (n_devices * n_clusters * n_devices),
                        )
    if not result.success:
        print(f"WARNING: Optimization did not converge: {result.message} with status {result.status}")
    
    plot_optimization(objective_function_history, latency_history, data_imbalance_history, adaptive_scaling_factor)

    transition_matrices = _tms_from_flat_unnormalized(result.x, n_devices, n_clusters)
        
    return transition_matrices


def _tms_from_flat_unnormalized(x, n_devices, n_clusters):
    transition_matrices = x.reshape((n_devices, n_clusters, n_devices))
    sums = np.sum(transition_matrices, axis=2, keepdims=True)
    mask = sums == 0
    transition_matrices = np.where(mask, 0, transition_matrices / np.where(mask, 1, sums))
    return transition_matrices

    
# Returns a list of the dataset clusters after shuffling the data
def _shuffle_clusters(transition_matrices: np.ndarray, cluster_distributions: List[int]) -> List[float]:
    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])
    dataset_distributions_post_shuffle = [[float(c) for c in cluster_distribution] for cluster_distribution in cluster_distributions]
    
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        cluster_distribution = cluster_distributions[d]
        for i in range(n_clusters):
            num_samples = float(cluster_distribution[i])
            for j in range(n_devices):
                if d != j:
                    transmitted_samples = transition_matrix[i][j] * num_samples
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
                    # Edge case where the sum of transmitted samples don't add up exactly to 1
                    if dataset_distributions_post_shuffle[d][i] < 0. and dataset_distributions_post_shuffle[d][i] > -1e-3:
                        dataset_distributions_post_shuffle[d][i] = 0.
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
    
    return dataset_distributions_post_shuffle


# Returns the data imbalance for a distribution
def _data_imbalance(distribution):
    n_samples = sum(distribution)
    if n_samples == 0:
        return np.float64(0.)
    n_classes = len(distribution)
    avg_samples = n_samples / n_classes
    balanced_distribution = [avg_samples] * n_classes
    js = jensenshannon(balanced_distribution, distribution)
    return js


def _latencies(uplinks, downlinks, computes, transition_matrices, dataset_distributions_pre_shuffle, dataset_distributions_post_shuffle):
    # Communication is computed before the shuffle
    t_communication = _t_communication(uplinks, downlinks, dataset_distributions_pre_shuffle, transition_matrices)
    
    # Computation time is measured on shuffled data
    t_computation = _t_computation(computes, dataset_distributions_post_shuffle)

    t_total = [sum(t) for t in zip(t_communication, t_computation)]

    return t_total

def _t_computation(computes, cluster_distributions):
    return [3. * compute * sum(cluster_distribution) for compute, cluster_distribution in zip(computes, cluster_distributions)]

# Returns a list of latencies for each device
def _t_communication(uplinks, downlinks, cluster_distributions, transition_matrices):
    n_devices = len(transition_matrices)
    t_transmission = [0.] * n_devices
    t_reception = [0.] * n_devices

    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        dataset_distribution = cluster_distributions[d]
        
        uplink = 1. / uplinks[d]

        for i in range(len(transition_matrix)):
            num_samples = dataset_distribution[i]
            for j in range(len(transition_matrix[0])):
                if d != j:
                    downlink = 1. / downlinks[j]
                    transferred_samples = transition_matrix[i][j] * num_samples
                    t_transmission[d] += uplink * transferred_samples
                    t_reception[j] += downlink * transferred_samples
    
    t_communication = [sum(t) for t in zip(t_transmission, t_reception)]
    return t_communication