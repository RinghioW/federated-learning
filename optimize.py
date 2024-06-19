import numpy as np
from scipy.optimize import minimize
from plots import plot_optimization
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import minmax_scale
from copy import deepcopy
# It's needed to compute data imbalance and optimize the data shuffling

# Function for optimizing equation 7 from ShuffleFL
def optimize_transmission_matrices(adaptive_scaling_factor, cluster_distributions, uplinks, downlinks, computes):

    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])

    objective_function_history = []

    def objective(x):
        transition_matrices = x.reshape((n_devices, n_clusters, n_devices))
        
        # Normalize the rows
        transition_matrices = transition_matrices / np.sum(transition_matrices, axis=2, keepdims=True)

        # Compute latencies and data imbalances
        cluster_distributions_post_shuffle = _shuffle_clusters(transition_matrices, cluster_distributions)

        data_imbalances = [_data_imbalance(c) for c in cluster_distributions_post_shuffle]
        latencies = _latencies(uplinks, downlinks, computes, transition_matrices, cluster_distributions, cluster_distributions_post_shuffle)

        # Min-max normalization in [0, 1]
        latencies = minmax_scale(latencies)
        data_imbalances = minmax_scale(data_imbalances)

        data_imbalance = np.mean(data_imbalances)
        system_latency = np.amax(latencies)
        obj_func = (1. - adaptive_scaling_factor)*system_latency + adaptive_scaling_factor*data_imbalance

        objective_function_history.append(obj_func)

        return obj_func

    
    result = minimize(objective,
                        x0=np.random.rand(n_devices, n_clusters, n_devices).flatten(),
                        method='SLSQP',
                        bounds=[(0.,1.)] * (n_devices * n_clusters * n_devices),
                        options={'maxiter': 50})
    if not result.success:
        print(f"WARNING: Optimization did not converge: {result.message} with status {result.status}")
    
    objective_function_history.append(result.fun)
    plot_optimization(objective_function_history)

    transition_matrices = result.x.reshape((n_devices, n_clusters, n_devices))
    transition_matrices = transition_matrices / np.sum(transition_matrices, axis=2, keepdims=True)
    return transition_matrices

    
# Returns a list of the dataset clusters after shuffling the data
def _shuffle_clusters(transition_matrices, cluster_distributions):
    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])
    dataset_distributions_post_shuffle = deepcopy(cluster_distributions)
    
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        cluster_distribution = cluster_distributions[d]
        for i in range(n_clusters):
            num_samples = cluster_distribution[i]
            for j in range(n_devices):
                if d != j:
                    transmitted_samples = transition_matrix[i][j] * num_samples
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
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
    js = jensenshannon(balanced_distribution, distribution) ** 2
    return js


def _latencies(uplinks, downlinks, computes, transition_matrices, dataset_distributions_pre_shuffle, dataset_distributions_post_shuffle):
    # Communication is computed before the shuffle
    t_communication = _t_communication(uplinks, downlinks, dataset_distributions_pre_shuffle, transition_matrices)
    
    # Computation time is measured on shuffled data
    t_computation = _t_computation(computes, dataset_distributions_post_shuffle)

    t_total = [sum(t) for t in zip(t_communication, t_computation)]

    return t_total
# Returns a list of latencies for each device
def _t_computation(computes, cluster_distributions):
    n_devices = len(cluster_distributions)
    t_computation = []
    for d in range(n_devices):
        compute = computes[d]
        n_samples = sum(cluster_distributions[d])
        t = 3. * n_samples * compute
        t_computation.append(t)
    return t_computation

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
                    t_transmission[d] +=  uplink * transferred_samples
                    t_reception[j] += downlink * transferred_samples
    
    t_communication = [t_transmission[i] + t_reception[i] for i in range(n_devices)]
    return t_communication