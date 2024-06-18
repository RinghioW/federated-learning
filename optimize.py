import numpy as np
from scipy.optimize import minimize
from plots import plot_optimization
from scipy.spatial.distance import jensenshannon

# A dataset cluster is an array of sample classes
# It's needed to compute data imbalance and optimize the data shuffling

# Function for optimizing equation 7 from ShuffleFL
def optimize_transmission_matrices(adaptive_scaling_factor, cluster_distributions, uplinks, downlinks, computes):
    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])

    objective_function_history = []

    def objective_function(x):
        tms = x.reshape((n_devices, n_clusters, n_devices))
        sums = np.sum(tms, axis=2)
        tms = tms / sums[:, :, np.newaxis]
        latencies, data_imbalances = _shuffle_metrics(uplinks, downlinks, computes, tms, cluster_distributions)

        latencies = np.array(latencies)
        data_imbalances = np.array(data_imbalances)

        data_imbalance = np.mean((data_imbalances-np.min(data_imbalances)) / (np.max(data_imbalances)-np.min(data_imbalances)))
        system_latency = np.max((latencies-np.min(latencies)) / (np.max(latencies)-np.min(latencies)))
        obj_func = system_latency + data_imbalance

        objective_function_history.append(obj_func)

        return obj_func

    n_variables = n_devices * (n_clusters * n_devices)
    bounds = [(0.,100.)] * n_variables
    
    result = minimize(objective_function,
                        x0=np.random.rand(n_devices, n_clusters, n_devices).flatten(),
                        method='SLSQP', bounds=bounds,
                        options={'maxiter': 50})
    
    # Update the transition matrices
    objective_function_history.append(result.fun)
    plot_optimization(objective_function_history)
    tms = result.x.reshape((n_devices, n_clusters, n_devices))
    sums = np.sum(tms, axis=2)
    tms = (tms / sums[:, :, np.newaxis]).tolist()
    return tms


def _shuffle_metrics(uplinks, downlinks, computes, transition_matrices, cluster_distributions):
    dataset_distributions_post_shuffle = _shuffle_clusters(transition_matrices, cluster_distributions)

    l = latencies(uplinks, downlinks, computes, transition_matrices, cluster_distributions, dataset_distributions_post_shuffle)
    d = data_imbalances(dataset_distributions_post_shuffle)
    return l, d
    
# Returns a list of the dataset clusters after shuffling the data
def _shuffle_clusters(transition_matrices, cluster_distributions):
    n_devices = len(cluster_distributions)

    dataset_distributions_post_shuffle = cluster_distributions.copy()
    
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]

        for i in range(len(transition_matrix)):
            num_samples = cluster_distributions[d][i]
            for j in range(len(transition_matrix[0])):
                if d != j:
                    transmitted_samples = transition_matrix[i][j] * num_samples
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
        
    return dataset_distributions_post_shuffle


# Returns the data imbalance for a distribution
def _data_imbalance(distribution):
    n_samples = sum(distribution)
    if n_samples == 0:
        return 0.
    n_classes = len(distribution)
    avg_samples = n_samples / n_classes
    reference_distribution = [avg_samples] * n_classes

    return jensenshannon(reference_distribution, distribution) ** 2

# Returns a list of the data imbalance for each device
def data_imbalances(dataset_distributions):
    return [_data_imbalance(distribution) for distribution in dataset_distributions]

def latencies(uplinks, downlinks, computes, transition_matrices, dataset_distributions_pre_shuffle, dataset_distributions_post_shuffle):
    # Communication is computed before the shuffle
    t_communication = _t_communication(uplinks, downlinks, dataset_distributions_pre_shuffle, transition_matrices)
    
    # Computation time is measured on shuffled data
    t_computation = _t_computation(computes, dataset_distributions_post_shuffle)

    return [sum(t) for t in zip(t_communication, t_computation)]

# Returns a list of latencies for each device
def _t_computation(computes, dataset_distributions):
    n_devices = len(dataset_distributions)
    t_computation = []
    for d in range(n_devices):
        compute = computes[d]
        n_samples = sum(dataset_distributions[d])
        t = 3 * n_samples * compute
        t_computation.append(t)
    return t_computation

# Returns a list of latencies for each device
def _t_communication(uplinks, downlinks, dataset_distributions, transition_matrices):
    n_devices = len(transition_matrices)
    t_transmission = [0.] * n_devices
    t_reception = [0.] * n_devices

    for d in range(n_devices):
        transition_matrix = transition_matrices[d]
        dataset_distribution = dataset_distributions[d]
        
        uplink = 1 / uplinks[d]

        for i in range(len(transition_matrix)):
            num_samples = dataset_distribution[i]
            for j in range(len(transition_matrix[0])):
                if d != j:
                    downlink = 1 / downlinks[j]
                    transferred_samples = transition_matrix[i][j] * num_samples
                    t_transmission[d] +=  uplink * transferred_samples
                    t_reception[j] += downlink * transferred_samples
    
    t_communication = [t_transmission[i] + t_reception[i] for i in range(n_devices)]
    return t_communication