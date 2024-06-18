from metrics import data_imbalances, latencies
import math
import numpy as np
from scipy.optimize import minimize
from plots import plot_optimization

# A dataset cluster is an array of sample classes
# It's needed to compute data imbalance and optimize the data shuffling

# Function for optimizing equation 7 from ShuffleFL
def optimize_transmission_matrices(adaptive_scaling_factor, cluster_distributions, uplinks, downlinks, computes):
    n_devices = len(cluster_distributions)
    n_clusters = len(cluster_distributions[0])

    latency_history = []
    data_imbalance_history = []
    objective_function_history = []

    def objective_function(x):
        tms = x.reshape((n_devices, n_clusters, n_devices))

        latencies, data_imbalances = _shuffle_metrics(uplinks, downlinks, computes, tms, cluster_distributions)

        system_latency = np.amax(latencies)
        data_imbalance = np.mean(data_imbalances)
        obj_func = system_latency + data_imbalance

        latency_history.append(system_latency)
        data_imbalance_history.append(data_imbalance)
        objective_function_history.append(obj_func)

        return obj_func

    # Sum(row) <= 1
    # Equivalent to [1 - Sum(row)] >= 0
    def one_minus_sum_rows(variables):
        tms = variables.reshape((n_devices, n_clusters, n_devices))
        return (1. - np.sum(tms, axis=2)).flatten()
    
    n_variables = n_devices * (n_clusters * n_devices)
    bounds = [(0.,1.)] * n_variables
    constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables)}]
    
    transition_matrices = np.random.rand(n_devices, n_clusters, n_devices)
    sums = np.sum(transition_matrices, axis=2)
    transition_matrices = (transition_matrices / sums[:, :, np.newaxis]).flatten()

    result = minimize(objective_function,
                        x0=transition_matrices,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 50})
    
    # Update the transition matrices
    if not result.success:
        print(f"Optimization did not converge after {result.nit} iterations. Status: {result.status} Message: {result.message}")
    objective_function_history.append(result.fun)
    plot_optimization(latency_history, data_imbalance_history, objective_function_history)
    return result.x.reshape((n_devices, n_clusters, n_devices))


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
                    transmitted_samples = math.floor(transition_matrix[i][j] * num_samples)
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
        
    return dataset_distributions_post_shuffle