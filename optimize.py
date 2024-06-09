from metrics import data_imbalances, latencies
import math
from config import STD_CORRECTION, Style
import numpy as np
from scipy.optimize import minimize

# A dataset cluster is an array of sample classes
# It's needed to compute data imbalance and optimize the data shuffling

# Function for optimizing equation 7 from ShuffleFL
def optimize_transmission_matrices(transition_matrices,
                                   n_devices, 
                                   n_clusters,
                                    adaptive_scaling_factor, 
                                    cluster_distributions, 
                                    uplinks, 
                                    downlinks, 
                                    computes):
    # Define the objective function to optimize
    # Takes as an input the transfer matrices
    # Returns as an output the result of Equation 7

    for m in transition_matrices:
        if m is None:
            transition_matrices = np.random.rand(n_devices, n_clusters, n_devices)
            sums = np.sum(transition_matrices, axis=2)
            transition_matrices = transition_matrices / sums[:, :, np.newaxis]
            break

    transition_matrices = np.array(transition_matrices)

    def objective_function(x):
        # Parse args
        tms = x.reshape((n_devices, n_clusters, n_devices))
            
        # Simulate the transferring of the data according to the matrices
        latencies, data_imbalances = _shuffle_metrics(uplinks, downlinks, computes, tms, cluster_distributions)

        # Compute the loss function
        # The factor of STD_CORRECTION was introduced to increase by an order of magnitude the importance of the time std
        # Time std is usually very small and the max time is usually very large
        # But a better approach would be to normalize the values or take the square of the std
        system_latency = STD_CORRECTION*np.std(latencies) + np.max(latencies)
        data_imbalance = adaptive_scaling_factor*np.max(data_imbalances)
        obj_func = system_latency + data_imbalance

        # Save the objective function for plotting
        return obj_func

    # Define the constraints for the optimization
    # Row sum represents the probability of data of each class that is sent
    # Sum(row) <= 1
    # Equivalent to [1 - Sum(row)] >= 0
    # Note that in original ShuffleFL the constraint is Sum(row) = 1
    # But in this case, we can use the same column as an additional dataset
    def one_minus_sum_rows(variables):
        # Reshape the flat variables back to the transition matrices shape
        tms = variables.reshape((n_devices, n_clusters, n_devices))
        return (1. - np.sum(tms, axis=2)).flatten()
    

    # Each element in the matrix is a probability, so it must be between 0 and 1
    n_variables = n_devices * (n_clusters * n_devices)
    bounds = [(0.,1.)] * n_variables
    # If the sum is less than one, we can use same-device column as additional dataset
    # constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables, num_devices, num_clusters)}]
    constraints = [{'type': 'ineq', 'fun': lambda variables: one_minus_sum_rows(variables)}]
    
    # Run the optimization
    # TODO: Should the initial guess be the old transition matrix?
    transition_matrices = transition_matrices.flatten()
    result = minimize(objective_function,
                        x0=transition_matrices,
                        method='SLSQP', bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 50, 'ftol': 1e-01, 'eps': 1e-01})
    
    # Update the transition matrices
    if not result.success:
        print(f"{Style.RED}[ERROR]{Style.RESET} Optimization did not converge after {result.nit} iterations. Status: {result.status} Message: {result.message}")
    
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

        for i in range(len(transition_matrices)):
            num_samples = cluster_distributions[d][i]
            for j in range(len(transition_matrices[0])):
                if d != j:
                    transmitted_samples = math.floor(transition_matrix[i][j] * num_samples)
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
        
    return dataset_distributions_post_shuffle