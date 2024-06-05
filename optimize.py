from metrics import data_imbalances, latencies
from copy import deepcopy
import math
# A dataset cluster is an array of sample classes
# It's needed to compute data imbalance and optimize the data shuffling

def shuffle_metrics(uplinks, downlinks, computes, transition_matrices, dataset_distributions):
    dataset_distributions_post_shuffle = _shuffle_data(transition_matrices, dataset_distributions)

    l = latencies(uplinks, downlinks, computes, transition_matrices, dataset_distributions, dataset_distributions_post_shuffle)
    d = data_imbalances(dataset_distributions_post_shuffle)
    return l, d
    
# Returns a list of the dataset clusters after shuffling the data
def _shuffle_data(transition_matrices, dataset_distributions):
    n_devices = len(dataset_distributions)

    dataset_distributions_post_shuffle = dataset_distributions.copy()
    
    for d in range(n_devices):
        transition_matrix = transition_matrices[d]

        for i in range(len(transition_matrices)):
            num_samples = dataset_distributions[d][i]
            for j in range(len(transition_matrices[0])):
                if d != j:
                    transmitted_samples = math.floor(transition_matrix[i][j] * num_samples)
                    dataset_distributions_post_shuffle[d][i] -= transmitted_samples
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
        
    return dataset_distributions_post_shuffle