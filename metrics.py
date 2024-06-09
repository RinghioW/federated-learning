import math
from scipy.spatial.distance import jensenshannon

# Returns the data imbalance for a distribution
def _data_imbalance(distribution):
    n_classes = len(distribution)
    n_samples = sum(distribution)
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
                    transferred_samples = math.floor(transition_matrix[i][j] * num_samples)
                    t_transmission[d] +=  uplink * transferred_samples
                    t_reception[j] += downlink * transferred_samples
    
    t_communication = [t_transmission[i] + t_reception[i] for i in range(n_devices)]
    return t_communication