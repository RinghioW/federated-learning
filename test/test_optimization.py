import numpy as np
from typing import List
'''
 Transition matrices: 
 [[[0.76817039 0.         0.23182961]
  [0.         0.47870019 0.52129981]
  [0.         0.51522379 0.48477621]]

 [[0.41981796 0.09207708 0.48810497]
  [0.         0.47605928 0.52394072]
  [0.17442033 0.32272364 0.50285603]]

 [[0.66066069 0.26728998 0.07204933]
  [0.         0.58916521 0.41083479]
  [0.3615728  0.36178587 0.27664133]]]. 
  Original cluster distributions: [[1058, 1191, 455], [1258, 1846, 514], [1390, 1638, 648]].
'''

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
                    # Edge case where the floats don't add up exactly to 1
                    if dataset_distributions_post_shuffle[d][i] < 0. and dataset_distributions_post_shuffle[d][i] > -1e-3:
                        dataset_distributions_post_shuffle[d][i] = 0.
                    dataset_distributions_post_shuffle[j][i] += transmitted_samples
    
    return dataset_distributions_post_shuffle

transition_matrices = np.array([[[0.76817039, 0., 0.23182961], [0., 0.47870019, 0.52129981], [0., 0.51522379, 0.48477621]], [[0.41981796, 0.09207708, 0.48810497], [0., 0.47605928, 0.52394072], [0.17442033, 0.32272364, 0.50285603]], [[0.66066069, 0.26728998, 0.07204933], [0., 0.58916521, 0.41083479], [0.3615728, 0.36178587, 0.27664133]]])
cluster_distributions = np.array([[1058, 1191, 455], [1258, 1846, 514], [1390, 1638, 648]])

res = _shuffle_clusters(transition_matrices, cluster_distributions)
print(res)