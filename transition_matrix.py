import numpy as np

class TransitionMatrix:
    def __init__(self, num_clusters, num_devices):
        rand_matrix = np.random.rand(num_clusters, num_devices)
        sums = np.sum(rand_matrix, axis=1)
        self.matrix = rand_matrix / sums[:, :, np.newaxis]
