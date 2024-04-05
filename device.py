import torch
from config import DEVICE, NUM_CLASSES
import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon

class Device():
    def __init__(self, config, dataset, valset) -> None:
        self.config = config
        self.dataset = dataset
        # The knowledge distillation dataset is created by sending a fraction of the dataset to itself when optimizing the transmission matrix
        self.kd_dataset = []

        self.valset = valset
        self.model = None

        self.datset_clusters = [None] * len(self.dataset) # For each sample, the cluster that the sample belongs to

        self.num_transferred_samples = 0

    def __repr__(self) -> str:
        return f"Device({self.config})"

    # Compute the JS divergence between the reference balanced distribution and the actual data distribution
    # Implements Equation 3 from ShuffleFL
    def data_imbalance(self):
        # Calculate the reference data distribution
        # The reference data distribution is a balanced distribution, all classes have the same number of samples
        reference_distribution = [len(self.dataset)/NUM_CLASSES] * NUM_CLASSES

        # Compute the JS divergence between the reference distribution and the actual data distribution
        dataloader = torch.utils.data.DataLoader(self.dataset)
        distribution = [0] * NUM_CLASSES
        for element in dataloader:
            distribution[element["label"]] += 1

        # Equation 3 from ShuffleFL
        js_divergence = jensenshannon(np.array(reference_distribution), np.array(distribution), base=2) ** 2
        return js_divergence

    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs=5, verbose=True):
        if self.model is None or self.dataset is None:
            raise ValueError("Model or dataset is None.")
        net = self.model
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=2)
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    # Used in the transfer function to send data to a different device
    # Remove data that matches the cluster and return it
    def remove_data(self, cluster, percentage_amount, add_to_kd_dataset=False):
        samples = []
        amount = math.floor(percentage_amount * len(self.dataset)) # Ensure that the amount is an integer
        removed = False
        for i in range(amount):
            for idx, c in enumerate(self.datset_clusters):
                if c == cluster:
                    sample = self.dataset[idx]
                    samples.append(sample)
                    np.delete(self.dataset, idx)
                    removed = True
                    break
            if not removed:
                print(f"Warning! Not enough samples. Could only remove {i} out of required {amount} samples of cluster {cluster} from the dataset.")
            removed = False

        # If the data is to be added to the knowledge distillation dataset, do so
        # And return immediately
        if add_to_kd_dataset:
            for sample in samples:
                self.kd_dataset.append(sample)
            return samples

        # Update the number of samples that have been sent
        self.num_transferred_samples += amount

        return samples

    # Used in the transfer function to receive data from a different device
    def add_data(self, samples):
        for sample in samples:
            np.append(self.dataset, sample)

    # Assing each datapoint to a cluster
    def cluster_data(self, shrinkage_ratio):
        # Use t-SNE to embed the dataset into 2D space
        # Aggregate only the features, not the labels
        feature_space = np.array(self.dataset["img"]).reshape(len(self.dataset), -1)
        feature_space_2D = TSNE(n_components=2).fit_transform(feature_space)

        # Cluster datapoints to k classes using KMeans
        n_clusters = math.floor(shrinkage_ratio*NUM_CLASSES)
        self.datset_clusters = KMeans(n_clusters).fit_predict(feature_space_2D)