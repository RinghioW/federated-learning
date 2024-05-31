import torch
from config import DEVICE, NUM_CLASSES
import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
import torchvision.transforms as transforms
import datasets
from copy import deepcopy
from torch.nn.utils import prune
class Device():
    def __init__(self, config, dataset, valset) -> None:
        self.config = config # Configuration of the device
        self.dataset = dataset

        self.valset = valset # TODO: Figure out how to use this

        self.model = None # Model class (NOT instance)
        self.model_params = None # Configuration to pass to the model constructor
        self.path = None # Relative path to save the model
        self.dataset_clusters = None # For each sample, the cluster that the sample belongs to

        self.num_transferred_samples = 0
        self.init = False # Useful to figure out whether we have a checkpoint or not

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"

    # Compute the JS divergence between the reference balanced distribution and the actual data distribution
    # Implements Equation 3 from ShuffleFL
    def data_imbalance(self):
        # Calculate the reference data distribution
        # The reference data distribution is a balanced distribution, all classes have the same number of samples
        reference_distribution = [math.floor(len(self.dataset)/NUM_CLASSES)] * NUM_CLASSES

        # Compute the JS divergence between the reference distribution and the actual data distribution<w
        distribution = [0] * NUM_CLASSES
        dataset = np.array(self.dataset)
        for sample in dataset:
            distribution[sample["label"]] += 1
        # Equation 3 from ShuffleFL
        js_divergence = jensenshannon(np.array(reference_distribution), np.array(distribution)) ** 2
        return js_divergence
    
    # Compute the JS divergence between the reference balanced distribution and the actual data distribution
    # In the mock version, only the clusters are used to compute the distribution
    def mock_data_imbalance(self):
        # In case the data distribution is empty, return 0.
        if len(self.dataset_clusters) == 0:
            return 0.
        # Calculate the reference data distribution
        NUM_CLUSTERS = 3
        reference_distribution = [math.floor(len(self.dataset_clusters)/NUM_CLUSTERS)] * NUM_CLUSTERS
        distribution = [0] * NUM_CLUSTERS
        for cluster in self.dataset_clusters:
            distribution[cluster] += 1
        js_divergence = jensenshannon(np.array(reference_distribution), np.array(distribution)) ** 2
        return js_divergence

    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs=10, verbose=False):
        print(f"Device {self.config['id']} - Training on {len(self.dataset)} samples")
        # Load the model
        net = self.model(**self.model_params)
        optimizer = torch.optim.Adam(net.parameters())

        if self.init:
            checkpoint = torch.load(self.path + f"device_{self.config['id']}.pt")
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.init = True
        
        net.train()

        to_tensor = transforms.ToTensor()
        dataset = self.dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()

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
                print(f"Device {self.config['id']} - Epoch {epoch+1}: loss {epoch_loss}, accuracy {epoch_acc}")

        # Save the model
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.path + f"device_{self.config['id']}.pt")

    # Mock functions are used to simulate the transfer of data between devices
    # Useful for optimization of the transmission matrices (and not the actual data transfer)
    def mock_remove_data(self, cluster, percentage_amount):
        clusters = np.array([], dtype=int)
        dataset_clusters = self.dataset_clusters
        num_cluster_samples = len([c for c in dataset_clusters if c == cluster])
        amount = math.floor(percentage_amount * num_cluster_samples) # Ensure that the amount is an integer
        for _ in range(amount):
            for idx, c in enumerate(dataset_clusters):
                if c == cluster:
                    # Add the sample to the list of samples to be sent if it matches the cluster
                    # Remove the sample from the dataset and the dataset clusters
                    clusters = np.append(clusters, c)
                    dataset_clusters = np.delete(dataset_clusters, idx, axis=0)
                    break
        
        # Update the dataset
        self.dataset_clusters = dataset_clusters
        # Update the number of samples that have been sent
        self.num_transferred_samples += amount
        return clusters
    
    def mock_add_data(self, clusters):
        self.dataset_clusters = np.append(self.dataset_clusters, clusters, axis=0)

    # Used in the transfer function to send data to a different device
    # Remove data that matches the cluster and return it
    def remove_data(self, cluster, percentage_amount):
        samples = np.array([])
        clusters = np.array([], dtype=int)
        dataset = np.array(self.dataset)
        dataset_clusters = self.dataset_clusters
        num_cluster_samples = len([c for c in dataset_clusters if c == cluster])
        amount = math.floor(percentage_amount * num_cluster_samples) # Ensure that the amount is an integer
        assert len(dataset) == len(dataset_clusters), f"Dataset length: {len(dataset)}, Dataset clusters length: {len(dataset_clusters)}"
        for _ in range(amount):
            for idx, c in enumerate(dataset_clusters):
                if c == cluster:
                    # Add the sample to the list of samples to be sent if it matches the cluster
                    samples = np.append(samples, [dataset[idx]], axis=0)
                    dataset = np.delete(dataset, idx, axis=0)
                    # Remove the sample from the dataset and the dataset clusters
                    clusters = np.append(clusters, c)
                    dataset_clusters = np.delete(dataset_clusters, idx, axis=0)
                    break
        
        samples = samples.tolist()

        # Update the dataset
        self.dataset = datasets.Dataset.from_list(dataset.tolist())
        self.dataset_clusters = dataset_clusters
        # Update the number of samples that have been sent
        self.num_transferred_samples += amount
        return samples, clusters

    # Used in the transfer function to receive data from a different device
    def add_data(self, samples, clusters):
        assert(len(samples) == len(clusters))
        dataset = np.array(self.dataset)
        dataset = np.append(dataset, samples, axis=0)
        self.dataset = datasets.Dataset.from_list(dataset.tolist())
        self.dataset_clusters = np.append(self.dataset_clusters, clusters, axis=0)
        assert(len(self.dataset) == len(self.dataset_clusters))
    
    # Function to sample a sub-dataset from the dataset
    def sample(self, percentage):
        print(f"Device {self.config['id']} - Sampling {percentage * 100}% of the data: {math.floor(percentage * len(self.dataset))} samples")
        dataset = np.array(self.dataset)
        amount = math.floor(percentage * len(dataset))

        # Randomly sample the dataset
        reduced_dataset = np.random.permutation(dataset)[:amount]
        return reduced_dataset

    def cluster_data(self, lda_estimator, kmeans_estimator):
        dataset = self.dataset
        dataset = np.array(dataset["img"]).reshape(len(dataset), -1)
        feature_space = lda_estimator.transform(dataset)
        # Cluster datapoints to k classes using KMeans
        self.dataset_clusters = kmeans_estimator.predict(feature_space)