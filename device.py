import torch
from config import DEVICE, NUM_CLASSES
import numpy as np
from utils import js_divergence
import pandas as pd
import math

class Device():
    def __init__(self, config, dataset, valset) -> None:
        self.config = config
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.valset = valset
        self.model = None
        self.transition_matrix = None
        self.num_transferred_samples = 0

    def __repr__(self) -> str:
        return f"Device(config={self.config})"
    
    # Compute the JS divergence between the reference balanced distribution and the actual data distribution
    # Implements Equation 3 from ShuffleFL
    def data_imbalance(self):
        # Calculate the reference data distribution
        # The reference data distribution is a balanced distribution, all classes have the same number of samples
        reference_distribution = [len(self.dataset)/NUM_CLASSES for _ in range(NUM_CLASSES)]

        # Compute the JS divergence between the reference distribution and the actual data distribution
        dataloader = torch.utils.data.DataLoader(self.dataset)
        distribution = [0 for _ in range(NUM_CLASSES)]
        for element in dataloader:
            distribution[element["label"]] += 1

        # Equation 3 from ShuffleFL
        return js_divergence(np.array(reference_distribution), np.array(distribution))
    

    def initialize_transition_matrix(self, num_devices):
        self.transition_matrix = np.ones((NUM_CLASSES, num_devices), dtype=int)

    # Compute the latency of the device wrt uplink rate, downlink rate, and compute
    # Implements Equation 4, 5 and 6 from ShuffleFL
    def latency(self, device_idx, devices, epochs):
        # Communication depends on the transition matrix
        # Equation 5 from ShuffleFL
        t_communication = 0
        for data_class_idx, data_class in enumerate(self.transition_matrix):
            for other_device_idx, other_device in enumerate(devices):
                if device_idx != other_device_idx:
                    # Transmitting
                    t_communication += self.transition_matrix[data_class_idx][other_device_idx] * ((1/self.config["uplink_rate"]) + (1/other_device.config["downlink_rate"]))
                    # Receiving
                    t_communication += other_device.transition_matrix[data_class_idx][device_idx] * ((1/self.config["downlink_rate"]) + (1/other_device.config["uplink_rate"]))

        # Equation 6 from ShuffleFL
        t_computation = 3 * epochs * self.dataset_size * self.config["compute"]

        # Equation 4 from ShuffleFL
        return t_communication + t_computation

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
    def remove_data(self, data_class, amount):
        samples = []
        for _ in range(math.floor(amount)):
            for idx, sample in enumerate(self.dataset):
                if sample["label"] == data_class:
                    samples.append(sample)
                    np.delete(self.dataset, idx)
                    break
            print("Warning! Not enough samples")
            return Exception(f"Could not remove {amount} samples of class {data_class} from the dataset")
        self.num_transferred_samples += len(samples)

        # Update size of dataset
        self.dataset_size = len(self.dataset)
        return samples

    # Used in the transfer function to receive data from a different device
    def add_data(self, sample):
        np.append(self.dataset, sample)
        # Update size of dataset
        self.dataset_size = len(self.dataset)