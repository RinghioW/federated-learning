import torch
from config import DEVICE, NUM_CLASSES
import numpy as np
from utils import js_divergence
class Device():
    def __init__(self, config, dataset, valset) -> None:
        self.config = config
        self.dataset = dataset
        self.valset = valset
        self.model = None
        self.transition_matrix = None

    def __repr__(self) -> str:
        return f"Device(config={self.config})"
    
    def data_imbalance(self):
        # Calculate the reference data distribution
        # The reference data distribution is a balanced distribution, all classes have the same number of samples
        reference_distribution = [len(self.dataset)/NUM_CLASSES for _ in range(NUM_CLASSES)]

        # Compute the JS divergence between the reference distribution and the actual data distribution
        dataloader = torch.utils.data.DataLoader(self.dataset)
        distribution = []
        for item in dataloader:
            distribution.append(item["label"])
        return js_divergence(np.array(reference_distribution), np.array(distribution))
    

    def latency(self, device_idx, devices, epochs):
        # Communication depends on the transition matrix
        t_communication = 0
        for data_class_idx, data_class in enumerate(self.transition_matrix):
            for other_device_idx, other_device in enumerate(devices):
                if device_idx != other_device_idx:
                    # Transmitting
                    t_communication += self.transition_matrix[data_class_idx][other_device_idx] * ((1/self.config["uplink_rate"]) + (1/other_device.config["downlink_rate"]))
                    # Receiving
                    t_communication += other_device.transition_matrix[data_class_idx][device_idx] * ((1/self.config["downlink_rate"]) + (1/other_device.config["uplink_rate"]))
        t_computation = 0
        t_computation += 3 * epochs * len(self.dataset) * self.config["compute"]
        return t_communication + t_computation

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