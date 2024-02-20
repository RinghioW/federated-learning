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
        reference_distribution = np.array([len(self.dataset)/NUM_CLASSES for _ in range(NUM_CLASSES)])

        # Compute the JS divergence between the reference distribution and the actual data distribution
        return js_divergence(reference_distribution["labels"], self.dataset)
    
    def latency(self, epochs):
        t_communication = self.config["uplink"] + self.config["downlink"]
        t_computation = 3 * epochs * len(self.dataset) * self.config["compute"]
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