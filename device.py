import torch
from config import DEVICE
import numpy as np
import math
import torchvision.transforms as transforms
class Device():
    def __init__(self, id) -> None:
        self.id = id
        self.config = None

        self.dataset = None
        self.valset = None # TODO: Figure out how to use this

        self.model = None # Model class (NOT instance)
        self.model_params = None
        self.path = None # Relative path to save the model
        self.has_checkpoint = False

        self.transition_matrix = None

        self.clusters = None # Clustered labels

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"
    
    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs=10, verbose=False):
        if len(self.dataset) == 0:
            return

        print(f"Device {self.config['id']} - Training on {len(self.dataset)} samples")
        # Load the model
        net = self.model(**self.model_params)
        optimizer = torch.optim.Adam(net.parameters())

        if self.has_checkpoint:
            checkpoint = torch.load(self.path + f"device_{self.config['id']}.pt")
            net.load_state_dict(checkpoint['model_state_dict'], strict=False, assign=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.has_checkpoint = True
        
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
                print(f"D{self.config['id']}, e{epoch+1} - Loss: {epoch_loss: .4f}, Accuracy: {epoch_acc: .3f}")

        # Save the model
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.path + f"device_{self.config['id']}.pt")

    
    # Function to sample a sub-dataset from the dataset
    # TODO: Implement uplink in a sensible way
    def sample(self, percentage, uplink=None):
        amount = math.floor(percentage * len(self.dataset))
        reduced_dataset = np.random.permutation(self.dataset)[:amount].tolist()
        return reduced_dataset

    def cluster(self, lda_estimator, kmeans_estimator):
        dataset = np.array(self.dataset["img"]).reshape(len(self.dataset), -1)
        feature_space = lda_estimator.transform(dataset)
        self.clusters = kmeans_estimator.predict(feature_space).tolist()
    
    def cluster_distribution(self):
        return np.bincount(self.clusters).tolist()

    def adapt(self, model, params, path):
        self.model = model
        self.model_params = params
        self.path = path

    def resources(self):
        return self.config["compute"] + self.config["memory"] + self.config["energy_budget"]

    # TODO: Figure out how to characterize the devices in a way that makes sense
    # The higher these numbers are, the higher the latency factor will be
    # If the latency is really high, this means that SL >> DI,
    # Meaning that data imbalance will not be accounted for
    # And devices will not share data with each other
    @staticmethod
    def generate_config(id):
        return {"id" : id,
                "compute" : np.random.randint(10**0, 10**1), # Compute capability in FLOPS
                "memory" : np.random.randint(10**0, 10**1), # Memory capability in Bytes
                "energy_budget" : np.random.randint(10**0,10**1), # Energy budget in J/hour
                "uplink_rate" : np.random.randint(10**0,10**1), # Uplink rate in Bps
                "downlink_rate" : np.random.randint(10**0,10**1) # Downlink rate in Bps
                }
    
    def update(self, trainset, valset, config):
        self.dataset = trainset
        self.valset = valset
        self.config = config