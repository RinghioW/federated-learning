import torch
import numpy as np
import math
import torchvision.transforms as transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Add specifications for Jetson Nano
# https://developer.nvidia.com/embedded/jetson-modules
# https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
JETSON_NANO_CONFIG = {
    'name' : 'Jetson Nano',
    'compute' : 472, # GFLOPS
    'memory': 4, # GB
    'energy_budget': 10, # W
}

JETSON_XAVIER_NX_8GB_CONFIG = {
    'name' : 'Jetson Xavier NX 8GB',
    'compute' : 21, # TOPS
    'memory': 8, # GB
    'energy_budget': 20, # W
}

RASPBERRY_PI_4_CONFIG = {
    'name' : 'Raspberry Pi 4 Model B',
    'compute' : 9.69, # GFLOPS
    'memory': 4, # GB
    'energy_budget': 7, # W
}

NUM_CLASSES = 10


class Device():
    def __init__(self, id, dataset=None, valset=None) -> None:
        self.id = id
        self.config = Device.generate_config(id)

        self.dataset = dataset
        self.valset = valset

        self.model = None # Model class (NOT instance)
        self.model_params = None

        self.clusters = None # Clustered labels

        self.training_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"
    
    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs):
        if len(self.dataset) == 0:
            return

        # Load the model
        net = self.model(**self.model_params)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

        checkpoint = torch.load(f"checkpoints/device_{self.config['id']}.pt")
        net.load_state_dict(checkpoint['model_state_dict'], strict=False, assign=True)
        
        net.train()

        to_tensor = transforms.ToTensor()
        dataset = self.dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=3)
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            correct, total, epoch_loss, epoch_acc = 0, 0, 0.0, 0.0
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if len(trainloader.dataset) != 0:
                epoch_loss /= len(trainloader.dataset)
            if total != 0:
                epoch_acc = correct / total
            self.training_losses.append(epoch_loss)
            self.training_accuracies.append(epoch_acc)

        # Save the model
        torch.save({
            'model_state_dict': net.state_dict(),
        }, f"checkpoints/device_{self.config['id']}.pt")

    
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
        return np.bincount(self.clusters, minlength=NUM_CLASSES)

    # TODO : Implement this function on-the-fly
    def adapt(self, model, state_dict, params):
        
        self.model = model
        torch.save({
            'model_state_dict': state_dict,
        }, f"checkpoints/device_{self.config['id']}.pt")
        self.model_params = params

    def validate(self):
        if self.valset is None:
            return 0
        net = self.model(**self.model_params)
        checkpoint = torch.load(f"checkpoints/device_{self.config['id']}.pt")
        net.load_state_dict(checkpoint['model_state_dict'], strict=False, assign=True)
        net.eval()
        to_tensor = transforms.ToTensor()
        dataset = self.valset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        valloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in valloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
        self.validation_accuracies.append(correct / total)
        return correct / total

    def resources(self):
        return self.config["compute"] + self.config["memory"] + self.config["energy_budget"]

    # TODO: Characterize the devices in a way that makes sense
    # The higher these numbers are, the higher the latency factor will be
    # If the latency is really high, this means that SL >> DI,
    # Meaning that data imbalance will not be accounted for
    # And devices will not share data with each other
    @staticmethod
    def generate_config(id):
        # TODO : Don't generate randomly but use the specs of actual devices
        return {"id" : id,
                "compute" : 1. + np.random.rand(), # Compute capability in FLOPS
                "memory" : 1. + np.random.rand(), # Memory capability in Bytes
                "energy_budget" : 1. + np.random.rand(), # Energy budget in J/hour
                "uplink_rate" : 1. + np.random.rand(), # Uplink rate in Bps
                "downlink_rate" : 1. + np.random.rand() # Downlink rate in Bps
                }
    
    def update(self, trainset, valset, config):
        self.dataset = trainset
        self.valset = valset
        self.config = config