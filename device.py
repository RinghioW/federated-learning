import torch
import numpy as np
import math
import torchvision.transforms as transforms
from scipy.spatial.distance import jensenshannon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        
        self.instantiated_model = None

    def __repr__(self) -> str:
        return f"Device({self.config}, 'samples': {len(self.dataset)})"
    
    # Perform on-device learning on the local dataset. This is simply a few rounds of SGD.
    def train(self, epochs):
        if len(self.dataset) == 0:
            return

        # Load the model
        net = self.instantiated_model
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


        to_tensor = transforms.ToTensor()
        dataset = self.dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=3)
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


    
    # Function to sample a sub-dataset from the dataset
    # TODO: Implement uplink in a sensible way
    def sample(self, percentage, uplink=None):
        amount = math.floor(percentage * len(self.dataset))
        reduced_dataset = np.random.permutation(self.dataset)[:amount].tolist()
        return reduced_dataset

    def cluster(self, lda_estimator, kmeans_estimator):
        if len(self.dataset) == 0:
            self.clusters = []
            return
        dataset = np.array(self.dataset["img"]).reshape(len(self.dataset), -1)
        feature_space = lda_estimator.transform(dataset)
        self.clusters = kmeans_estimator.predict(feature_space).tolist()
    
    def cluster_distribution(self):
        return np.bincount(self.clusters, minlength=NUM_CLASSES)

    def adapt(self, model, state_dict, params):
        net = model()
        quantize = params["quantize"]
        pruning_factor = params["pruning_factor"]
        low_rank = params["low_rank"]
        net.adapt(state_dict, pruning_factor=pruning_factor, quantize=quantize, low_rank=low_rank)
        self.instantiated_model = net

    def validate(self):
        if self.valset is None:
            return 0.
        net = self.instantiated_model
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
        return correct / total

    def resources(self):
        return self.memory_usage() + self.computation_time() + self.energy_usage()

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

    # Ram
    def memory_usage(self):
        return len(self.dataset) / self.config["memory"]

    def computation_time(self):
        return self.config["compute"]*len(self.dataset)
    
    def energy_usage(self):
        return len(self.dataset) / self.config["energy_budget"]
    
    def data_imbalance(self):
        distribution = np.bincount(self.dataset["label"], minlength=10)
        n_samples = sum(distribution)
        if n_samples == 0:
            return np.float64(0.)
        n_classes = len(distribution)
        avg_samples = n_samples / n_classes
        balanced_distribution = [avg_samples] * n_classes
        js = jensenshannon(balanced_distribution, distribution)
        return js

    