import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.profiler
import optimum.quanto as quanto
import os
import random
import time
from data.cifar10 import load_datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
class Target:

    def __init__(self, id, compute, memory, energy_budget):
        self.id = id
        self.compute = compute
        self.memory = memory
        self.energy_budget = energy_budget

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
    models = [RASPBERRY_PI_4_CONFIG, JETSON_NANO_CONFIG, JETSON_XAVIER_NX_8GB_CONFIG]

    @staticmethod
    def generate(id):
        model = random.choice(Target.models)
        return Target(id, model['compute'], model['memory'], model['energy_budget'])
    

class AdaptiveCifar10CNN(nn.Module):

    def __init__(self):
        super(AdaptiveCifar10CNN, self).__init__()


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def _quantize(self):
        quanto.quantize(self, weights=quanto.qint8)
    
    def _prune(self, pruning_factor):
        # TODO: Weights should be the same as when the network was first initialized on the device for pruning to be consistent
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(module=layer, name="weight", amount=pruning_factor, n=2, dim=0)
                prune.remove(layer, name="weight")
    
    def _low_rank(self, rank=10):
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self._svd(layer, rank)

    def _svd(self, layer, rank):
        with torch.no_grad():
            W = layer.weight
            U, S, V = torch.svd(W.flatten(1))  # SVD on 2D tensor

            # Reduce the number of singular values/vectors
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]

            # Replace the current layer (linear or convolutional) with 3 layers performing the SVD
            # TODO
            pass


            if layer.bias is not None:
                # Bias compression is typically not done, but could be added if needed
                pass

        return layer

    def quantize(self):
        self._quantize()

    def prune(self, pruning_factor):
        self._prune(pruning_factor)
    
    def low_rank(self, rank=10):
        self._low_rank(rank)



    def forward(self, x):

        # In case of low rank, need to multiply the U, S and Vh matrices

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = F.softmax(self.fc2(x))

        return x
    
    # Function to measure the size of the model
    def model_size(self):
        s = sum(p.numel() * p.element_size() for p in self.parameters())
        torch.save(self.state_dict(), "temp.pt")
        z = os.path.getsize("temp.pt")
        os.remove("temp.pt")
        return (s, z)


    # Function to measure the energy consumption of the model
    def energy_consumption(self, target: Target):
        pass

    # Function to measure the latency of the model
    def latency(self, target: Target):
        pass


def profile_inference(trainset, model, name):
    name = name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=3)
    model.eval()
    res = None
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", name),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        for batch in dataloader:
            prof.step()
            images = batch["img"]
            model(images)

    res = prof.key_averages().table(sort_by="self_cpu_time_total")
    return res

def profile_train(trainset, model, name):
    name = name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    res = None
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=5, active=10, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", name),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        for batch in dataloader:
            prof.step()
            images, labels = batch["img"], batch["label"]
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    res = prof.key_averages().table(sort_by="self_cpu_time_total")
    return res

# Create different targets

os.system('rm log/*')
# targets = [Target.generate(i) for i in range(3)]

# Create the model in different configurations
default_model = AdaptiveCifar10CNN()
pruned_model = AdaptiveCifar10CNN()
pruned_model.prune(0.5)
quantized_model = AdaptiveCifar10CNN()
quantized_model.quantize()
low_rank_model = AdaptiveCifar10CNN()
low_rank_model.low_rank()

# Get a batch (32 samples) of CIFAR10
trainset, valset, testset = load_datasets(10)

trainset = trainset[0]


# Profile the models
to_tensor = transforms.ToTensor()
trainset = trainset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")

res = profile_inference(trainset, default_model, "default")
print(res)

res= profile_inference(trainset, pruned_model, "pruned")
print(res)

res= profile_inference(trainset, quantized_model, "quantized")
print(res)

res =profile_inference(trainset, low_rank_model, "low_rank")
print(res)
