import torch
import torch.nn as nn
from nets import AdaptiveCifar10CNN
import torch.profiler
import os
import random
from data.cifar10 import load_datasets
import torchvision.transforms as transforms
import datetime
from torch.utils.tensorboard import SummaryWriter
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
    


# Function to measure the size of the model
def model_size(model):
    s = sum(p.numel() * p.element_size() for p in model.parameters())
    torch.save(model.state_dict(), "temp.pt")
    z = os.path.getsize("temp.pt")
    os.remove("temp.pt")
    return (s, z)


# Function to measure the energy consumption of the model
def energy_consumption(model, target: Target):
    pass

# Function to measure the latency of the model
def latency(model, target: Target):
    pass

writer = SummaryWriter("log")

def profile_inference(trainset, model, name):
    name = name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    model.eval()
    res = None
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", name),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        e= 0
        for batch in dataloader:
            prof.step()
            images, labels = batch["img"] , batch["label"]
            output = model(images)
            accuracy = (output.argmax(1) == labels).float().mean()
            writer.add_scalar("Accuracy/test", accuracy.item(), e)
            e+=1

    res = prof.key_averages().table(sort_by="self_cpu_time_total")
    prof.export_chrome_trace("log/" + name + "_inference.json")
    prof.export_memory_timeline("log/" + name + "_inference_memory.json")
    return res



def profile_train(trainset, model, epochs, name):
    name = name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    res = None
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=5, active=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log", name),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        for e in range(epochs):
            for batch in dataloader:
                prof.step()
                images, labels = batch["img"], batch["label"]
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), e)

    res = prof.key_averages().table(sort_by="self_cpu_time_total")

    writer.flush()
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

# res = profile_inference(trainset, default_model, "default")
# print(res)

# res= profile_inference(trainset, pruned_model, "pruned")
# print(res)

# res= profile_inference(trainset, quantized_model, "quantized")
# print(res)

# res =profile_inference(trainset, low_rank_model, "low_rank")
# print(res)

epochs = 10
res = profile_train(trainset, default_model, epochs, "default")
print(res)

# res= profile_train(trainset, pruned_model, "pruned")
# print(res)

# res= profile_train(trainset, quantized_model, "quantized")
# print(res)

# res =profile_train(trainset, low_rank_model, "low_rank")
# print(res)

writer.close()