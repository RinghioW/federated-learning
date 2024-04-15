import torchvision.transforms
from flwr_datasets import FederatedDataset
import torch
BATCH_SIZE = 32

def load_datasets(num_clients):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients*100})

    # Create train/val for each partition and wrap it into DataLoader
    trainsets = []
    valsets = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.train_test_split(train_size=0.8)
        trainsets.append(partition["train"])
        valsets.append(partition["test"])
    testset = fds.load_split("test")
    return trainsets, valsets, testset

