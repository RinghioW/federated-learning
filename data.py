from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision.transforms import ToTensor
import datasets
def load_datasets(num_clients, name, label_name="fine_label"):
    to_tensor = ToTensor()
    fds = FederatedDataset(
        dataset=name,
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_clients,
                partition_by=label_name,
                alpha=0.3,
            ),
        },
        preprocessor=lambda ds: ds.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"),
    )

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

def load_iid_datasets(num_clients, name, label_name="fine_label"):
    to_tensor = ToTensor()
    fds = FederatedDataset(
        dataset=name,
        partitioners={'train': num_clients},
        preprocessor=lambda ds: ds.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"),
    )

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

def load_cifar10():
    to_tensor = ToTensor()
    cifar10 = datasets.load_dataset("cifar10")
    cifar10 = cifar10.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
    return cifar10["train"]