from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner, ExponentialPartitioner, IidPartitioner
from torchvision.transforms import ToTensor

# Partition the dataset into num_user parts (IID)
# Each user then partitions their data in non-IID manner
def load_datasets(num_users, clients_per_user, name):
    to_tensor = ToTensor()
    fds = FederatedDataset(
        dataset=name,
        partitioners={
            "train": num_users,
        },
        preprocessor=lambda ds: ds.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"),
    )

    # For each train partition, create a train/val split
    trainsets = [[] for _ in range(num_users)]
    i = 10
    for partition_id in range(num_users):
        partition = fds.load_partition(partition_id, "train")
        if partition_id % i == 0:
            # This could be done with partition by super class
            partitioner = PathologicalPartitioner(num_partitions=clients_per_user, partition_by="fine_label", num_classes_per_partition=40)
        elif partition_id % i == 1:
            partitioner = DirichletPartitioner(num_partitions=clients_per_user, partition_by="fine_label", alpha=0.3)
        elif partition_id % i == 2:
            partitioner = ExponentialPartitioner(num_partitions=clients_per_user)
        elif partition_id % i == 3:
            partitioner = IidPartitioner(num_partitions=clients_per_user)
        elif partition_id % i == 4:
            partitioner = PathologicalPartitioner(num_partitions=clients_per_user, partition_by="coarse_label", num_classes_per_partition=10)

        partitioner.dataset = partition
        for k in range(clients_per_user):
            dataset = partitioner.load_partition(k)

            trainsets[partition_id].append(dataset)

    return trainsets, fds.load_split("test")


to_tensor = ToTensor()
ds= FederatedDataset(
    dataset="cifar100",
    preprocessor=lambda ds: ds.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch"),
    partitioners={"train": PathologicalPartitioner(num_partitions=1, partition_by="fine_label", num_classes_per_partition=25)},
).load_split("train").shuffle().select(range(4000))
def fedmd():
    return ds