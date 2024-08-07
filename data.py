from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner, ExponentialPartitioner
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
    for partition_id in range(num_users):
        partition = fds.load_partition(partition_id, "train")
        if partition_id % 3 == 0:
            partitioner = PathologicalPartitioner(num_partitions=clients_per_user)
        elif partition_id % 3 == 1:
            partitioner = DirichletPartitioner(num_partitions=clients_per_user, alpha=0.3)
        elif partition_id % 3 == 2:
            partitioner = ExponentialPartitioner(num_partitions=clients_per_user)

        partitioner.dataset = partition
        for _ in range(clients_per_user):
            trainsets[partition_id].append(partitioner.load_partition(partition_id))

    return trainsets, fds.load_split("test")
