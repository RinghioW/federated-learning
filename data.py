from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner, ExponentialPartitioner, IidPartitioner
from torchvision.transforms import ToTensor

# Partition the dataset into num_user parts (IID)
# Each user then partitions their data in non-IID manner
def load_datasets(num_users, clients_per_user, name):
    print(f"No. of users: {num_users}, clients per user: {clients_per_user}")
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
            partitioner = PathologicalPartitioner(num_partitions=clients_per_user, partition_by="fine_label", num_classes_per_partition=100//clients_per_user)
        elif partition_id % i == 1:
            partitioner = DirichletPartitioner(num_partitions=clients_per_user, partition_by="fine_label", alpha=0.3)
        elif partition_id % i == 2:
            partitioner = ExponentialPartitioner(num_partitions=clients_per_user)
        elif partition_id % i == 3:
            partitioner = IidPartitioner(num_partitions=clients_per_user)

        partitioner.dataset = partition
        for k in range(clients_per_user):
            dataset = partitioner.load_partition(k)
            print(len(dataset))
            trainsets[partition_id].append(dataset)
        

    return trainsets, fds.load_split("test")
