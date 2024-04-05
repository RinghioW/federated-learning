import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset

BATCH_SIZE = 32

def load_datasets(num_clients):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainsets = []
    valsets = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainsets.append(partition["train"])
        valsets.append(partition["test"])
    testset = fds.load_split("test").with_transform(apply_transforms)
    return trainsets, valsets, testset