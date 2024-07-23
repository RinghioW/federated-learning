from argparse import ArgumentParser
from device import Device
from user import User
from server import Server
import os
from copy import deepcopy
from data import load_datasets
import nets

LABEL_NAME = "fine_label"
NUM_CLASSES = 100
def main():
    
    # Define arguments
    parser = ArgumentParser(description="Heterogeneous federated learning framework using pytorch")

    parser.add_argument("-u", "--users", dest="users", type=int, default=3, help="Total number of users")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=9, help="Total number of devices")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs")
    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    num_devices = args.devices
    server_epochs = args.epochs

    # Log
    os.makedirs("results", exist_ok=True)

    # Load dataset and split it according to the number of devices
    dataset = "cifar100"
    trainsets, valsets, testset = load_datasets(num_devices, dataset)

    server_model = nets.LargeCifar100CNN
    # Generate configs for devices
    devices = [Device(id=i,
                       trainset=trainsets.pop(),
                       valset=valsets.pop())
                for i in range(num_devices)]

    # Order devices by resources
    large_model = nets.MediumCifar100CNN()
    small_model = nets.SmallCifar100CNN()
    # The top half of devices are given a large model, and the bottom half are given a small model
    # TODO: This should be done at the user level (probably)
    for i, device in enumerate(devices):
        if i < num_devices // 2:
            device.model = deepcopy(small_model)
        else:
            device.model = deepcopy(large_model)

    users = [User(id=i,
                  devices=[devices.pop() for _ in range(num_devices // num_users)],
                  testset=testset) for i in range(num_users)]

    server = Server(model=server_model, users=users, testset=testset)

    # Evaluate the server model before training
    server.test()

    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        
        print(f"EPOCH {epoch}")
        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        server.train()

        # Server evaluates the model
        server.test()



if __name__ == "__main__":
    main()