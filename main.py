from argparse import ArgumentParser
from device import Device
from user import User
from server import Server
import os
from data import load_datasets
from config import DATASET
import nets
import torch
from data import load_datasets
def main():
    
    # Define arguments
    parser = ArgumentParser(description="Heterogeneous federated learning framework using pytorch")

    parser.add_argument("-u", "--users", dest="users", type=int, default=3, help="Total number of users")
    parser.add_argument("-d", "--devices-per-client", dest="devices", type=int, default=9, help="Total number of devices")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-s", "--sampling", dest="sampling", type=str, default="iid", help="Sampling method")

    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    devices_per_client = args.devices
    server_epochs = args.epochs
    sampling = args.sampling

    # Log
    results_dir = f"results/{sampling}"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset and split it according to the number of devices
    trainsets, testset = load_datasets(num_users, devices_per_client, DATASET)


    # Create models and train them all on the public dataset
    server_model = nets.LargeCifar100CNN
    torch.save(server_model().state_dict(), "checkpoints/server.pth")
    
    # Generate configs for devices
    devices = [[] for _ in range(num_users)]

    # Create devices
    for user in range(num_users):
        devices[user] = [Device(id=devices_per_client*user + i,
                                 trainset=trainsets[user][i], 
                                 testset=testset, 
                                 model=nets.assign_model(i)
                                ) for i in range(devices_per_client)]

    users = [User(id=i,
                  devices=devices[i],
                  testset=testset,
                  model=server_model,
                  sampling=sampling) for i in range(num_users)]

    server = Server(model=server_model, users=users, testset=testset)


    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        
        print(f"EPOCH {epoch}")
        # Server aggregates the updates from the users
        server.train()

        # Server evaluates the model
        server.test()

    # Save the results
    with open(f"{results_dir}/server-accuracies.txt", "w") as f:
        f.write(f"{server.log}\n")
    with open(f"{results_dir}/latencies.txt", "w") as f:
        for user in users:
            f.write(f"{user.latency}\n")
    with open(f"{results_dir}/user-accuracies.txt", "w") as f:
        for user in users:
            f.write(f"{user.log}\n")
    with open(f"{results_dir}/device-accuracies.txt", "w") as f:
        for user in users:
            for device in user.devices:
                f.write(f"{device.log}\n")
    

if __name__ == "__main__":
    main()