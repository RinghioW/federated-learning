from argparse import ArgumentParser
from device import Device
from user import User
from server import Server
import os
from data import load_datasets
from config import DATASET
import nets
import torch
import random
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
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load dataset and split it according to the number of devices
    trainsets, testset = load_datasets(num_devices, DATASET)


    # Create models and train them all on the public dataset
    server_model = nets.LargeCifar100CNN
    torch.save(server_model().state_dict(), "checkpoints/server.pth")
    
    # Generate configs for devices
    devices = [[] for _ in range(num_users)]

    # Create devices
    for user in range(num_users):
        devices[user] = [Device(id=(num_devices//num_users)*user + i,
                                 trainset=trainsets[user][i], 
                                 testset=testset, 
                                 model=(nets.SmallCifar100CNN if random.random() > 0.5 else nets.MediumCifar100CNN)
                            ) for i in range(num_devices // num_users)]


    users = [User(id=i,
                  devices=devices[i],
                  testset=testset,
                  model=server_model) for i in range(num_users)]

    server = Server(model=server_model, users=users, testset=testset)

    # Evaluate the server model before training
    server.test()

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
    server.flush()
    for user in users:
        user.flush()
        for device in user.devices:
            device.flush()
    

if __name__ == "__main__":
    main()