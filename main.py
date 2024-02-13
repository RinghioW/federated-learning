import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from device import Device
from user import User
from server import Server

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    # Define arguments
    parser = argparse.ArgumentParser(description=f"Heterogeneous federated learning framework using pytorch.")
    parser.add_argument("-u", "--users", dest="users", type=int, default=10, help="Total number of users (default: 10)")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=30, help="Total number of devices (default: 30)")
    parser.add_argument("-s", "--dataset", dest="dataset", type=str, default="cifar10", help="Dataset to use (default: cifar10)")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs (default: 10)")
    print(parser.description)

    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    num_devices = args.devices
    dataset = args.dataset
    epochs = args.epochs

    datasets = ["cifar10"]
    if dataset not in datasets:
        raise ValueError(f"Invalid dataset. Please choose from {datasets}")
    
    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    else:
        pass # Add more datasets here
    
    # Create device configurations
    configs = [{"compute" : np.random.randint(1, 15), "memory" : np.random.randint(1, 15)} for _ in range(num_devices)]

    # Create devices and users
    devices = [Device(configs[i], trainsets[i], valsets[i]) for i in range(num_devices)]
    devices_grouped = np.array_split(devices, num_users)
    users = [User(devices_grouped[i]) for i in range(num_users)]
    server = Server(dataset)

    # The server sends the model to the users
    for user in users:
        # Each user adapts the model for their devices
        user.adapt_model(server.model)
    
    # Evaluate the server model before training
    print("Evaluating server model before training...")
    loss, accuracy = server.evaluate(testset)
    print(f"Initial Loss: {loss}, Initial Accuracy: {accuracy}")

    # Perform federated learning for the server model
    for epoch in range(epochs):
        print(f"FL epoch {epoch+1}/{epochs}")
        for user in users:
            user.shuffle_data()
            for device in user.devices:
                print(f"Training device {device}...")
                device.train(epochs=3)
            print(f"Aggregating updates from user {user}...")
            user.aggregate_updates()
        print(f"Updating server model...")
        server.aggregate_updates(users)
        print(f"Evaluating trained server model...")
        loss, accuracy = server.evaluate(testset)
        print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")

    print("Done.")

if __name__ == "__main__":
    main()