import torch
import numpy as np
import argparse
import time
from device import Device
from user import User
from server import Server
from config import DEVICE
def main():
    
    # Define arguments
    parser = argparse.ArgumentParser(description=f"Heterogeneous federated learning framework using pytorch.")
    parser.add_argument("-u", "--users", dest="users", type=int, default=3, help="Total number of users (default: 3)")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=6, help="Total number of devices (default: 6)")
    parser.add_argument("-s", "--dataset", dest="dataset", type=str, default="cifar10", help="Dataset to use (default: cifar10)")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=2, help="Number of epochs (default: 2)")
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
    

    # Number of epochs for on-device trianing
    device_epochs = 3

    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    else:
        pass # Add more datasets here

    # Create device configurations
    configs = [{"compute" : np.random.randint(1, 15),
                "memory" : np.random.randint(1, 15),
                "energy_budget" : np.random.randint(1,15),
                "uplink_rate" : np.random.randint(1,15),
                "downlink_rate" : np.random.randint(1,15)
                } for _ in range(num_devices)]

    # Create devices and users
    devices = [Device(configs[i], trainsets[i], valsets[i]) for i in range(num_devices)]
    devices_grouped = np.array_split(devices, num_users)
    users = [User(devices_grouped[i]) for i in range(num_users)]
    server = Server(users, dataset)

    # Initialize transition matrices
    for user in users:
        for device in user.devices:
            device.initialize_transition_matrix(len(user.devices))

    time_start = time.time()
    
    # Evaluate the server model before training
    print("Evaluating server model before training...")
    loss, accuracy = server.evaluate(testset)
    print(f"Initial Loss: {loss}, Initial Accuracy: {accuracy}")

    # Perform federated learning for the server model
    for epoch in range(epochs):
        print(f"FL epoch {epoch+1}/{epochs}")
        # Server sends the model to the users
        for user_idx, user in enumerate(users):
            # The server sends the adaptive coefficient to the users
            server.adaptive_coefficient_users()

            # User adapts the model for their devices
            print(f"Adapting model for user {user_idx+1}/{len(users)}...")
            user.adapt_model(server.model)


            # Reduce dimensionality of the transmission matrices
            print(f"Reducing feature space for user {user_idx+1}...")
            user.reduce_dimensionality()
            
            # User optimizes the transmission matrices
            print(f"Optimizing transition matrices for user {user_idx+1}...")
            user.optimize_transmission_matrices()

            # User shuffles the data and creates a knowledge distillation dataset
            print(f"Shuffling data for user {user_idx+1}...")
            user.shuffle_data(user.transition_matrices)
            user.create_kd_dataset()
            
            # User updates parameters based on last iteration
            user.update_average_capability()

            # User measures the system latencies
            user.latency_devices(epochs=device_epochs)
            print(f"System latencies for user {user_idx}: {user.system_latencies}")

            # User measures the data imbalance
            user.data_imbalance_devices()
            print(f"Data imbalance for user {user_idx}: {user.data_imbalances}")

            # User trains devices
            user.train_devices(epochs=device_epochs, verbose=True)

            # User trains the model using knowledge distillation
            print(f"Aggregating updates from user {user_idx}...")
            user.aggregate_updates()

        print(f"Updating server model...")
        server.aggregate_updates(users)

        print(f"Evaluating trained server model...")
        loss, accuracy = server.evaluate(testset)
        print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")

    time_end = time.time()
    print(f"Elapsed time: {time_end - time_start} seconds. Done.")

if __name__ == "__main__":
    main()