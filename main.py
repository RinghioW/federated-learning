import numpy as np
import argparse
import time
from device import Device
from user import User
from server import Server
from config import Style
from plots import plot_results

def main():

    # Define arguments
    parser = argparse.ArgumentParser(description=f"Heterogeneous federated learning framework using pytorch.")
    print(parser.description)

    parser.add_argument("-u", "--users", dest="users", type=int, default=4, help="Total number of users (default: 2)")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=12, help="Total number of devices (default: 6)")
    parser.add_argument("-s", "--dataset", dest="dataset", type=str, default="cifar10", help="Dataset to use (default: cifar10)")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs (default: 2)")
    parser.add_argument("--shuffle", dest="shuffle", type=bool, default=True, help="Enable data shuffling")
    parser.add_argument("--adapt", dest="adapt", type=bool, default=True, help="Enable model adaptation")

    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    num_devices = args.devices
    dataset = args.dataset
    server_epochs = args.epochs
    shuffle = args.shuffle
    adapt = args.adapt

    if not shuffle:
        results_dir = "results/no-shuffle/"
    else:
        results_dir = "results/final/"

    datasets = ["cifar10", "femnist", "shakespeare"]
    if dataset not in datasets:
        raise ValueError(f"Invalid dataset. Please choose from {datasets}")

    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_clients=num_devices, epochs=server_epochs)
    elif dataset == "femnist":
        from data.femnist import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    elif dataset == "shakespeare":
        from data.shakespeare import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)

    # Create device configurations
    # TODO: Figure out how to characterize the devices in a way that makes sense
    # The higher these numbers are, the higher the latency factor will be
    # If the latency is really high, this means that SL >> DI,
    # Meaning that data imbalance will not be accounted for
    # And devices will not share data with each other
    configs = Device.generate_configs(num_devices)
    devices = [Device(configs[i], trainsets[i*server_epochs], valsets[i*server_epochs]) for i in range(num_devices)]
    devices_grouped = np.array_split(devices, num_users)
    users = [User(id=i, devices=devices_grouped[i]) for i in range(num_users)]
    server = Server(dataset, users)

    time_start = time.time()
    
    # Evaluate the server model before training
    losses = []
    accuracies = []
    initial_loss, initial_accuracy = server.test(testset)
    losses.append(initial_loss)
    accuracies.append(initial_accuracy)
    
    latency_histories = [[] for _ in range(num_users)]
    data_imbalance_histories = [[] for _ in range(num_users)]
    obj_functions = [[] for _ in range(num_users)]
    
    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        print(f"{Style.YELLOW}FL epoch {epoch+1}/{server_epochs}{Style.RESET}")

        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        server.train()
        
        # Server evaluates the model
        loss, accuracy = server.test(testset)
        print(f"{Style.YELLOW}SERVER :{Style.RESET} Loss: {loss}, Accuracy: {accuracy}")
        losses.append(loss)
        accuracies.append(accuracy)

        # Users change their available compute and memory 
        configs = Device.generate_configs(num_devices)
        
        # Update the data on the devices
        for user_idx, user in enumerate(users):
            for device_idx, device in enumerate(user.devices):
                idx = user_idx*num_devices//num_users + device_idx
                device.dataset = trainsets[idx*server_epochs + epoch]
                device.valset = valsets[idx*server_epochs + epoch]
                user.devices[device_idx] = device
                device.config = configs[device.config["id"]]

    # Save the results
    plot_results(results_dir, num_users, latency_histories, data_imbalance_histories, obj_functions, losses, accuracies)

    time_end = time.time()
    print(f"Elapsed time: {time_end - time_start} seconds.")

if __name__ == "__main__":
    main()