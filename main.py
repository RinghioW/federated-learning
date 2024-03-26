import numpy as np
import argparse
import time
from device import Device
from user import User
from server import Server
import flwr as fl

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
    server_epochs = args.epochs

    datasets = ["cifar10", "femnist", "shakespeare"]
    if dataset not in datasets:
        raise ValueError(f"Invalid dataset. Please choose from {datasets}")

    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    elif dataset == "femnist":
        from data.femnist import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    elif dataset == "shakespeare":
        from data.shakespeare import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)

    # Number of epochs that each device will train for
    on_device_epochs = 3

    # Create device configurations
    # TODO: Figure out how to characterize the devices in a way that makes sense
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


    server = Server(dataset)

    # Flower: Create a numpy client function
    def user_fn(cid) -> User:
        return users[cid]

    # Initialize transition matrices
    for user in users:
        for device in user.devices:
            device.initialize_transition_matrix(len(user.devices))
    
    # Evaluate the server model before training
    print("Evaluating server model before training...")
    initial_loss, initial_accuracy = server.evaluate(testset)
    print(f"Initial Loss: {initial_loss}, Initial Accuracy: {initial_accuracy}")

    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        print(f"FL epoch {epoch+1}/{server_epochs}")

        # Server performs selection of the users
        # ShuffleFL step 3
        # TODO: Can be done using Flower when specifying the strategy
        server.select_users(users)

        # Users report the staleness factor to the server, and
        # The server sends the adaptive scaling factor to the users
        # ShuffleFL step 4, 5
        # TODO : Implement it as a config in flower
        server.send_adaptive_scaling_factor()
            

        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        print(f"Updating server model...")
        server.aggregate_updates()
        
        # Server evaluates the model
        print(f"Evaluating trained server model...")
        loss, accuracy = server.evaluate(testset)
        print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")

    def evaluate_fn(
        server_round,
        parameters,
        config,
    ):
        # Evaluate the server model after training
        pass
        # Compute scaling factors for next round
        pass

    strategy = fl.server.strategy.FedAvg(evaluate_fn=evaluate_fn,)
    # Flower: Start simulation
    fl.simulation.start_simulation(
        client_fn=user_fn,
        num_clients=num_users,
        config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
        strategy=fl.server.strategy.FedAvg(),
        
    )

if __name__ == "__main__":
    main()