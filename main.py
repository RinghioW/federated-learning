import numpy as np
import argparse
import time
from device import Device
from user import User
from server import Server

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

    # Initialize transition matrices
    for user in users:
        for device in user.devices:
            device.initialize_transition_matrix(len(user.devices))

    time_start = time.time()
    
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
        server.select_users(users)

        # Users report the staleness factor to the server, and
        # The server sends the adaptive scaling factor to the users
        # ShuffleFL step 4, 5
        server.send_adaptive_scaling_factor()

        # ShuffleFL step 6
        for user_idx, user in enumerate(server.users):
            
            # User adapts the model for their devices
            # ShuffleFL Novelty
            print(f"Adapting model for user {user_idx+1}/{len(users)}...")
            user.adapt_model(server.model)

            # Reduce dimensionality of the transmission matrices
            # ShuffleFL step 7, 8
            print(f"Reducing feature space for user {user_idx+1}...")
            user.reduce_dimensionality()
            
            # User optimizes the transmission matrices
            # ShuffleFL step 9
            print(f"Optimizing transition matrices for user {user_idx+1}...")
            user.optimize_transmission_matrices()

            # User shuffles the data
            # ShuffleFL step 10
            print(f"Shuffling data for user {user_idx+1}...")
            user.shuffle_data(user.transition_matrices)

            # User creates the knowledge distillation dataset
            # ShuffleFL Novelty
            print(f"Creating knowledge distillation dataset for user {user_idx+1}...")
            user.create_kd_dataset()

            # User updates parameters based on last iteration
            user.update_average_capability()

            # User measures the system latencies
            user.latency_devices(epochs=on_device_epochs)
            print(f"System latencies for user {user_idx}: {user.system_latencies}")

            # User measures the data imbalance
            user.data_imbalance_devices()
            print(f"Data imbalance for user {user_idx}: {user.data_imbalances}")

            # User trains devices
            # ShuffleFL step 11-15
            user.train_devices(epochs=on_device_epochs, verbose=True)

            # User trains the model using knowledge distillation
            # ShuffleFL step 16, 17
            print(f"Aggregating updates from user {user_idx}...")
            user.aggregate_updates()

        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        print(f"Updating server model...")
        server.aggregate_updates()
        
        # Server evaluates the model
        print(f"Evaluating trained server model...")
        loss, accuracy = server.evaluate(testset)
        print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")

    time_end = time.time()
    print(f"Elapsed time: {time_end - time_start} seconds.")
    print(f"Accuracy improvement: {accuracy - initial_accuracy}")

if __name__ == "__main__":
    main()