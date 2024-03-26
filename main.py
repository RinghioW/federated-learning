import numpy as np
import argparse
from device import Device
from user import User
import flwr as fl
import torch
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
STD_CORRECTION = 10

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
    device_configs = [{"compute" : np.random.randint(1, 15),
                "memory" : np.random.randint(1, 15),
                "energy_budget" : np.random.randint(1,15),
                "uplink_rate" : np.random.randint(1,15),
                "downlink_rate" : np.random.randint(1,15)
                } for _ in range(num_devices)]

    # Create devices and users
    devices = [Device(device_configs[i], trainsets[i], valsets[i]) for i in range(num_devices)]
    devices_grouped = np.array_split(devices, num_users)

    users = [User(devices_grouped[i]) for i in range(num_users)]
    # Flower: Create a numpy client function
    def user_fn(cid) -> User:
        return users[cid]

    # Flower: Define the server model
    if dataset == "cifar10":
        model = models.mobilenet_v3_large()

    def evaluate_fn(
        server_round,
        parameters,
        config,
    ):
        # Evaluate the server model after training
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=2)
        net = model
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
    
        # Compute estimated performances of the users
        estimated_performances = [user.diff_capability * config["wall_clock_training_times"][idx] for idx, user in enumerate(config["users"])]

        # Compute average user performance
        average_user_performance = sum(estimated_performances) / len(estimated_performances)

        # Compute adaptive scaling factor for each user
        for idx, user in enumerate(config["users"]):
            user.adaptive_scaling_factor = (average_user_performance / estimated_performances[idx]) * config["scaling_factor"]
        return loss, accuracy

    def on_fit_config_fn(server_round):
        pass

    def on_evaluate_config_fn(server_round):
        pass 

    strategy = fl.server.strategy.FedAvg(evaluate_fn=evaluate_fn,
                                         on_fit_config_fn=on_fit_config_fn,
                                         on_evaluate_config_fn=on_evaluate_config_fn,)

    # Flower: Start simulation
    fl.simulation.start_simulation(
        client_fn=user_fn,
        num_clients=num_users,
        config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    main()