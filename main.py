from argparse import ArgumentParser
from time import time
from datetime import timedelta
from device import Device
from user import User
from server import Server
import os
import nets
import plots
def main():
    
    # Define arguments
    parser = ArgumentParser(description="Heterogeneous federated learning framework using pytorch")

    parser.add_argument("-u", "--users", dest="users", type=int, default=3, help="Total number of users")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=12, help="Total number of devices")
    parser.add_argument("-s", "--dataset", dest="dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-c", "--compare", dest="compare", type=bool, default=False, help="Compare with centralized learning")
    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    num_devices = args.devices
    dataset = args.dataset
    server_epochs = args.epochs
    compare = args.compare

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_devices)
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    devices_per_user = num_devices // num_users
    users = [User(id=i, devices=[Device(j+(devices_per_user*i), trainsets.pop(), valsets.pop()) for j in range(devices_per_user)], testset=testset) for i in range(num_users)]
    
    # Print devices configuration
    for user in users:
        for device in user.devices:
            print(device)

    model = nets.AdaptiveCifar10CNN
    server = Server(dataset, model, users)

    time_start = time()

    # Evaluate the server model before training
    losses = []
    accuracies = []
    initial_loss, initial_accuracy = server.test(testset)
    losses.append(initial_loss)
    accuracies.append(initial_accuracy)
    

    print(f"S, e0 - Loss: {initial_loss: .4f}, Accuracy: {initial_accuracy: .3f}")
    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        
        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        server.train()
        
        # Server evaluates the model
        loss, accuracy = server.test(testset)

        losses.append(loss)
        accuracies.append(accuracy)

        print(f"S, e{epoch+1} - Loss: {loss: .4f}, Accuracy: {accuracy: .3f}")

    print(f"Elapsed Time: {str(timedelta(seconds=time() - time_start))}")
    # Plot the results
    plots.plot_devices(users)
    plots.plot_users(users)

    plots.plot_server(losses, accuracies)

if __name__ == "__main__":
    main()