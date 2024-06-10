from argparse import ArgumentParser
from time import time
from datetime import timedelta
from device import Device
from user import User
from server import Server
from adaptivenet import AdaptiveNet
import plots
def main():
    
    # Define arguments
    parser = ArgumentParser(description=f"Heterogeneous federated learning framework using pytorch")

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

    results_dir = "results/final/"

    # Load dataset and split it according to the number of devices
    if dataset == "cifar10":
        from data.cifar10 import load_datasets
        trainsets, valsets, testset = load_datasets(num_clients=num_devices, epochs=server_epochs+1)
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    users = [User(id=i, devices=[Device(j) for j in range(num_devices//num_users)]) for i in range(num_users)]
    model = AdaptiveNet
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
        # Update the data and resources on the devices
        for user in users:
            for device in user.devices:
                device.update(trainset=trainsets.pop(), valset=valsets.pop(), config=Device.generate_config(device.id))
        
        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        server.train()
        
        # Server evaluates the model
        loss, accuracy = server.test(testset)

        losses.append(loss)
        accuracies.append(accuracy)

        print(f"S, e{epoch+1} - Loss: {loss: .4f}, Accuracy: {accuracy: .3f}")


    time_end = time()
    print(f"Elapsed Time: {str(timedelta(seconds=time_end - time_start))}")
    
    # Save the results
    plots.plot_training("results/final/server/", losses, accuracies)


if __name__ == "__main__":
    main()