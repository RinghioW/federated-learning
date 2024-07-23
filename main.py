from argparse import ArgumentParser
from device import Device
from user import User
from server import Server
from log import Logger
import os
import nets
from data import load_datasets
def main():
    
    # Define arguments
    parser = ArgumentParser(description="Heterogeneous federated learning framework using pytorch")

    parser.add_argument("-u", "--users", dest="users", type=int, default=3, help="Total number of users")
    parser.add_argument("-d", "--devices", dest="devices", type=int, default=9, help="Total number of devices")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-n", "--no-adaptation", dest="no_adaptation", action="store_true", help="Run without adaptation and shuffling")
    # Parse arguments
    args = parser.parse_args()
    num_users = args.users
    num_devices = args.devices
    server_epochs = args.epochs
    no_adaptation = args.no_adaptation

    # Log
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    if no_adaptation:
        os.makedirs("results/no_adaptation_no_shuffle", exist_ok=True)
        logger = Logger(log_dir="results/no_adaptation_no_shuffle")
    else:
        logger = Logger(log_dir="results")

    # Load dataset and split it according to the number of devices
    trainsets, valsets, testset = load_datasets(num_devices)


    devices_per_user = num_devices // num_users
    users = [User(id=i, 
                  devices=[Device(j+(devices_per_user*i), trainsets.pop(), valsets.pop()) for j in range(devices_per_user)],
                  testset=testset) for i in range(num_users)]

    server = Server(nets.AdaptiveCifar10CNN, users, logger)

    # Evaluate the server model before training
    initial_loss, initial_accuracy = server.test(testset)

    print(f"S, init - Loss: {initial_loss: .4f}, Accuracy: {initial_accuracy: .3f}")
    # Perform federated learning for the server model
    # Algorithm 1 in ShuffleFL
    # ShuffleFL step 1, 2
    for epoch in range(server_epochs):
        
        # Server aggregates the updates from the users
        # ShuffleFL step 18, 19
        if no_adaptation:
            server.train_no_adaptation_no_shuffle()
        else:
            server.train()

        # Server evaluates the model
        loss, accuracy = server.test(testset)


        print(f"S, e{epoch} - Loss: {loss: .4f}, Accuracy: {accuracy: .3f}")

    logger.dump()


    # Run without adaptation and shuffling



if __name__ == "__main__":
    main()