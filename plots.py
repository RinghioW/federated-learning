import json
import os
import matplotlib.pyplot as plt

# Function to plot the log.json file
    # Epoch: id
    # |--- "Server"
    #     |--- Accuracy: float
    #     |--- Loss: float
    #     |--- Latency: float
    # |--- User: id
    #     |--- Device: id
    #         |--- Config
    #         |--- Accuracy: float
    #         |--- Loss
    #         |--- Latency
    #         |--- Data Imbalance
    #         |--- Energy Usage
    #         |--- Memory Usage
    #     |--- "Optimization"
    #       |--- Step: int
    #           |--- Latency
    #           |--- Data Imbalance
    #           |--- Objective function

def plot(spath):
    with open(os.path.join(spath, "log.json"), "r") as f:
        log = json.load(f)

        # Plot server metrics
        # Extract epochs
        epochs = list(log.keys())[1:]

        # Server metrics
        accuracies = [log[epoch]["server"]["accuracy"] for epoch in epochs]
        losses = [log[epoch]["server"]["loss"] for epoch in epochs]
        latencies = [log[epoch]["server"]["latency"] for epoch in epochs]

        plt.title("Server Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(epochs, accuracies)
        plt.savefig(os.path.join(spath, "server_accuracy.png"))
        plt.close()

        plt.title("Server Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, losses)
        plt.savefig(os.path.join(spath, "server_loss.png"))
        plt.close()

        plt.title("Server Latency")
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.plot(epochs, latencies)
        plt.savefig(os.path.join(spath, "server_latency.png"))
        plt.close()

        for user_id, user in log["0"]["users"].items():
            accuracies = [log[epoch]["users"][user_id]["test_accuracy"] for epoch in epochs]
            # kd_losses = [log[epoch]["users"][user_id]["train_kd_loss"] for epoch in epochs]
            # ce_losses = [log[epoch]["users"][user_id]["train_ce_loss"] for epoch in epochs]
            # train_accuracies = [log[epoch]["users"][user_id]["train_accuracy"] for epoch in epochs]
            for device_id, device in user["devices"].items():
                accuracies = [log[epoch]["users"][user_id]["devices"][device_id]["accuracy"] for epoch in epochs]
                losses = [log[epoch]["users"][user_id]["devices"][device_id]["loss"] for epoch in epochs]
                # latencies = [log[epoch]["users"][user_id]["devices"][device_id]["latency"] for epoch in epochs]
                # data_imbalances = [log[epoch]["users"][user_id]["devices"][device_id]["data_imbalance"] for epoch in epochs]
                # energy_usages = [log[epoch]["users"][user_id]["devices"][device_id]["energy_usage"] for epoch in epochs]
                # memory_usages = [log[epoch]["users"][user_id]["devices"][device_id]["memory_usage"] for epoch in epochs]

                # Plot device metrics
                plt.title(f"User {user_id} Device {device_id} Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.plot(epochs, accuracies)
                plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_accuracy.png"))
                plt.close()

                plt.title(f"User {user_id} Device {device_id} Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(epochs, losses)
                plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_loss.png"))
                plt.close()

            #     plt.title(f"User {user_id} Device {device_id} Latency")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Latency")
            #     plt.plot(epochs, latencies)
            #     plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_latency.png"))
            #     plt.close()

            #     plt.title(f"User {user_id} Device {device_id} Data Imbalance")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Data Imbalance")
            #     plt.plot(epochs, data_imbalances)
            #     plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_data_imbalance.png"))
            #     plt.close()

            #     plt.title(f"User {user_id} Device {device_id} Energy Usage")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Energy Usage")
            #     plt.plot(epochs, energy_usages)
            #     plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_energy_usage.png"))
            #     plt.close()

            #     plt.title(f"User {user_id} Device {device_id} Memory Usage")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Memory Usage")
            #     plt.plot(epochs, memory_usages)
            #     plt.savefig(os.path.join(spath, f"user_{user_id}_device_{device_id}_memory_usage.png"))
            #     plt.close()

                

            plt.title(f"User {user_id} Test Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(epochs, accuracies)
            plt.savefig(os.path.join(spath, f"user_{user_id}_test_accuracy.png"))
            plt.close()

            # plt.title(f"User {user_id} KD Loss")
            # plt.xlabel("Epoch")
            # plt.ylabel("Loss")
            # plt.plot(epochs, kd_losses, color="orange")
            # plt.plot(epochs, ce_losses, color="blue")
            # plt.legend(["KD Loss", "CE Loss"])
            # plt.savefig(os.path.join(spath, f"user_{user_id}_train_loss.png"))
            # plt.close()

            # plt.title(f"User {user_id} Train Accuracy")
            # plt.xlabel("Epoch")
            # plt.ylabel("Accuracy")
            # plt.plot(epochs, train_accuracies)
            # plt.savefig(os.path.join(spath, f"user_{user_id}_train_accuracy.png"))
            # plt.close()


            # # Plot optimization
            # objs = [log[epoch]["users"][user_id]["optimization"]["objective"] for epoch in epochs]
            # latencies = [log[epoch]["users"][user_id]["optimization"]["latency"] for epoch in epochs]
            # data_imbalances = [log[epoch]["users"][user_id]["optimization"]["data_imbalance"] for epoch in epochs]
            # for ob, la, d, i in zip(objs, latencies, data_imbalances, range(len(objs))):
            #     # Plot objective function
            #     plt.title(f"User {user_id} Objective Function")
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Objective Function")
            #     plt.plot(ob)
            #     plt.plot(la)
            #     plt.plot(d)
            #     plt.legend(["Objective Function", "Latency", "Data Imbalance"])
            #     plt.savefig(os.path.join(spath, f"user_{user_id}_objective_{i}.png"))
            #     plt.close()


with open("log.json", "r") as f, open("no_adaptation_log.json", "r") as d:
    log = json.load(f)
    dlog = json.load(d)
    # Plot server metrics
    # Extract epochs
    epochs = list(log.keys())[1:]

    # Server metrics
    accuracies = [log[epoch]["server"]["accuracy"] for epoch in epochs]
    d_accuracies = [dlog[epoch]["server"]["accuracy"] for epoch in epochs]

    plt.title("Server Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, accuracies, color="orange")
    plt.plot(epochs, d_accuracies, color="blue")
    plt.legend(["Adaptation", "No Adaptation"])
    plt.savefig("comparison.png")
    plt.close()

    plt.title("Server Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, [log[epoch]["server"]["loss"] for epoch in epochs], color="orange")
    plt.plot(epochs, [dlog[epoch]["server"]["loss"] for epoch in epochs], color="blue")
    plt.legend(["Adaptation", "No Adaptation"])
    plt.savefig("comparison_loss.png")
    plt.close()