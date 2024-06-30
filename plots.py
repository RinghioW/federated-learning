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
def plot():
    with open(os.path.join("results", "log.json"), "r") as f:
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
        plt.savefig(os.path.join("results", "server_accuracy.png"))
        plt.close()

        plt.title("Server Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, losses)
        plt.savefig(os.path.join("results", "server_loss.png"))
        plt.close()

        plt.title("Server Latency")
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.plot(epochs, latencies)
        plt.savefig(os.path.join("results", "server_latency.png"))
        plt.close()

        # for user_id, user in log[0]["users"].items():
        #     accuracies = [log[epoch]["users"][user_id]["test_accuracy"] for epoch in epochs]
        #     kd_losses = [user[epoch]["train_kd_loss"] for epoch in epochs]
        #     ce_losses = [user[epoch]["train_ce_loss"] for epoch in epochs]
        #     train_accuracies = [user[epoch]["train_accuracy"] for epoch in epochs]
        #     latencies = [user[epoch]["latency"] for epoch in epochs]
        #     data_imbalances = [user[epoch]["data_imbalance"] for epoch in epochs]
        #     energy_usages = [user[epoch]["energy_usage"] for epoch in epochs]
        #     memory_usages = [user[epoch]["memory_usage"] for epoch in epochs]

        #     plt.title(f"User {user_id} Test Accuracy")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Accuracy")
        #     plt.plot(epochs, accuracies)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_test_accuracy.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} KD Loss")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Loss")
        #     plt.plot(epochs, kd_losses)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_kd_loss.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} CE Loss")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Loss")
        #     plt.plot(epochs, ce_losses)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_ce_loss.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Train Accuracy")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Accuracy")
        #     plt.plot(epochs, train_accuracies)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_train_accuracy.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Latency")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Latency")
        #     plt.plot(epochs, latencies)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_latency.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Data Imbalance")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Data Imbalance")
        #     plt.plot(epochs, data_imbalances)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_data_imbalance.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Energy Usage")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Energy Usage")
        #     plt.plot(epochs, energy_usages)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_energy_usage.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Memory Usage")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Memory Usage")
        #     plt.plot(epochs, memory_usages)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_memory_usage.png"))
        #     plt.close()

        #     # Plot optimization
        #     objs = user["optimization"]["objective"]
        #     latencies = user["optimization"]["latency"]
        #     data_imbalances = user["optimization"]["data_imbalance"]

        #     plt.title(f"User {user_id} Objective Function")
        #     plt.xlabel("Step")
        #     plt.ylabel("Objective Function")
        #     plt.plot(objs)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_objective.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Latency")
        #     plt.xlabel("Step")
        #     plt.ylabel("Latency")
        #     plt.plot(latencies)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_latency.png"))
        #     plt.close()

        #     plt.title(f"User {user_id} Data Imbalance")
        #     plt.xlabel("Step")
        #     plt.ylabel("Data Imbalance")
        #     plt.plot(data_imbalances)
        #     plt.savefig(os.path.join("results", f"user_{user_id}_data_imbalance.png"))
        #     plt.close()

plot()