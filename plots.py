import matplotlib.pyplot as plt

def plot_optimization(dir, latency_histories, data_imbalance_histories, obj_function_histories, losses, accuracies):
    # Plot latency and data imbalance
    for idx, data_imbalance, latency, obj_function in enumerate(zip(data_imbalance_histories, latency_histories, obj_function_histories)):
        plt.plot(latency)
        plt.title(f"Latency for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.savefig(dir + f"latency_user_{idx}.png")
        plt.close()

        plt.plot(data_imbalance)
        plt.title(f"Data Imbalance for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Imbalance")
        plt.savefig(dir + f"data_imbalance_user_{idx}.png")
        plt.close()

        plt.plot(obj_function)
        plt.title(f"Objective Function for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Objective Function")
        plt.savefig(dir + f"objective_function_user_{idx}.png")
        plt.close()

def plot_devices(users):
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for user in users:
        for device in user.devices:
            plt.plot(device.training_losses, label=f"U{user.id} D{device.config['id']}")
    plt.legend()
    plt.savefig("results/devices_training_loss.png")
    plt.close()
    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        for device in user.devices:
            plt.plot(device.training_accuracies, label=f"U{user.id} D{device.config['id']}")
    plt.legend()
    plt.savefig("results/devices_training_accuracy.png")
    plt.close()

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        for device in user.devices:
            plt.plot(device.validation_accuracies, label=f"U{user.id} D{device.config['id']}")
    plt.legend()
    plt.savefig("results/devices_test_accuracy.png")
    plt.close()

    
def plot_users(users):
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for user in users:
        plt.plot(user.training_losses, label=f"U{user.id}")
    plt.legend()
    plt.savefig("results/users_training_loss.png")
    plt.close()
    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        plt.plot(user.training_accuracies, label=f"U{user.id}")
    plt.legend()
    plt.savefig("results/users_training_accuracy.png")
    plt.close()

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        plt.plot(user.test_accuracies, label=f"U{user.id}")
    plt.legend()
    plt.savefig("results/users_test_accuracy.png")
    plt.close()

def plot_server(losses, accuracies):
    plt.title("Server Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.savefig("results/server_loss.png")
    plt.close()
    
    plt.title("Server Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(accuracies)
    plt.savefig("results/server_accuracy.png")
    plt.close()


# def plot_server_with_users(server_losses, server_accuracies, users):
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     for user in users:
#         plt.plot(user.training_losses, label=f"U{user.id}")
#     plt.plot(server_losses, label="Server", linewidth=3)
#     plt.legend()
#     plt.savefig("results/users_training_loss.png")
#     plt.close()
    
#     plt.title("Training Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     for user in users:
#         plt.plot(user.training_accuracies, label=f"U{user.id}")
#     plt.plot(server_accuracies, label="Server", linewidth=3)
#     plt.legend()
#     plt.savefig("results/users_training_accuracy.png")
#     plt.close()

#     plt.title("Test Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     for user in users:
#         plt.plot(user.test_accuracies, label=f"U{user.id}")
#     plt.plot(server_accuracies, label="Server", linewidth=3)
#     plt.legend()
#     plt.savefig("results/users_test_accuracy.png")
#     plt.close()