import matplotlib.pyplot as plt
import os
from datetime import datetime

results_dir = f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(results_dir)
ctr = 0

def plot_optimization(obj_function_history, latency_history, data_imbalance_history, scaling_factor):
    global ctr

    plt.title(f"Optimization (scaling factor: {scaling_factor})")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function")
    plt.plot(obj_function_history, label="Objective Function")
    plt.plot(latency_history, label="System Latency")
    plt.plot(data_imbalance_history, label="Data Imbalance")
    plt.legend()
    plt.savefig(f"{results_dir}/opt_{ctr}.png")
    plt.close()
    ctr += 1

def plot_devices(users):
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for user in users:
        for device in user.devices:
            plt.plot(device.training_losses, label=f"D{device.config['id']}")
    plt.legend()
    plt.savefig(f"{results_dir}/devices_training_loss.png")
    plt.close()
    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        for device in user.devices:
            plt.plot(device.training_accuracies, label=f"D{device.config['id']}")
    plt.legend()
    plt.savefig(f"{results_dir}/devices_training_accuracy.png")
    plt.close()

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        for device in user.devices:
            plt.plot(device.validation_accuracies, label=f"D{device.config['id']}")
    plt.legend()
    plt.savefig(f"{results_dir}/devices_test_accuracy.png")
    plt.close()

    
def plot_users(users):
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for user in users:
        plt.plot(user.training_losses, label=f"U{user.id}")
    plt.legend()
    plt.savefig(f"{results_dir}/users_training_loss.png")
    plt.close()
    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        plt.plot(user.training_accuracies, label=f"U{user.id}")
    plt.legend()
    plt.savefig(f"{results_dir}/users_training_accuracy.png")
    plt.close()

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for user in users:
        plt.plot(user.test_accuracies, label=f"U{user.id}")
    plt.legend()
    plt.savefig(f"{results_dir}/users_test_accuracy.png")
    plt.close()

def plot_server(losses, accuracies):
    plt.title("Server Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.savefig(f"{results_dir}/server_loss.png")
    plt.close()
    
    plt.title("Server Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(accuracies)
    plt.savefig(f"{results_dir}/server_accuracy.png")
    plt.close()
