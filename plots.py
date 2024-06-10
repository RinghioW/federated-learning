import matplotlib.pyplot as plt

def plot_optimization(results_dir, latency_histories, data_imbalance_histories, obj_function_histories, losses, accuracies):
    # Plot latency and data imbalance
    for idx, data_imbalance, latency, obj_function in enumerate(zip(data_imbalance_histories, latency_histories, obj_function_histories)):
        plt.plot(latency)
        plt.title(f"Latency for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.savefig(results_dir + f"latency_user_{idx}.png")
        plt.close()

        plt.plot(data_imbalance)
        plt.title(f"Data Imbalance for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Imbalance")
        plt.savefig(results_dir + f"data_imbalance_user_{idx}.png")
        plt.close()

        plt.plot(obj_function)
        plt.title(f"Objective Function for User {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Objective Function")
        plt.savefig(results_dir + f"objective_function_user_{idx}.png")
        plt.close()

def plot_training(results_dir, losses, accuracies):
    # Plot loss
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(results_dir + "loss.png")
    plt.close()

    # Plot accuracy
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(results_dir + "accuracy.png")
    plt.close()