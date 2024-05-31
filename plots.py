import matplotlib.pyplot as plt

def plot_results(results_dir, num_users, latency_histories, data_imbalance_histories, obj_functions, losses, accuracies):
    # Plot latency and data imbalance history
    for user_idx in range(num_users):
        # Plot latency history
        plt.plot(latency_histories[user_idx])
        plt.title(f"Latency History for User {user_idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Latency")
        plt.savefig(results_dir + f"latency_history_user_{user_idx}.png")
        plt.close()
        # Plot data imbalance history
        plt.plot(data_imbalance_histories[user_idx])
        plt.title(f"Data Imbalance History for User {user_idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Imbalance")
        plt.savefig(results_dir + f"data_imbalance_history_user_{user_idx}.png")
        plt.close()

        plt.plot(obj_functions[user_idx])
        plt.title(f"Objective Function History for User {user_idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Objective Function")
        plt.savefig(results_dir + f"objective_function_user_{user_idx}.png")
        plt.close()

    # Plot loss history
    plt.plot(losses)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(results_dir + "loss_history.png")
    plt.close()
    # Plot accuracy history
    plt.plot(accuracies)
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(results_dir + "accuracy_history.png")
    plt.close()