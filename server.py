import torch
from config import DEVICE

class Server():
    def __init__(self, model, users, testset) -> None:
        self.model = model
        self.users = users
        self.testset = testset
        self.log = []

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    def _aggregate_updates(self):
        # Load the first user model
        state_dicts = [torch.load(f"checkpoints/user_{user.id}.pth") for user in self.users]
        n_samples = [user.n_samples() for user in self.users]
        total_samples = sum(n_samples)
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum([state_dict[key] * n_samples[i] for i, state_dict in enumerate(state_dicts)]) / total_samples
        torch.save(avg_state_dict, "checkpoints/server.pth")

    # Evaluate the server model on the test set
    def test(self):
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, num_workers=3)
        net = self.model().to(DEVICE)
        net.load_state_dict(torch.load("checkpoints/server.pth"))
        net.eval()
        
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["fine_label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        print(f"SERVER Test Accuracy: {accuracy}")
        self.log.append(accuracy)

    def _poll_users(self, kd_epochs, on_device_epochs):
        for user in self.users:
            # User trains devices
            user.train(kd_epochs, on_device_epochs)

    def train(self):
        # Wait for users to send their model
        self._poll_users(kd_epochs=10, on_device_epochs=10)

        # Aggregate the updates from the users
        self._aggregate_updates()

    def flush(self, results_dir):
        with open(f"{results_dir}/server.log", "w") as f:
            for accuracy in self.log:
                f.write(f"{accuracy}\n")
        import matplotlib.pyplot as plt
        plt.plot(self.log)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Server Test Accuracy")
        plt.savefig(f"{results_dir}/server.svg")
        plt.close()