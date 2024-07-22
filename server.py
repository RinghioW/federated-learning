import torch
from copy import deepcopy
from config import DEVICE
class Server():
    def __init__(self, model, users, testset) -> None:
        self.model = model
        self.users = users
        self.testset = testset

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    def _aggregate_updates(self):
        # Load the first user model
        state_dicts = [user.model.state_dict() for user in self.users]
        n_samples = [user.n_samples() for user in self.users]
        total_samples = sum(n_samples)
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum([state_dict[key] * n_samples[i] for i, state_dict in enumerate(state_dicts)]) / total_samples
        # Update model
        self.model.load_state_dict(avg_state_dict)

    # Evaluate the server model on the test set
    def test(self):
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, num_workers=3)
        net = self.model.to(DEVICE)
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
    
    # Select users for the next round of training
    def _select_users(self):
        for user in self.users:
            user.model = deepcopy(self.model)

    def _poll_users(self, kd_epochs, on_device_epochs):
        for user in self.users:
            # User trains devices
            user.train(kd_epochs, on_device_epochs)

    def train(self):
        # Choose the users for the next round of training
        self._select_users()

        # Wait for users to send their model
        self._poll_users(kd_epochs=10, on_device_epochs=10)

        # Aggregate the updates from the users
        self._aggregate_updates()
