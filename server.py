import torchvision.models as models
import torch
from config import DEVICE

class Server():
    def __init__(self, users, dataset) -> None:
        if dataset == "cifar10":
            self.model = models.mobilenet_v3_large()
        else:
            raise ValueError(f"Invalid dataset. Please choose from valid datasets")
        self.users = users
        self.wall_clock_training_times = {user: 1. for user in users}
        self.staleness_factors = [1.0 for _ in users]
        self.estimated_performances = [1.0 for _ in users]
        self.user_capabilities = [1.0 for _ in users]
        self.average_user_latency = 1.0
        self.scaling_factor = 1.0

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    def aggregate_updates(self, users):
        sum_weights = users[0].model.state_dict()
        for user in users[1:]:
            for key in sum_weights:
                sum_weights[key] += user.model.state_dict()[key]
        for key in sum_weights:
            sum_weights[key] = type(sum_weights[key])(sum_weights[key]/len(users))
        self.model.load_state_dict(sum_weights)

    # Evaluate the server model on the test set
    def evaluate(self, testset):
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=2)
        net = self.model
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy
    
    # Equation 10 in ShuffleFL
    def send_adaptive_scaling_factor(self):
        # Compute average user latency
        average_user_latency = sum(self.estimated_performances) / len(self.estimated_performances)
        for idx, user in enumerate(self.users):
            adaptive_coefficient = (average_user_latency / self.estimated_performances[idx]) * self.scaling_factor
            user.adaptive_coefficient = adaptive_coefficient
    
    def estimated_performance_users(self):
        for idx, user in enumerate(self.users):
            self.estimated_performances[idx] = user.average_capability * self.wall_clock_training_times[idx]
        return self.estimated_performances