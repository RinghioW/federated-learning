import torch
from config import DEVICE
import torchvision
from adaptivenet import AdaptiveNet
from copy import deepcopy
class Server():
    def __init__(self, dataset):
        if dataset == "cifar10":
            self.model = AdaptiveNet()
            self.model_state_dict = None
        else:
            # TODO: Add more datasets
            # Femnist, Shakespeare
            raise ValueError(f"Invalid dataset. Please choose from valid datasets")
        self.users = None
        self.wall_clock_training_times = None
        self.scaling_factor = 1.0

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    # TODO: Use FedAvg instead of parameter averaging (aggregate the gradients instead of the weights)
    def aggregate_updates(self):
        sum_weights = self.users[0].model_state_dict
        for user in self.users[1:]:
            for key in sum_weights:
                sum_weights[key] += user.model_state_dict[key]
        for key in sum_weights:
            sum_weights[key] = type(sum_weights[key])(sum_weights[key]/len(self.users))
        self.model.load_state_dict(sum_weights)
        self.model_state_dict = deepcopy(sum_weights)
        return self

    # Evaluate the server model on the test set
    # TODO: Find a way to compare this evaluation to some other H-FL method
    def evaluate(self, testset):
        to_tensor = torchvision.transforms.ToTensor()
        testset = testset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=3)
        net = self.model
        if self.model_state_dict is not None:
            net.load_state_dict(self.model_state_dict)
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
        # Compute estimated performances of the users
        estimated_performances = [user.diff_capability * self.wall_clock_training_times[idx] for idx, user in enumerate(self.users)]

        # Compute average user performance
        average_user_performance = sum(estimated_performances) / len(estimated_performances)

        # Compute adaptive scaling factor for each user
        for idx, user in enumerate(self.users):
            user.adaptive_scaling_factor = (average_user_performance / estimated_performances[idx]) * self.scaling_factor
            self.users[idx] = user
        return self

    # Select users for the next round of training
    # TODO: Consider tier-based selection (TiFL) instead of random selection
    def select_users(self, users, split=1.):
        # self.users = random.choices(users, k=math.floor(split*len(users)))
        self.users = users
        self.wall_clock_training_times = [1.] * len(self.users)
        return self