import torch
from config import DEVICE
import torchvision
from statistics import fmean
class Server():
    def __init__(self, dataset, model, users):
        self.model = model
        self.dataset = dataset
        self.users = users
        self.wall_clock_training_times = None
        self.scaling_factor = 10 ** 5
        self.init = False

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    def _aggregate_updates(self):
        sum_weights = torch.load(f"checkpoints/user_0/user.pt")['model_state_dict']
        for user in self.users[1:]:
            state_dict = torch.load(f"checkpoints/user_{user.id}/user.pt")['model_state_dict']
            for key, val in state_dict.items():
                sum_weights[key] += val
        for key in sum_weights:
            sum_weights[key] = type(sum_weights[key])(sum_weights[key] * (1/len(self.users)))
        
        # Save the aggregated weights
        torch.save({'model_state_dict': sum_weights}, "checkpoints/server/server.pt")

    # Evaluate the server model on the test set
    # TODO: Find a way to compare this evaluation to some other H-FL method
    def test(self, testset):
        to_tensor = torchvision.transforms.ToTensor()
        testset = testset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=3)
        net = self.model()
        if self.init:
            checkpoint = torch.load("checkpoints/server/server.pt")
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.init = True
        net.eval()
        
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
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
    def _send_adaptive_scaling_factor(self):
        # Compute estimated performances of the users
        estimated_performances = [user.diff_capability * training_time for user, training_time in zip(self.users, self.wall_clock_training_times)]

        # Compute average user performance
        avg_user_performance = fmean(estimated_performances)

        # Compute adaptive scaling factor for each user
        for user, performance in zip(self.users, estimated_performances):
            user.adaptive_scaling_factor = (avg_user_performance / performance) * self.scaling_factor

    # Select users for the next round of training
    # TODO: Consider tier-based selection (TiFL) instead of random selection
    def _select_users(self, split=1.):
        # TODO: Select users
        pass

        for user in self.users:
            user.model = self.model

        # self.users = random.choices(users, k=math.floor(split*len(users)))
        self.wall_clock_training_times = [1.] * len(self.users)

    def _poll_users(self, adapt=True, shuffle=True, on_device_epochs=10):
        for user in self.users:

            # User trains devices
            # ShuffleFL step 11-15
            user.train(epochs=on_device_epochs, verbose=True)
    
    def train(self):
        # Choose the users for the next round of training
        self._select_users()

        # Send the adaptive scaling factor to the users
        self._send_adaptive_scaling_factor()

        # Wait for users to send their model
        self._poll_users()

        # Aggregate the updates from the users
        self._aggregate_updates()