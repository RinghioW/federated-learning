import torch
import torchvision
from statistics import fmean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Server():
    def __init__(self, dataset, model, users):
        self.model = model
        torch.save({'model_state_dict': model().state_dict()}, "checkpoints/server.pt")
        self.dataset = dataset
        self.users = users
        self.wall_clock_training_times = None
        self.scaling_factor = 0.5
        self.init = False

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    # Step 18 in the ShuffleFL algorithm
    def _aggregate_updates(self):
        # Load the first user model
        state_dicts = [torch.load(f"checkpoints/user_{i}.pt")['model_state_dict'] for i in range(len(self.users))]
        n_samples = [user.n_samples() for user in self.users]
        total_samples = sum(n_samples)
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum([state_dict[key] * n_samples[i] for i, state_dict in enumerate(state_dicts)]) / total_samples
        # Save the aggregated weights
        torch.save({'model_state_dict': avg_state_dict}, "checkpoints/server.pt")

    # Evaluate the server model on the test set
    def test(self, testset):
        to_tensor = torchvision.transforms.ToTensor()
        testset = testset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=3)
        net = self.model()
        checkpoint = torch.load("checkpoints/server.pt")
        net.load_state_dict(checkpoint['model_state_dict'])
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
        estimated_performances = [user.diff_capability() * training_time for user, training_time in zip(self.users, self.wall_clock_training_times)]

        # Compute average user performance
        avg_user_performance = fmean(estimated_performances)

        # Compute adaptive scaling factor for each user
        for user, performance in zip(self.users, estimated_performances):
            user.adaptive_scaling_factor = (avg_user_performance / performance) * self.scaling_factor

    # Select users for the next round of training
    def _select_users(self, split=1.):
        # TODO: Select users
        pass

        for user in self.users:
            user.model = self.model

        # self.users = random.choices(users, k=math.floor(split*len(users)))
        self.wall_clock_training_times = [1.] * len(self.users)

    def _poll_users(self, kd_epochs, on_device_epochs, adapt=True, shuffle=True):
        transferred_samples = []
        for user in self.users:
            # User trains devices
            # ShuffleFL step 11-15
            n_transferred_samples = user.train(kd_epochs, on_device_epochs)
            transferred_samples.append(n_transferred_samples)
            print(f"User validation {user.id} accuracy: {user.validate()}")
        return transferred_samples

    def train(self):
        # Choose the users for the next round of training
        self._select_users()

        # Send the adaptive scaling factor to the users
        if self.init:
            self._send_adaptive_scaling_factor()

        # Wait for users to send their model
        self._poll_users(kd_epochs=10, on_device_epochs=10)

        # Aggregate the updates from the users
        self._aggregate_updates()

        if not self.init:
            self.init = True
