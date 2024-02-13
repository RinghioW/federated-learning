import torchvision.models as models
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Server():
    def __init__(self, dataset) -> None:
        if dataset == "cifar10":
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Aggregate the updates from the users
    # In this case, averaging the weights will be sufficient
    def aggregate_updates(self, users):
        sum_weights = users[0].model.state_dict()
        for user in users[1:]:
            for key in sum_weights:
                sum_weights[key] += user.model.state_dict()[key]
        for key in sum_weights:
            sum_weights[key] /= len(users)
        self.model.load_state_dict(sum_weights)

    def evaluate(self, testset):
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
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