import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import optimum.quanto as quanto

class Cifar10CNN(nn.Module):

    def __init__(self, quantize=False, pruning_factor=0., low_rank=False):
        super(Cifar10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = F.softmax(self.fc2(x))

        return x
    
class AdaptiveCifar10CNN(nn.Module):

    def __init__(self):
        super(AdaptiveCifar10CNN, self).__init__()


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def _quantize(self):
        quanto.quantize(self, weights=quanto.qint8)
    
    def _prune(self, pruning_factor):
        # TODO: Weights should be the same as when the network was first initialized on the device for pruning to be consistent
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(module=layer, name="weight", amount=pruning_factor, n=2, dim=0)
                prune.remove(layer, name="weight")
    
    def _low_rank(self, rank=10):
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self._svd(layer, rank)

    def _svd(self, layer, rank):
        with torch.no_grad():
            W = layer.weight
            U, S, V = torch.svd(W.flatten(1))  # SVD on 2D tensor

            # Reduce the number of singular values/vectors
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]

            # Reconstruct the low-rank weight matrix
            compressed_weight = torch.mm(U, torch.mm(torch.diag(S), V.T)).view_as(W)
            layer.weight = nn.Parameter(compressed_weight)

            if layer.bias is not None:
                # Bias compression is typically not done, but could be added if needed
                pass

        return layer
    
    def adapt(self, state_dict, pruning_factor=0., quantize=False, low_rank=False):
        self.load_state_dict(state_dict)

        if pruning_factor > 0.:
            self._prune(pruning_factor)
        elif quantize:
            self._quantize()
        elif low_rank:
            self._low_rank()



    def forward(self, x):

        # In case of low rank, need to multiply the U, S and Vh matrices

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = F.softmax(self.fc2(x))

        return x
    
    
