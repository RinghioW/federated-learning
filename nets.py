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


# Simple CNN that supports quantization (PTQ), pruning and low-rank adaption
# Built for CIFAR-10 dataset (3x32x32)
class AdaptiveNet(nn.Module):
    def __init__(self, quantize=False, pruning_factor=0., low_rank=False):
        super(AdaptiveNet, self).__init__()
        self.quantize = quantize
        self.low_rank = low_rank
        self.pruning_factor = pruning_factor
        if quantize and pruning_factor > 0:
            raise ValueError("Cannot quantize a pruned network.")

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batchnorm = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if pruning_factor > 0.:
            self.conv2 = prune.ln_structured(module=self.conv2, name="weight", amount=pruning_factor, n=2, dim=0)
            self.conv2 = prune.remove(self.conv2, name="weight")
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        if pruning_factor > 0.:
            self.fc1 = prune.ln_structured(module=self.fc1, name="weight", amount=pruning_factor, n=2, dim=0)
            self.fc1 = prune.remove(self.fc1, name="weight")
        self.fc2 = nn.Linear(120, 84)
        self.batchnorm2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        
        if low_rank:
            # Get the svd of the weight matrix
            # And truncate
            conv1_U, conv1_S, conv1_Vh = torch.linalg.svd(self.conv1.weight, full_matrices=False)
            conv1_U = conv1_U[:, :6]
            conv1_S = conv1_S[:6]
            conv1_Vh = conv1_Vh[:6, :]
            conv2_U, conv2_S, conv2_Vh = torch.linalg.svd(self.conv2.weight, full_matrices=False)
            conv2_U = conv2_U[:, :16]
            conv2_S = conv2_S[:16]
            conv2_Vh = conv2_Vh[:16, :]
            fc1_U, fc1_S, fc1_Vh = torch.linalg.svd(self.fc1.weight, full_matrices=False)
            fc1_U = fc1_U[:, :120]
            fc1_S = fc1_S[:120]
            fc1_Vh = fc1_Vh[:120, :]
            fc2_U, fc2_S, fc2_Vh = torch.linalg.svd(self.fc2.weight, full_matrices=False)
            fc2_U = fc2_U[:, :84]
            fc2_S = fc2_S[:84]
            fc2_Vh = fc2_Vh[:84, :]
            fc3_U, fc3_S, fc3_Vh = torch.linalg.svd(self.fc3.weight, full_matrices=False)
            fc3_U = fc3_U[:, :10]
            fc3_S = fc3_S[:10]
            fc3_Vh = fc3_Vh[:10, :]
            

        
        if quantize:
            quanto.quantize(self, weights=quanto.qint8)
    
    
    def forward(self, x):
        x = self.pool(torch.relu(self.batchnorm(self.conv1(x))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.batchnorm2(self.fc2(x)))
        x = self.fc3(x)
        return x
