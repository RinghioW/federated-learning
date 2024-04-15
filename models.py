import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision

# Simple CNN that supports quantization (PTQ), pruning and low-rank adaption
# Built for CIFAR-10 dataset (3x32x32)
class AdaptiveNet(nn.Module):
    def __init__(self, quantization_factor=0., pruning_factor=0., low_rank_factor=0.):
        super(AdaptiveNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        prune.ln_structured(module=self.conv2, name="weight", amount=pruning_factor, n=2, dim=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        prune.ln_structured(module=self.fc1, name="weight", amount=pruning_factor, n=2, dim=0)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x





    