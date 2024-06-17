import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import optimum.quanto as quanto

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

        if low_rank:
            # TODO: Figure out what to do here
            pass
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
