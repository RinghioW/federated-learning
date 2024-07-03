import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import optimum.quanto as quanto



class SVDConv2d(nn.Module):
    def __init__(self, conv_layer, truncation_rank=None):
        super(SVDConv2d, self).__init__()
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.bias = conv_layer.bias

        weight = conv_layer.weight.data
        weight_flat = weight.view(weight.size(0), -1)
        U, S, Vt = torch.linalg.svd(weight_flat, full_matrices=False)

        rank = truncation_rank or min(U.size(1), Vt.size(0))
        self.U = nn.Parameter(U[:, :rank])
        self.S = nn.Parameter(S[:rank])
        self.Vt = nn.Parameter(Vt[:rank, :])

    def forward(self, x):
        weight = torch.mm(self.U, torch.mm(torch.diag(self.S), self.Vt)).view(-1, *self.U.shape[0], *self.Vt.shape[1])
        return F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding)

class SVDLinear(nn.Module):
    def __init__(self, linear_layer, truncation_rank=None):
        super(SVDLinear, self).__init__()
        self.bias = linear_layer.bias

        weight = linear_layer.weight.data
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

        rank = truncation_rank or min(U.size(1), Vt.size(0))
        self.U = nn.Parameter(U[:, :rank])
        self.S = nn.Parameter(S[:rank])
        self.Vt = nn.Parameter(Vt[:rank, :])

    def forward(self, x):
        weight = torch.mm(self.U, torch.mm(torch.diag(self.S), self.Vt))
        return F.linear(x, weight, bias=self.bias)

    
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

    def _qat(self, dataset, q_epochs):
        # Quantize the model
        quanto.quantize(self, weights=quanto.qint8)

        # Train the model for q_epochs
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(q_epochs):
            for batch in trainloader:
                images, labels = batch["img"], batch["label"]
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Freeze the model
        quanto.freeze(self)

    def _ptq(self, calibration_data):
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        fused_self = torch.ao.quantization.fuse_modules(self, [['conv1', 'bn1', 'conv2', 'bn2', 'pool1'],
                                                                ['conv3', 'bn3', 'conv4', 'bn4', 'pool2'],
                                                                ['conv5', 'bn5', 'conv6', 'bn6', 'pool3'],
                                                                ['fc1', 'fc2']])
        torch.ao.quantization.prepare(fused_self, inplace=True)
        fused_self(calibration_data)
        torch.ao.quantization.convert(fused_self, inplace=True)
        self = fused_self
    def _prune(self, pruning_factor):
        # TODO: Weights should be the same as when the network was first initialized on the device for pruning to be consistent
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(module=layer, name="weight", amount=pruning_factor, n=2, dim=0)
                prune.remove(layer, name="weight")
    

    def _low_rank(self, truncation_rank=10):
        # Traverse all modules and replace Conv2d and Linear with SVD variants
        for name, module in self.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(self, name, SVDConv2d(module, truncation_rank))
            elif isinstance(module, nn.Linear):
                setattr(self, name, SVDLinear(module, truncation_rank))
    
    def adapt(self, state_dict=None, pruning_factor=0., quantize=False, low_rank=False):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        if pruning_factor > 0.:
            self._prune(pruning_factor)
        elif quantize:
            self._quantize()
        elif low_rank:
            self._low_rank()

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
    
    
