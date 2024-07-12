import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import optimum.quanto as quanto
import torchvision.transforms as transforms
import tensorly as tl
from tensorly.decomposition import tucker



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tl.set_backend("pytorch")
class DecomposedConv2d(nn.Module):
    def __init__(self, original_layer, rank):
        super(DecomposedConv2d, self).__init__()
        self.rank = rank
        
        # Decompose the original layer's weights
        self.core, self.factors = tucker(original_layer.weight.data, rank=self.rank)
        
        # Register the decomposed components as parameters
        self.core = nn.Parameter(self.core)
        self.factors = [nn.Parameter(factor) for factor in self.factors]
        for i, factor in enumerate(self.factors):
            self.register_parameter(f'factor_{i}', self.factors[i])

        # Store the original layer's bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data)
        else:
            self.bias = None
            
        # Store layer attributes
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups

    def forward(self, x):
        # Recompute the weight matrix
        weight = tl.tucker_to_tensor((self.core, self.factors))
        
        # Perform the convolution
        return nn.functional.conv2d(x, weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                

class SVDLinear(nn.Module):
    def __init__(self, linear_layer, truncation_rank=10):
        super(SVDLinear, self).__init__()
        self.bias = linear_layer.bias

        weight = linear_layer.weight.data
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

        rank = truncation_rank
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
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

        if torch.cuda.is_available():
            self.cuda()

    def _qat(self, dataset, q_epochs):
        # Quantize the model
        quanto.quantize(self, weights=quanto.qint8)

        # Train the model for q_epochs
        to_tensor = transforms.ToTensor()
        dataset = dataset.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.cuda()
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(q_epochs):
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        quanto.freeze(self)


    def _ptq(self, calibration_data):
        to_tensor = transforms.ToTensor()
        calibration_data = calibration_data.map(lambda img: {"img": to_tensor(img)}, input_columns="img").with_format("torch")

        data = torch.utils.data.DataLoader(calibration_data, batch_size=32, num_workers=3)
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        # torch.ao.quantization.fuse_modules(self, [['bn1', 'relu1'], ['bn2', 'relu2'], ['bn3', 'relu3'], ['bn4', 'relu4'], ['bn5', 'relu5'], ['bn6', 'relu6'],])
        torch.ao.quantization.prepare(self, inplace=True)
        for batch in data:
            self(batch["img"].to(DEVICE))
        torch.ao.quantization.convert(self, inplace=True)


    def _prune(self, pruning_factor):
        # TODO: Weights should be the same as when the network was first initialized on the device for pruning to be consistent
        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(module=layer, name="weight", amount=pruning_factor, n=2, dim=0)
                prune.remove(layer, name="weight")
    
    def _low_rank(self, truncation_rank=10):
        # TODO: Traverse all modules and replace Conv2d and Linear with SVD variants
        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                setattr(self, name, SVDLinear(module, truncation_rank))
            elif isinstance(module, nn.Conv2d):
                setattr(self, name, DecomposedConv2d(module, truncation_rank))

    
    def adapt(self, state_dict=None, pruning_factor=0., quantize=False, low_rank=False, calibration_data=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        if pruning_factor > 0.:
            self._prune(pruning_factor)
        elif quantize:
            # Option tuo use qat or ptq. 
            # The difference is that qat is more accurate but slower (requires training the model for a few epochs)
            # While ptq is faster but less accurate (only requires running the model on calibration data)
            # self._ptq(calibration_data)
            self._qat(calibration_data)
        elif low_rank:
            self._low_rank()

    def forward(self, x):

        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.bn4(self.relu4(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(self.relu5(self.conv5(x)))
        x = self.bn6(self.relu6(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.softmax(self.fc2(x))

        return x
    
    
