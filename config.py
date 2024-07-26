import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "cifar100"
LABEL_NAME = "fine_label"
NUM_CLASSES = 100

# https://developer.nvidia.com/embedded/jetson-modules
# https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
JETSON_NANO_CONFIG = {
    'name' : 'Jetson Nano',
    'compute' : 472, # GFLOPS
    'memory': 4, # GB
}

JETSON_XAVIER_NX_8GB_CONFIG = {
    'name' : 'Jetson Xavier NX 8GB',
    'compute' : 21, # TOPS
    'memory': 8, # GB
}

RASPBERRY_PI_4_CONFIG = {
    'name' : 'Raspberry Pi 4 Model B',
    'compute' : 9.69, # GFLOPS
    'memory': 4, # GB
}

def train_cifar10(model, cifar10):
    dataloader = torch.utils.data.DataLoader(cifar10, batch_size=32, shuffle=True, num_workers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(15):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()