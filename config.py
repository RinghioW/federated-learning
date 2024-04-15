import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
STD_CORRECTION = 10 ** 6