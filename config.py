import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
STD_CORRECTION = 10 ** 9

class Style():
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  RESET = '\033[0m'