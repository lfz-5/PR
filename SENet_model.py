import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

class SEModule(nn.Module):

    def __init__(self, channels, reduction=3):
        super(SEModule, self).__init__()
        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1,padding=0),
            nn.Sigmoid(),
        )
    def forward(self, x):
        module_input = x
        x=self.SE(x)
        return module_input * x

