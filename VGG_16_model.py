import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

#定义VGG-16网络模型
class VGG_16(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGG_16, self).__init__()
        self.features = nn.Sequential(
            #第一层 卷积层
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            #第二层 卷积后最大池化层
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True), 

            #第三层 卷积层
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            #第四层 卷积最大池化层
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #第五层 卷积层
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #第六层 卷积层
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #第七层 卷积&最大池化层
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #第八层 卷积层
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #第九层 卷积层
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #第十层 卷积&最大池化层
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #第十一层 卷积层
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #第十二层 卷积层
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #第十三层 卷积&最大池化&平均池化层
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )

        self.classifier = nn.Sequential(
            #第十四层 
            nn.Linear(2048,4096),
            nn.ReLU(True),
            nn.Dropout(),

            #第十五层
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            #第十六层
            nn.Linear(4096,num_classes),
            
        )

    def forward(self, x):
        out = self.features(x) 
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out