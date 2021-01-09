import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm

from VGG_16_model import VGG_16

#定义学习速率
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHES = 30

transform = transforms.Compose(
        [transforms.Resize([224,224]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 下载训练集 CIFAR-10 10分类训练集
train_dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#实例化
# device = "cpu"
model = VGG_16().to(device)
writer = SummaryWriter('runs/cifar')
init_img = torch.zeros((1, 3, 224, 224), device=device)
writer.add_graph(model,init_img)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

#训练模型
for epoch in range(EPOCHES):
    print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        img, label = data
        img = Variable(img).to(device)
        label = Variable(label).to(device)

        #向前传播
        out = model(img)

        #计算损失函数
        loss = criterion(out, label)
        running_loss = loss.item() * label.size(0)
        _, pred = torch.max(out, dim = 1)

        num_correct = (pred == label).sum()
        
        running_acc += num_correct.item()

        #手动清空梯度
        optimizer.zero_grad()

        #向后传播
        loss.backward()
        optimizer.step()

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
    writer.add_scalar('training loss',running_loss / (len(train_dataset)),epoch)
    writer.add_scalar('train_accuracy',running_acc / (len(train_dataset)),epoch)
    writer.add_scalar('learning_rate',optimizer.param_groups[0]["lr"], epoch)

    # 模型评估
    model.eval()   
    eval_loss = 0
    eval_acc = 0
    # 测试模型
    for data in test_loader:      
        img, label = data
        with torch.no_grad():
            img = Variable(img).to(device)
        with torch.no_grad():
            label = Variable(label).to(device)

        out = model(img)

        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)

        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()
    writer.add_scalar('test_loss',eval_loss / (len(test_dataset)),epoch)
    writer.add_scalar('train_accuracy',eval_acc / (len(test_dataset)),epoch)
    
# 保存模型
torch.save(model.state_dict(), './model.pkl')
