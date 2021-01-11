import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=data_transform['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=data_transform['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#实例化
net = Net().to(device)
writer = SummaryWriter()
init_img = torch.zeros((1, 3, 32, 32), device=device)
writer.add_graph(net,init_img)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
 for epoch in range(50):  # loop over the dataset multiple times
    print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim = 1)
        num_correct = (pred == labels).sum()
        running_acc += num_correct.item()
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(trainset)), running_acc / (len(trainset))))
    writer.add_scalar('train_loss',running_loss / (len(trainset)),epoch)
    writer.add_scalar('train_accuracy',running_acc / (len(trainset)),epoch)
    writer.add_scalar('learning_rate',optimizer.param_groups[0]["lr"], epoch)    

        # 模型评估
    net.eval()   
    eval_loss = 0
    eval_acc = 0
    # 测试模型
    for data in testloader:      
        img, label = data
        with torch.no_grad():
            img = Variable(img).to(device)
            label = Variable(label).to(device)

        out = net(img).to(device)

        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)

        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        testset)), eval_acc / (len(testset))))
    print()
    writer.add_scalar('test_loss',eval_loss / (len(testset)),epoch)
    writer.add_scalar('test_accuracy',eval_acc / (len(testset)),epoch)

# print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)