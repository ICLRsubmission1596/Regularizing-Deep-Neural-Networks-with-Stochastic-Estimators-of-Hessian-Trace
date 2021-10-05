import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import math

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

net=Wide_ResNet(28, 10, 0, 100).to(device)
#print(net)
criterion = nn.CrossEntropyLoss()
lr = 0.1
epochs = 200
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
train_record = []
test_record = []
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# Train
start_time = datetime.datetime.now()
max_test_1 = 0
max_test_5 = 0
correct = 0
total = 0
correct_1 = 0
correct_5 = 0
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4, nesterov=True)
    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if times % 10000 == 9999 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/10000))
    print('Accuracy of the network on the 10000 train inputs: %.3f %%' % (100 * correct / total))
    train_record.append(correct / total)
    with open('wideSGD.txt','a',encoding='utf-8') as f:
        f.write('[%d/%d] \n' % (epoch+1, epochs))
        f.write('Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * correct / total))
    # Test
    correct = 0
    total = 0
    correct_1 = 0
    correct_5 = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('top 1 Accuracy of the network on the 10000 test inputs: %.3f %%' % (100 * correct_1 / total))
    print('top 5 Accuracy of the network on the 10000 test inputs: %.3f %%' % (100 * correct_5 / total))
    test_record.append(correct / total)
    if correct_1 / total >  max_test_1:
        max_test_1 = correct_1 / total
    if correct_5 / total >  max_test_5:
        max_test_5 = correct_5 / total
    with open('wideSGD.txt','a',encoding='utf-8') as f:
        f.write('Top 1 Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * correct_1 / total))
        f.write('Top 5 Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * correct_5 / total))
    scheduler.step()
print('Finished Training, max test top accuracy', 100 * max_test_1, 100 * max_test_5)
with open('wideSGD.txt','a',encoding='utf-8') as f:
    f.write('best top 1 Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * max_test_1))
    f.write('best top 5 Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * max_test_5))
end_time = datetime.datetime.now()
delta = end_time - start_time
delta_gmtime = time.gmtime(delta.total_seconds())
duration_str = time.strftime("%H:%M:%S", delta_gmtime)
print(duration_str)
# plt.plot(train_record,label="train accuracy")
# plt.plot(test_record,label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.savefig('wideSGD.png')
