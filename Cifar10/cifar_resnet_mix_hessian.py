import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
# from mixup_generator import MixupGenerator
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

net = ResNet18().to(device)
#print(net)
criterion = nn.CrossEntropyLoss()
lr = 0.01
epochs = 200
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
train_record = []
test_record = []
# training_generator = MixupGenerator(x_train, y_train, batch_size=batch_size, alpha=0.2)()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
lambda_JR = 0.1 # hyperparameter
maxIter = 1
prob = 0.01

# Train
start_time = datetime.datetime.now()
correct = 0
total = 0
max_test = 0
for epoch in range(epochs):
    hessian_loss = 0.0
    jacobian_loss = 0.0
    running_loss = 0.0
    correct = 0
    total = 0
    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        # Zero the parameter gradients
        optimizer.zero_grad()
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        outputs = net(inputs)
        loss_super = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        # loss = criterion(outputs, labels)
        params = [outputs]
        for _, param in net.named_parameters():
            if param.requires_grad:
                # param= param.reshape(-1)
                p = np.random.binomial(1, prob)
                if p == 1:
                    params.append(param)
        # print(params)
        # params = torch.tensor(params)
        grads = torch.autograd.grad(loss_super, params, retain_graph=True, create_graph=True)
        jacobi_norm = 0
        hessian_tr = 0
        grad_list = []
        for i in grads: 
            # jacobi_norm += torch.norm(i)
            grad_list.append(i)

        # calculate hessian trace
        trace_vhv = []
        trace = 0.

        for iii in range(maxIter):
            with torch.no_grad():
                v = [torch.randint_like(p, high=1, device=device) for p in params]
                for v_i in v:
                   #  v_i[v_i == 0] = -1
                    v_i[v_i == 0] = np.random.binomial(1, prob * 2)
                for v_i in v:
                    v_i[v_i == 1] = 2 * np.random.binomial(1, prob * 2) - 1
            Hv = torch.autograd.grad(grad_list,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=True)
            # hessian_tr = torch.trace(hess)
            hessian_tr = group_product(Hv, v).cpu().item()
            #trace_vhv.append(hessian_tr)
            trace += hessian_tr
        # loss += lambda_JR * (trace / maxIter)
        loss = loss_super + lambda_JR * (jacobi_norm)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
        # print statistics
            running_loss += loss_super.item()
            jacobian_loss += jacobi_norm
            hessian_loss += hessian_tr

        if times % 10000 == 9999 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/10000))
    print('Accuracy of the network on the 10000 train inputs: %.3f %%' % (100 * correct / total))
    train_record.append(correct / total)
    with open('mixup_hessian.txt','a',encoding='utf-8') as f:
        f.write('[%d/%d] \n' % (epoch+1, epochs))
        f.write('Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * correct / total))
    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test inputs: %.3f %%' % (100 * correct / total))
    test_record.append(correct / total)
    if correct / total >  max_test:
        max_test = correct / total
    with open('mixup_hessian.txt','a',encoding='utf-8') as f:
        f.write('Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * correct / total))
    scheduler.step()
print('Finished Training, max test accuracy', 100 * max_test)
with open('mixup_hessian.txt','a',encoding='utf-8') as f:
    f.write('best Accuracy of the network on the 10000 train inputs: %.3f %% \n' % (100 * max_test))
end_time = datetime.datetime.now()
delta = end_time - start_time
delta_gmtime = time.gmtime(delta.total_seconds())
duration_str = time.strftime("%H:%M:%S", delta_gmtime)
print(duration_str)
plt.plot(train_record,label="train accuracy")
plt.plot(test_record,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig('resnet_mixup_hessian.png')
