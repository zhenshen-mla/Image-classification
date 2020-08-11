import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from models.googlenet import GoogLeNet

# Visualization
writer = SummaryWriter(comment='_google')
# obtain directory path
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]
if not os.path.exists(path+'/weight'):
    os.mkdir(path+'/weight')

'''
Step 1: Data pre-processing
'''

print('==> Preparing data..')
# Transform the training set
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Transform the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Dataset download and packaging to Dataloader
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
# Label
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


'''
Step 2: Allocate GPU
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
Step 3: Building model, loss function, optimizer and scheduler etc.
'''

print('==> Building model..')
net = GoogLeNet()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
# Record the best test accuracy(Global var)
best_acc = 0


'''
Step 4: Model Training 
'''

def train(epoch):

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    losses = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        losses = train_loss / (batch_idx + 1)

        if batch_idx % 20 == 0:
            print('Training | Epoch: %d | Best_acc: %.2f | Batches: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.5f'
                  % (epoch, best_acc, batch_idx, len(trainloader), losses, accuracy, correct, total, optimizer.param_groups[0]['lr']))

    scheduler.step()
    writer.add_scalar('scalar/loss_train', losses, epoch)
    writer.add_scalar('scalar/accuracy_train', accuracy, epoch)


'''
Step 5: Model Testing
'''

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    losses = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            losses = test_loss / (batch_idx + 1)

            if batch_idx % 20 == 0:
                print(
                    'Testing | Epoch: %d | Best_acc: %.2f | Batches: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d) '
                    % (epoch, best_acc, batch_idx, len(testloader), losses, accuracy, correct, total))

    writer.add_scalar('scalar/loss_test', losses, epoch)
    writer.add_scalar('scalar/accuracy_test', accuracy, epoch)

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), path+'/weight/google.pkl')


if __name__ == '__main__':
    for epoch in range(350):
        train(epoch)
        test(epoch)
