import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet import ResNet18
from models.googlenet import GoogLeNet
from models.vgg import VGG


net = ResNet18()
name = ['vgg.pkl', 'resnet.pkl', 'google.pkl', 'vgg_poly.pkl', 'resnet_poly.pkl', 'google_poly.pkl', 'which_one.pkl']
if name[6] == 'vgg.pkl':
    # Best_acc: 92.58
    net = VGG('VGG11')
if name[6] == 'resnet.pkl':
    # Best_acc: 95.15
    net = ResNet18()
if name[6] == 'google.pkl':
    # Best_acc: 95.51
    net = GoogLeNet()
if name[6] == 'vgg_poly.pkl':
    # Best_acc: 92.27
    net = VGG('VGG11')
if name[6] == 'resnet_poly.pkl':
    # Best_acc: 95.46
    net = ResNet18()
if name[6] == 'google_poly.pkl':
    # Best_acc: 95.60
    net = GoogLeNet()


# obtain directory path
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]
path = path+'/weight/'

# Transform the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = net.to(device)
# load weight
pretrain_dict = torch.load(path+name[6])
model_dict = {}
state_dict = net.state_dict()
for k, v in pretrain_dict.items():
    if k in state_dict:
        model_dict[k] = v
state_dict.update(model_dict)
net.load_state_dict(state_dict)


def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.*correct/total
    print('Acc: %.3f%%' % (accuracy))


if __name__ == '__main__':
    test()
