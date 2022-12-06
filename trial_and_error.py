# pytorch에 익숙해지기 위해 이것저것 시도해본것
# tutorial을 그대로 따라해본 것도 있고 스스로 코딩해본 것도 있음
# colab으로 코딩한 것
import torch.nn as nn
import torch.nn.functional as F 
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class IdentityPadding(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super(IdentityPadding,self).__init__()
    self.pooling = nn.MaxPool2d(1, stride = stride)
    self.add_channels = out_channels-in_channels

  def forward(self, x):
    out = F.pad(x, (0,0,0,0,0, self.add_channels))
    out = self.pooling(out)
    return out




class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride = 1, down_sample = False):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride= stride, padding =1,bias = False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.stride = stride

    if  down_sample:
      self.down_sample = IdentityPadding(in_channels, out_channels, stride)
    else:
      self.down_sample = None


  def forward(self, x):
    shortcut = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.down_sample is not None:
      shortcut  = self.down_sample(x)
    
    out = out + shortcut
    out = self.relu(out)
    return out



class ResNet(nn.Module):
  def __init__(self, num_layers ,block, num_classes = 10):
    super(ResNet, self).__init__()
    self.num_layers= num_layers
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace = True)

    self.layers_2n = self.get_layers(block, 16,16, stride = 1)
    self.layers_4n = self.get_layers(block, 16,32, stride = 2)
    self.layers_6n = self.get_layers(block, 32,64, stride = 2)

    self.avg_pool = nn.AvgPool2d(8, stride = 1)
    self.fc_out = nn.Linear(64, num_classes)

    for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    
  def get_layers(self, block, in_channels, out_channels, stride):
    if stride == 2:
      down_sample = True
    else:
      down_sample = False
    layers_list = nn.ModuleList(
      [block(in_channels, out_channels, stride, down_sample)])
      
    for _ in range(self.num_layers - 1):
      layers_list.append(block(out_channels, out_channels))

    return nn.Sequential(*layers_list)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layers_2n(x)
    x = self.layers_4n(x)
    x = self.layers_6n(x)

    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc_out(x)
    return x

    
def resnet():
  block = ResidualBlock
  # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
  model = ResNet(5, block) 
  return model


import torch.optim as optim
net = resnet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4)


for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i%2000 == 1999:
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
      running_loss = 0.0

print("finish")


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
 