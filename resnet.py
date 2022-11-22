# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:15:01 2022

@author: 86178
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:21:29 2022

@author: 86178
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet34
from torchvision.transforms.transforms import Resize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
#采用自带的Cifar100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
 
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
net=models.resnet34(pretrained=True)
##迁移学习
for param in net.parameters(): #固定参数
    print(param.names)
    param.requires_grad = False
 
fc_inputs = net.fc.in_features #获得fc特征层的输入
net.fc = nn.Sequential(         #重新定义特征层，根据需要可以添加自己想要的Linear层
    nn.Linear(fc_inputs, 100),  #多加几层都没关系
    nn.LogSoftmax(dim=1)
)
 
net = net.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
 
history_loss=[]
history_acc=[]
testval_loss=[]
testval_acc=[]
 
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
 
        print(batch_idx+1,'/', len(trainloader),'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #history_loss.append(loss.item())
    history_loss.append(train_loss/(batch_idx+1))
    history_acc.append(100.*correct/total)
    net.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            test_inputs, test_targets = inputs.to('cuda'), targets.to('cuda')
            test_outputs = net(torch.squeeze(test_inputs, 1))
            test_loss1 = criterion(test_outputs, test_targets)
            test_loss += test_loss1.item()
            _, predicted = test_outputs.max(1)
            test_total += test_targets.size(0)
            test_correct += predicted.eq(test_targets).sum().item()
            print(batch_idx,'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
    print("epoch",'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
    #testval_loss.append(test_loss1.item())
    testval_loss.append(test_loss/(batch_idx+1))
    testval_acc.append(100.*test_correct/test_total)
    
 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            test_inputs, test_targets = inputs.to('cuda'), targets.to('cuda')
            test_outputs = net(torch.squeeze(inputs, 1))
            test_loss1 = criterion(test_outputs, test_targets)
            test_loss += test_loss1.item()
            _, predicted = test_outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            print(batch_idx,'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
    print("epoch",'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))
    #testval_loss.append(test_loss.item())
    testval_loss.append(test_loss/(batch_idx+1))
    testval_acc.append(100.*test_correct/test_total)
 
for epoch in range(2): #设置成100循环
    train(epoch)
torch.save(net.state_dict(),'resnet34cifar100.pkl') #训练完成后保存模型，供下次继续训练使用
print(history_loss)
print(history_acc)
print(testval_loss)
print(testval_acc)
plt.plot(history_loss)
plt.plot(testval_loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['xunlian', 'test'], loc='upper left')
plt.show()
print('begin  test ')
plt.plot(history_acc)
plt.plot(testval_acc)
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['xunlian', 'test'], loc='upper left')
plt.show()