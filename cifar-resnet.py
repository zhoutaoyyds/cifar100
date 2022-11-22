# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:17:32 2022

@author: 10203
"""

import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter
myWriter = SummaryWriter('./tensorboard/log/')



myTransforms = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])


#  load
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          transform=myTransforms )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)



# 定义模型
myModel = torchvision.models.resnet50(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

# 损失函数及优化器
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)


learning_rate=0.001
myOptimzier = optim.SGD(myModel.parameters(), lr = learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(10):
    training_loss = 0.0
    print(_epoch)
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)   # GPU加速
        predict_label = myModel.forward(image)
       
        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step = _epoch*len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0 :
            print('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0
    #torch.save(myModel, 'Resnet50_Own.pkl') # 保存整个模型
    myModel.eval()
    for images,labels in test_loader:
        # GPU加速
        images = images.to(myDevice)
        labels = labels.to(myDevice)     
        outputs = myModel(images)   
        numbers,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
    myWriter.add_scalar('test_Accuracy',100 * correct / total)
