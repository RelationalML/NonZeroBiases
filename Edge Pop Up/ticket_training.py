from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

def rescaling(model, data, target, criterion, args):
    with torch.no_grad():
        output = model(data)
    #output.requires_grad = False
    #scale = nn.Parameter(torch.Tensor(1))
    #scale.data = torch.ones(1)
    scale = torch.ones(1, device=output.device, requires_grad=True)
    #ll = criterion(scale*output, target)
    opt = optim.SGD([scale], lr=0.1)
    for i in range(20):
        ll = criterion(scale*output, target)
        opt.zero_grad()
        ll.backward()
        opt.step()
    depth = 0
    print("scale: ", scale)
    #print(scale)
    for m in model.modules():
        if isinstance(m,(nn.Linear, nn.Conv2d)):
            depth = depth+1
    scale_per_layer = (scale)**(1/depth)
    scale_bias = scale_per_layer
    for m in model.modules():
        if isinstance(m,(nn.Linear, nn.Conv2d)):
            m.weight.data = scale_per_layer*m.weight.data
            m.weight.requires_grad = False
            m.bias.data = scale_bias*m.bias.data
            m.bias.requires_grad = False
            scale_bias = scale_bias*scale_per_layer
    args.lr = args.lr/scale
    return model

def trainClass(model, scheduler, device, train_loader, optimizer, criterion, epoch, log_interval, scaling, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.squeeze(target).to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler(epoch, batch_idx)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_loss = loss.item()
    if scaling:
        model = rescaling(model, data, target, criterion)
        
    return train_loss



def testClass(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.squeeze(target).to(device=device, dtype=torch.long)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))

        
def trainReg(model, scheduler, device, train_loader, optimizer, criterion, epoch, log_interval, scaling, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        #print(target.size())
        output = model(data)
        #print(output.size())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler(epoch, batch_idx)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_loss = loss.item()
    if scaling:
        model = rescaling(model, data, target, criterion)
        
    return train_loss


def testReg(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    #correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device=device, dtype=torch.float)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
    
    return test_loss
    
