from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import operator
from functools import reduce
import numpy as np
import torch.nn.functional as F

from resnet_new import SupermaskLinear, SupermaskConv

class tabular(nn.Module):
    def __init__(self, sparsity, input_shape, output_shape, depth, width, task):
    #def __init__(self, sparsity, zerobias, input_shape, output_shape, depth, width, task):
        super(tabular, self).__init__()
        self.sparsity = sparsity
        #self.zerobias = zerobias
        self.arch = self.make_architecture(input_shape, output_shape, depth, width)
        self.net = self.make_layers(task)
        #self.task = task
   
    def make_architecture(self, input_shape, output_shape, depth, width):
       arch = np.ones(depth, dtype=int)*width
       arch[0] = input_shape
       arch[-1] = output_shape
       return arch
   
    def make_layers(self, task):
        layerStack = []
        dd = len(self.arch)
        for i in np.arange(dd-2):
            l = SupermaskLinear(self.sparsity, self.arch[i], self.arch[i+1])
            #l = SupermaskLinear(self.sparsity, self.zerobias, self.arch[i], self.arch[i+1])
            layerStack += [l, nn.ReLU()]
        #last layer
        i =  dd-2
        l = SupermaskLinear(self.sparsity, self.arch[i], self.arch[i+1])
        #l = SupermaskLinear(self.sparsity, self.zerobias, self.arch[i], self.arch[i+1])
        layerStack += [l]
        #if task == 'class':
        #    layerStack += [nn.Softmax()]
        return nn.Sequential(*layerStack)
    
    def forward(self, x):
        #y = F.linear(self.net(x))
        #print(y.shape)
        return self.net(x) #F.linear(self.net(x))
