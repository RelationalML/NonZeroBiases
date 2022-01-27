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


# Defines the forward and backward pass for retaining the top k scores, 
# the backward pass is from the straight through estimator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        #nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
        #fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        #bound = math.sqrt(6.0/fan)
        #nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        nn.init.constant_(self.scoresBias,0.5)
        nn.init.constant_(self.scoresWeights, 0.5)

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity)
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SupermaskConvTranspose(nn.ConvTranspose2d):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        #nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
        #fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        #bound = math.sqrt(6.0/fan)
        #nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))
        nn.init.constant_(self.scoresBias,0.5)
        nn.init.constant_(self.scoresWeights, 0.5)

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity)
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        

        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        return x


class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
    #def __init__(self, sparsity, zerobias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad=True)
        fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)
       
        #self.zero_bias = zerobias

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        
        subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity)
        
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias[:self.scoresBias.size(0)] * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        
        x = F.linear(x, w, b)
        return x 


##### Taken and modified from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html #####

class Encoder(nn.Module):

    def __init__(self,
                 sparsity : float,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            SupermaskConv(0.01, num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            SupermaskConv(sparsity, c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            SupermaskConv(sparsity, c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            SupermaskConv(sparsity, 2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            SupermaskConv(sparsity, 2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            SupermaskLinear(sparsity, 2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)



class Decoder(nn.Module):

    def __init__(self,
                 sparsity: float,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            SupermaskLinear(sparsity, latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            SupermaskConvTranspose(sparsity, 2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            SupermaskConv(sparsity, 2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            SupermaskConvTranspose(sparsity, 2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            SupermaskConv(sparsity, c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            SupermaskConvTranspose(sparsity, c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x



class Autoencoder(nn.Module):

    def __init__(self,
                 sparsity: float,
                 latent_dim: int,
                 base_channel_size: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.sparsity = sparsity
        # Creating encoder and decoder
        self.encoder = encoder_class(self.sparsity, num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(self.sparsity, num_input_channels, base_channel_size, latent_dim) 

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def clampScores(self, min=0):
        with torch.no_grad():
            l = [module for module in self.modules() if isinstance(module, (SupermaskConv, SupermaskLinear, SupermaskConvTranspose))]
            for layer in l:
                layer.scoresWeights.clamp_(min=min)
                layer.scoresBias.clamp_(min=min)


class AE2(nn.Module):

    def __init__(self, sparsity):
        super(AE2, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.sparsity = sparsity
        self.encoder = nn.Sequential(
            SupermaskConv(sparsity, in_channels = 3, out_channels = 12, kernel_size = 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            SupermaskConv(sparsity, in_channels = 12, out_channels = 24, kernel_size = 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			SupermaskConv(sparsity, in_channels = 24, out_channels = 48, kernel_size = 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
#             
			SupermaskConvTranspose(sparsity, in_channels = 48, out_channels = 24, kernel_size = 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			SupermaskConvTranspose(sparsity, in_channels = 24, out_channels = 12, kernel_size = 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            SupermaskConvTranspose(sparsity, in_channels = 12, out_channels = 3, kernel_size = 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def clampScores(self, min=0):
        with torch.no_grad():
            l = [module for module in self.modules() if isinstance(module, (SupermaskConv, SupermaskLinear, SupermaskConvTranspose))]
            for layer in l:
                layer.scoresWeights.clamp_(min=min)
                layer.scoresBias.clamp_(min=min)



class AeMnist(nn.Module):
    def __init__(self, sparsity):
        super(AeMnist, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.sparsity = sparsity
        self.encoder = nn.Sequential(
            SupermaskLinear(0.01, 784, 512),            
            nn.ReLU(),
            SupermaskLinear(sparsity, 512, 256),           
            nn.ReLU()
            )
			
        self.decoder = nn.Sequential(
			SupermaskLinear(sparsity, 256, 512),  # [batch, 24, 8, 8]
            nn.ReLU(),
			SupermaskLinear(sparsity, 512, 784),
            nn.Tanh()
            )

    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], -1))
        decoded = self.decoder(encoded)

        return decoded

    def clampScores(self, min=0):
        with torch.no_grad():
            l = [module for module in self.modules() if isinstance(module, (SupermaskConv, SupermaskLinear, SupermaskConvTranspose))]
            for layer in l:
                layer.scoresWeights.clamp_(min=min)
                layer.scoresBias.clamp_(min=min)