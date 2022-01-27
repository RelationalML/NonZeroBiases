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

#import ticket_models
import tabular_dataloader
import ticket_training
from fc import tabular

from initializers import *

import numpy as np

import sys

args = None

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch planted ticket edge-popup implementation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--warmup_length', type=int, default=5)

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')

    parser.add_argument('--anneal', type=bool, default=False,
                        help='whether sparsity should slowly be annealed towards target sparsity')
    
    parser.add_argument('--dataset', type=str, default='circle',
                        help='path to dataset')
    
    parser.add_argument('--din', type=int, default=1,
                        help='number of features (first columns in dataset)')
    
    parser.add_argument('--dout', type=int, default=1,
                        help='number of features (first columns in dataset)')
 
    parser.add_argument('--init', type=str, default='kn-nonzero-bias', 
                        help='initialization approach')
    
    parser.add_argument('--depth', type=int, default=5, metavar='depth',
                        help='depth of fully-connected architecture')
    
    parser.add_argument('--width', type=int, default=100, metavar='width',
                        help='width of fully-connected architecture')
    
    parser.add_argument('--bfac', type=float, default=0.05, 
                        help='downscaling of bias initialization')

    parser.add_argument('--anneal-epochs', type=int, default=1, 
                        help='downscaling of bias initialization')
    
    parser.add_argument('--scaling', action="store_true", default=False, help="scale parameters in every epoch")
    
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    
    parser.add_argument(
        "--zerobias", action="store_true", default=False, help="do not score biases"
    )
    
    parser.add_argument('--task', default='class', choices=['class', 'reg'], help='classification (class) or regression (reg) task')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cpu"
    args.workers = 1

    
    train_loader, _ = tabular_dataloader.dataloader(args.dataset, args.din, args.batch_size, True, args.workers, args.seed)
    test_loader, dataTest = tabular_dataloader.dataloader(args.dataset, args.din, args.test_batch_size, False, args.workers, args.seed)

    ## Model
    #model = tabular(args.sparsity, args.zerobias, dataTest.d_in, args.dout, args.depth, args.width, args.task).to(device)
    model = tabular(args.sparsity, dataTest.d_in, args.dout, args.depth, args.width, args.task).to(device)
    
    ##initialize
    if args.init == "kn-nonzero-bias":
        init_with_bias(args, model)
    if args.init == "ortho-nonzero-bias":
        init_with_bias_ortho(args, model)
    if args.init == "kn-zero-bias":
        init_zero_bias(args, model)
    if args.init == "ortho-bias-special":
        init_ortho_with_dep_bias(args, model)
    if args.init == "ortho-zero-bias":
        init_ortho_with_zero_bias(args, model)
    if args.init == "univ":
        init_univ(args, model)
    
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    if (args.task == 'reg'):
        loss = nn.MSELoss()
    else:
        loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        )

    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = cosine_lr(optimizer, args)

    if args.sparsity == -1:
        args.sparsity = model.get_sparsity_gt()

    if (args.task == 'reg'):
        for anneal_epoch in range(1, args.anneal_epochs + 1):
            # anneal sparsity
            if (args.anneal):
                sparsity = args.sparsity**(anneal_epoch / args.anneal_epochs)
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l:
                    if isinstance(layer, nn.Linear):
                        layer.sparsity = sparsity
                print("Sparsity: ", sparsity)
            
            for epoch in range(1, args.epochs + 1):
                ticket_training.trainReg(model, scheduler, device, train_loader, optimizer, loss, epoch, args.log_interval, args.scaling)
                ticket_training.testReg(model, device, loss, test_loader)
                # scheduler.step()
    else:
        for anneal_epoch in range(1, args.anneal_epochs + 1):
            # anneal sparsity
            if (args.anneal):
                sparsity = args.sparsity**(anneal_epoch / args.anneal_epochs)
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l:
                    if isinstance(layer, nn.Linear): #tabular.SupermaskLinear):
                        layer.sparsity = sparsity
                print("Sparsity: ", sparsity)
            
            for epoch in range(1, args.epochs + 1):
                ticket_training.trainClass(model, scheduler, device, train_loader, optimizer, loss, epoch, args.log_interval, args.scaling)
                ticket_training.testClass(model, device, loss, test_loader)
                # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "ticket.pt")
        
    #if args.plot_model:
        


if __name__ == '__main__':
    main()
