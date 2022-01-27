import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import math

#python cifar_main.py --initBias kn-nonzero-bias --sparsity 0.1 --anneal True --epochs 50 --lr 0.1 --levels 5

class TicketDataset(Dataset):

    def __init__(self, data, d_in, train_proportion, is_training):

        if is_training:
            ran = np.arange(0,math.ceil(train_proportion*data.shape[0]))
        else:
            ran = np.arange(math.ceil(train_proportion*data.shape[0]),data.shape[0])
        m = data.shape[1] - d_in
        self.data_feats = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in)[np.newaxis,:]])
        self.data_resp = torch.Tensor(data[ran[:,np.newaxis],np.arange(d_in,d_in + m)[np.newaxis,:]])
        self.d_in = d_in
        self.d_out = m
        #print(self.data_resp.shape)

    def __len__(self):
        return self.data_feats.shape[0]


    def __getitem__(self, index):
        return self.data_feats[index,:], self.data_resp[index]


def load_file(file_name, d_in, seed=42, is_train = True, split = .8):
    #d_in: number of features, remaining (last) columns correspond to target
    data = np.loadtxt(file_name, delimiter=',', skiprows=0)  #draw_data_helix(n, 1, noise)
    n = data.shape[0]
    ## shuffle with fixed seed for same effect across datasets of synflow
    perm = np.random.RandomState(seed=seed).permutation(n)
    data = data[perm,]
    
    dataset = TicketDataset(data, d_in, split, is_train)
    return dataset


def dataloader(file_name, nbr_features, batch_size, train, workers, seed, length=None):
    # Dataset
    dataset = load_file(file_name, nbr_features, seed=seed, is_train=train)
    
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             **kwargs)

    return dataloader, dataset
