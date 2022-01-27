import numpy as np
import matplotlib.pylab as plt
from itertools import combinations
import torch
import torch.nn as nn
import pickle
from Utils import load

device = torch.device("cpu")

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, weightl, biasl):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.data = torch.tensor(weightl)
        self.bias.data = torch.tensor(biasl)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

def fc(weight, bias, nonlinearity=nn.ReLU()): 
  L = len(weight)
  # Linear feature extractor
  modules = [nn.Flatten()]
  for i in range(L-1):
    modules.append(Linear(weight[i].shape[1], weight[i].shape[0], weight[i], bias[i]))
    modules.append(nonlinearity)
    
  modules.append(Linear(weight[L-1].shape[1], weight[L-1].shape[0], weight[L-1], bias[L-1]))
  model = nn.Sequential(*modules)
  return model

def subset_fixed_size_best(target, numbers, subsize, errBest):
    n = len(numbers)
    cand = 0
    indBest = np.array([np.NAN])
    for ind in combinations(range(n),subsize):
        inda = np.array(ind,dtype="int")
        napprox = np.sum(numbers[inda])
        diff = np.abs(target-napprox)
        if diff < errBest:
            errBest = diff
            cand = napprox
            indBest = inda
    return cand, indBest, errBest

def exhaustiveBest(target, numbers, nmax):
    n = len(numbers)
    err = np.abs(target)
    errBest = err
    cand = 0
    indBest = np.array([-1])
    nmax = min(nmax, n)
    for k in range(nmax):
        cank, indk, errk = subset_fixed_size_best(target, numbers, k, errBest)
        if errk < errBest:
            errBest = errk
            cand = cank
            indBest = indk
    return cand, indBest

def subset_fixed_size(target, numbers, eps, subsize, errBest):
    n = len(numbers)
    cand = 0
    indBest = np.array([np.NAN])
    for ind in combinations(range(n),subsize):
        inda = np.array(ind,dtype="int")
        napprox = np.sum(numbers[inda])
        diff = np.abs(target-napprox)
        if diff < errBest:
            errBest = diff
            cand = napprox
            indBest = inda
        if diff <= eps:
            break
    return cand, indBest, errBest

def exhaustive(target, numbers, eps, nmax):
    n = len(numbers)
    err = np.abs(target)
    errBest = err
    cand = 0
    indBest = np.array([-1])
    nmax = min(nmax, n)
    for k in range(nmax):
        cank, indk, errk = subset_fixed_size(target, numbers, eps, k, errBest)
        if errk < errBest:
            errBest = errk
            cand = cank
            indBest = indk
        if errBest <= eps:
            break
    return cand, indBest
    
def solve_subset_sum(target, nbrVar, eps, addwidth, wp1):
    # print('solve called once')

    variables = np.random.uniform(-1, 1, nbrVar)
    cand, ind = exhaustive(target, variables*wp1, eps, 15)
    err = np.abs(target-cand)
    subset = variables[ind]
    addwidth += 1
    if err <= eps:
        return cand, subset, ind, addwidth
    else:
        addwidth = addwidth+1
        return solve_subset_sum(target, nbrVar, eps, addwidth, wp1)
    # return cand, subset, ind, addwidth
    
def weight_pruning(rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt):
    addwidth = 0
    for i in range(n_in):
        indRange = np.arange(rho*i,rho*(i+1))
        wcand = wp1[indRange,indIn[i]].copy()
        indRangePos = indRange[wcand>0]
        indRangeNeg = indRange[wcand<0]
        wcandPos = wcand[wcand>0]
        wcandNeg = wcand[wcand<0]
        for j in range(n_out):
            target = wt[indOut[j],indIn[i]]
            if(np.abs(target) > eps):
                param, subset, ind, nbrTrials = solve_subset_sum(target, len(wcandPos), eps, 0, wcandPos)
                wp2[j, indRangePos[ind]] = subset
                addwidth = addwidth+nbrTrials
                param, subset, ind, nbrTrials = solve_subset_sum(target, len(wcandNeg), eps, 0, wcandNeg)
                wp2[j, indRangeNeg[ind]] = subset
                addwidth = addwidth+nbrTrials
    return wp2, addwidth

def bias_pruning(rho, n_in, n_out, eps, wp2, bp, indIn, indOut, bt):
    addwidth = 0
    indRange = np.arange(rho*n_in,rho*(n_in+1))
    bp1 = bp[indRange]
    for j in range(n_out):
        target = bt[indOut[j]]
        if(np.abs(target) > eps):
            param, subset, ind, nbrTrials = solve_subset_sum(target, rho, eps, 0, bp1)
            wp2[j, n_in*rho + ind] = subset
            addwidth = addwidth+nbrTrials
    return wp2, addwidth  

def prune_layer(wt, bt, rho, eps):
    # wt: target tensor of dimension nt_out x nt_in: target weight parameters
    # bt: target bias vector
    # rho: mutiplicity/ nbr of copies that should be constructed of each input in the intermediate layer (40 is usually a good size)
    # eps allowed error in each parameter
    dimt = wt.shape
    n_out = dimt[0]
    n_in = dimt[1]
    #inputs in need of construction:
    degOut = np.sum(np.abs(wt) > eps, axis=0)
    indIn = np.arange(n_in)
    
    #ouputs in need of construction
    degIn = np.sum(np.abs(wt) > eps, axis=1)
    indOut = np.arange(n_out)
    
    n_out = len(indOut)
    n_in = len(indIn)
    print('Nout, Nin')
    print(n_out, n_in)
    #assume that 
    dimp2 = tuple([n_out, rho*(n_in+1)])
    dimp1 = tuple([rho*(n_in+1), wt.shape[1]]) 
    
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    bp1 = np.zeros(rho*(n_in+1))
    
    
    #first layer init
    for i in range(n_in):
        indRange = np.arange(i*rho, (i+1)*rho)
        wp1[indRange,indIn[i]] = np.random.uniform(-1, 1, rho)
    #bias
    indRange = np.arange(n_in*rho, (n_in+1)*rho)
    bp1[indRange] = np.random.uniform(0, 1, rho) #for simplicity, these are aranged like that. With high probability we can first select biases nodes with positive biases to identify the nodes that we want to prune down to biases
    wp2 = np.zeros(dimp2)
    #weight pruning
    wp2, addwidth = weight_pruning(rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt)
    #biases
    wp2, addw = bias_pruning(rho, n_in, n_out, eps, wp2, bp1, indIn, indOut, bt)  
    addwidth = addwidth + addw
    return wp1, wp2, bp1, addwidth

def prune_fc(wt, bt, rho, eps):
    #wt: list of target weights
    #bt: list of target biases
    #rho: multiplicity of intermediate layer
    #eps: allowed error per parameter
    L = len(wt)
    wpruned = list()
    bpruned = list()
    #mother network architecture
    architect = np.zeros(2*L+1)
    architect[0] = wt[0].shape[1]
    for l in range(L):
        print("l: " + str(l))
        wp1, wp2, bp1, addwidth = prune_layer(wt[l], bt[l], rho, eps)
        wpruned.append(wp1)
        wpruned.append(wp2)
        bpruned.append(bp1)
        bpruned.append(np.zeros(wp2.shape[0]))
        architect[2*l+1] = wp2.shape[1] 
        architect[2*l+2] = wp2.shape[0] + addwidth
        print(wp1.shape, wp2.shape)
    return wpruned, bpruned, architect

def target_net(param_list): #, pathMask):
    
    target_params = dict(filter(lambda v: (v[0].endswith(('weight', 'bias'))), param_list))
    
    wtl = list()
    btl = list()
    L=0
    width=0
    scale = list()
    for name, param in param_list:
        if "weight" in name:
            wt = param.data.clone().cpu().detach().numpy()
            wtl.append(wt)
            sc = np.max(np.abs(wtl[L]))
        if "bias" in name:
            bt = param.data.clone().cpu().detach().numpy()
            btl.append(bt)
            width = max(width,len(bt))
            sc = max(sc,np.max(np.abs(wtl[L])))
            scale.append(sc)
            L=L+1
    scale = np.array(scale)
    
    return L, width, wtl, btl, scale

def number_params(weight, bias, eps):
    L = len(weight)
    nn=0
    for l in range(L):
        nn = nn + np.sum(np.abs(weight[l]) >= eps)
        print(str(l) + ": " + str(np.sum(np.abs(weight[l]) >= eps))) 
        print(np.max(np.abs(weight[l])), np.min(np.abs(weight[l])))
    L = len(bias)
    for l in range(L):
        nn = nn + np.sum(np.abs(bias[l]) >= eps)
    return nn

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

#load target network, adapt target network directory  

model = load.model('fc_mnist', 'default')(((1, 28, 28), 10), 
                                                     10, 
                                                     False,
                                                     False).to(device)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
mask_list = torch.load(PATH, map_location=torch.device('cpu'))

i = 0
param_list = []

for param in model.parameters():
    print(param.data.max(), param.data.min())
    param_list.append([mask_list[i][0], (param.data * mask_list[i][1])])
    i += 1
# print('Num Params: ', i)
# print('Num Mask Params: ', len(mask_list))

L, width, wtl, btl, scale = target_net(param_list)
print(L)
print(width)
print(scale)
nbr = number_params(wtl, btl, 0.01)
print(nbr)

for i in range(len(wtl)):
    eps = 0.01
    wt = wtl[i]
    dimt = wt.shape
    n_out = dimt[0]
    n_in = dimt[1]
    #inputs in need of construction:
    degOut = np.sum(np.abs(wt) > eps, axis=0)
    indIn = np.arange(n_in)
    indIn = indIn[degOut > 0]
    #ouputs in need of construction
    degIn = np.sum(np.abs(wt) > eps, axis=1)
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    print(wt.shape, indIn.shape, indOut.shape)



wpruned, bpruned, architect = prune_fc(wtl, btl, 40, 0.01)
print(architect)
with open('..ticket_relu_2L', 'wb') as f:
    pickle.dump([wpruned, bpruned, architect], f)

torch.save(wpruned, 'SAVEPATH/const_weight_list.pt')
torch.save(bpruned, 'SAVEPATH/const_bias_list.pt')

model = fc(wpruned, bpruned)
torch.save(model.state_dict(), 'SAVEPATH/constructed_model.pt')


