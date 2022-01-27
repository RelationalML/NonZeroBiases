import numpy as np
import matplotlib.pylab as plt
from itertools import combinations
import torch
import torch.nn as nn
import pickle

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

def draw_ortho(nout, nin):
    H = np.random.randn(nout, nin)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat = u @ vh
    return mat/np.sqrt(np.mean(mat**2))

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
    variables = np.random.uniform(-1,1,nbrVar)
    cand, ind = exhaustive(target, variables*wp1, eps, 10)
    err = np.abs(target-cand)
    subset = variables[ind]
    if err <= eps:
        return cand, subset, ind, addwidth
    else:
        addwidth = addwidth+1
        return solve_subset_sum(target, nbrVar, eps, addwidth, wp1)
    
def solve_subset_sum_general(target, eps, variables):
    cand, ind = exhaustive(target, variables, eps, 15)
    err = np.abs(target-cand)
    subset = variables[ind]
    if err > eps:
        print("failed subset sum")
    return cand, subset, ind

    
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
    indIn = indIn[degOut > 0]
    #ouputs in need of construction
    degIn = np.sum(np.abs(wt) > eps, axis=1)
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    n_out = len(indOut)
    n_in = len(indIn)
    #assume that 
    dimp2 = tuple([n_out, rho*(n_in+1)])
    dimp1 = tuple([rho*(n_in+1), wt.shape[1]]) 
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    bp1 = np.zeros(rho*(n_in+1))
    #first layer init
    for i in range(n_in):
        indRange = np.arange(i*rho, (i+1)*rho)
        wp1[indRange,indIn[i]] = np.random.uniform(-1,1,rho)
    #bias
    indRange = np.arange(n_in*rho, (n_in+1)*rho)
    bp1[indRange] = np.random.uniform(0,1,rho) #for simplicity, these are aranged like that. With high probability we can first select biases nodes with positive biases to identify the nodes that we want to prune down to biases
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
    return wpruned, bpruned, architect

def weight_pruning_general(w01, w02, rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt):
    addwidth = 0
    indExtend = 0
    for i in range(n_in):
        indRange = np.arange(rho*i,rho*(i+1))
        wcand = w01[indRange,i].copy()
        indRangePos = indRange[wcand>0]
        indRangeNeg = indRange[wcand<0]
        wcandPos = wcand[wcand>0]
        wcandNeg = wcand[wcand<0]
        for j in range(n_out):
            target = wt[indOut[j],indIn[i]]
            if(np.abs(target) > eps):
                param, subset, ind = solve_subset_sum_general(target, eps, wcandPos*w02[j,indRangePos])
                #print(ind)
                #print(len(indRangePos))
                if np.abs(param-target) <= eps:
                    wp2[j, indRangePos[ind]] = w02[j,indRangePos[ind]]
                    wp1[indRangePos[ind],i] = wcandPos[ind]
                else:
                    indExtend = indExtend+1
                    jj =  n_out+indExtend
                    param, subset, ind = solve_subset_sum_general(target, eps, wcandPos*w02[jj,indRangePos])
                    wp2[j, indRangePos[ind]] = w02[jj,indRangePos[ind]]
                    wp1[indRangePos[ind],i] = wcandPos[ind]
                if np.abs(param-target) <= eps:
                    param, subset, ind = solve_subset_sum_general(target, eps, wcandNeg*w02[j,indRangeNeg])
                    wp2[j, indRangeNeg[ind]] = w02[j,indRangeNeg[ind]]
                    wp1[indRangeNeg[ind],i] = wcandNeg[ind]
                else:
                    indExtend = indExtend+1
                    jj =  n_out +indExtend
                    param, subset, ind = solve_subset_sum_general(target, eps, wcandNeg*w02[jj,indRangeNeg])
                    wp2[j, indRangeNeg[ind]] = w02[jj,indRangeNeg[ind]]
                    wp1[indRangeNeg[ind],i] = wcandNeg[ind]
    return wp2, indExtend

def weight_pruning_general_relu(w01, w02, rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt):
    addwidth = 0
    indExtend = 0
    for i in range(n_in):
        indRange = np.arange(rho*i,rho*(i+1))
        wcand = np.abs(w01[indRange,i].copy()) #both the positive weight and the negative one are available anyways in the looks linear init
        for j in range(n_out):
            target = wt[indOut[j],indIn[i]]
            if(np.abs(target) > eps):
                param, subset, ind = solve_subset_sum_general(target, eps, wcand*w02[j,indRange])
                #print(ind)
                #print(len(indRangePos))
                if np.abs(param-target) <= eps:
                    wp2[j, indRange[ind]] = w02[j,indRange[ind]]
                    wp1[indRange[ind],i] = wcand[ind]
                else:
                    indExtend = indExtend+1
                    jj = n_out + indExtend
                    if jj >= w02.shape[0]:
                        print("warning: no backup left")
                        jj = j
                    param, subset, ind = solve_subset_sum_general(target, eps, wcand*w02[jj,indRange])
                    wp2[j, indRange[ind]] = w02[jj,indRange[ind]]
                    wp1[indRange[ind],i] = wcand[ind]
    return wp2, indExtend

def bias_pruning_general(b01, w02, indstart, n_out, eps, wp2, bp, indOut, bt, usedbackup):
    for j in range(n_out):
        target = bt[indOut[j]]
        if(np.abs(target) > eps):
            param, subset, ind = solve_subset_sum_general(target, eps, b01[indstart:]*w02[j,indstart:])
            if np.abs(param-target) <= eps:
                wp2[j, indstart + ind] = w02[j,indstart+ind]
                bp[indstart + ind] = b01[indstart+ind]
            else:
                usedbackup = usedbackup+1
                jj = j+usedbackup
                if j+usedbackup >= w02.shape[0]:
                    print("warning: no backup left")
                    jj = j
                param, subset, ind = solve_subset_sum_general(target, eps, b01[indstart:]*w02[jj,indstart:])
                wp2[j, indstart + ind] = w02[jj,indstart+ind]
                bp[indstart + ind] = b01[indstart+ind]
    return wp2, bp, usedbackup

def prune_layer_general(wt, bt, w01, w02, b0, eps, backup):
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
    indIn = indIn[degOut > 0]

    #ouputs in need of construction
    degIn = np.sum(np.abs(wt) > eps, axis=1)
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    n_out = len(indOut)
    n_in = len(indIn)
    
    dimp2 = w02.shape
    dimp1 = w01.shape
    
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    rho = int(np.ceil(dimp1[0]/(n_in+1)))
    print("rho: "+str(rho))
    bp1 = np.zeros(dimp1[0])
    wp2 = np.zeros(dimp2)
    #weight pruning
    wp2, usedbackup = weight_pruning_general(w01, w02, rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt)
    #biases
    wp2, bp1, usedbackup = bias_pruning_general(b0, w02, n_in*rho, n_out, eps, wp2, bp1, indOut, bt, usedbackup)  
    print("backup: " +str(usedbackup))
    return wp1, wp2, bp1

def prune_layer_general_relu(wt, bt, w01, w02, b0, eps, backup):
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
    indIn = indIn[degOut > 0]
   
    #ouputs in need of construction
    degIn = np.sum(np.abs(wt) > eps, axis=1)
    indOut = np.arange(n_out)
    indOut = indOut[degIn > 0]
    n_out = len(indOut)
    n_in = len(indIn)
   
    #assume that
    dimp2 = w02.shape
    dimp1 = w01.shape
   
    #parameters after pruning
    wp1 = np.zeros(dimp1)
    rho = int(np.ceil(dimp1[0]/(n_in+1)))
    print("rho: "+str(rho))
    bp1 = np.zeros(dimp1[0]) 
    wp2 = np.zeros(dimp2)

    #weight pruning
    wp2, usedbackup = weight_pruning_general_relu(w01, w02, rho, n_in, n_out, eps, wp1, wp2, indIn, indOut, wt)
    wp2neg = - wp2

    #biases
    wp2, bp1, usedbackup = bias_pruning_general(b0, w02, n_in*rho, n_out, eps, wp2, bp1, indOut, bt, usedbackup)  
    print("backup: " +str(usedbackup))
    return wp1, wp2, wp2neg, bp1, usedbackup


def prune_fc_general(wt, bt, w0, b0, eps, backup):
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
        if l < L-1:
            wp1, wp2, bp1 = prune_layer_general(wt[l], bt[l], w0[2*l], w0[2*l+1], b0[2*l], eps, backup)
        else:
            wp1, wp2, bp1 = prune_layer_general(wt[l], bt[l], w0[2*l], w0[2*l+1], b0[2*l], eps, 0)
        wpruned.append(wp1)
        wpruned.append(wp2)
        bpruned.append(bp1)
        bpruned.append(np.zeros(wp2.shape[0]))
        architect[2*l+1] = wp2.shape[1] 
        architect[2*l+2] = wp2.shape[0]
    return wpruned, bpruned, architect

def prune_fc_general_relu(wt, bt, w0, b0, eps, backup):
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
        if l < L-1:
            wp1, wp2, wp2neg, bp1, usedbackup = prune_layer_general_relu(wt[l], bt[l], w0[2*l], w0[2*l+1], b0[2*l], eps, backup)
        else:
            wp1, wp2, wp2neg, bp1, usedbackup = prune_layer_general_relu(wt[l], bt[l], w0[2*l], w0[2*l+1], b0[2*l], eps, 0)
        wp1 = np.vstack([wp1,-wp1])
        wp2 = np.hstack([wp2,wp2neg])
        bp1 = np.hstack([bp1,0*bp1]) #bp1 = np.hstack([bp1,-bp1])
        wpruned.append(wp1)
        wpruned.append(wp2)
        bpruned.append(bp1)
        bpruned.append(np.zeros(wp2.shape[0]))
        architect[2*l+1] = wp2.shape[1] 
        architect[2*l+2] = wp2.shape[0]+usedbackup
    return wpruned, bpruned, architect

def prune_fc_ortho(wtl, btl, rho, backup, eps):
    w0 = list()
    b0 = list()
    #rho=50
    #backup = 10
    for l in range(L):
        nout, nin = wtl[l].shape
        if l > 0:
            w01 = draw_ortho(rho*(nin+1), nin+backup)
        else: 
            w01 = draw_ortho(rho*(nin+1), nin)
        if l < L-1:
            w02 = draw_ortho(nout+backup, rho*(nin+1))
        else:
            w02 = draw_ortho(nout, rho*(nin+1))
        w0.append(w01)
        w0.append(w02)
        b01 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=rho*(nin+1))) #just for simplicity, otherwise would need to select right bias indicees, which is possible with high probability
        b02 = np.zeros(nout)
        b0.append(b01)
        b0.append(b02)
    wprunedOrtho, bprunedOrtho, architectOrtho =  prune_fc_general(wtl, btl, w0, b0, eps, backup)
    return wprunedOrtho, bprunedOrtho, architectOrtho

def prune_fc_ortho_relu(wtl, btl, rho, backup, eps):
    w0 = list()
    b0 = list()
    #rho=50
    #backup = 10
    rho = int(rho/2)
    for l in range(L):
        nout, nin = wtl[l].shape
        if l > 0:
            w01 = draw_ortho(rho*(nin+1), nin+backup)
        else: 
            w01 = draw_ortho(rho*(nin+1), nin)
        if l < L-1:
            w02 = draw_ortho(nout+backup, rho*(nin+1))
        else:
            w02 = draw_ortho(nout, rho*(nin+1))
        w0.append(w01)
        w0.append(w02)
        b01 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=rho*(nin+1))) #just for simplicity, otherwise would need to select right bias indicees, which is possible with high probability
        b02 = np.zeros(nout)
        b0.append(b01)
        b0.append(b02)
    wprunedOrtho, bprunedOrtho, architectOrtho =  prune_fc_general_relu(wtl, btl, w0, b0, eps, backup)
    return wprunedOrtho, bprunedOrtho, architectOrtho


def target_net(pathTarget): #, pathMask):
    target_dict = torch.load(pathTarget, map_location=torch.device('cpu'))
    #mask_dict = torch.load(pathMask)
    target_params = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), target_dict.items()))
    #wt = target_params[list(target_params.keys())[0]]
    wtl = list()
    btl = list()
    L=0
    width=0
    scale = list()
    for ll in target_params.keys():
        if ll.endswith("weight"):
            #mask = mask_dict[ll].data.clone().cpu().detach().numpy()
            wt = target_params[ll].data.clone().cpu().detach().numpy()
            #print(str(L)  + ": " + str(np.sum(mask)/np.prod(wt.shape)))
            #wtl.append(wt*mask)
            wtl.append(wt)
            sc = np.max(np.abs(wtl[L]))
        if ll.endswith("bias"):
            bt = target_params[ll].data.clone().cpu().detach().numpy()
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
    L = len(bias)
    for l in range(L):
        nn = nn + np.sum(np.abs(bias[l]) >= eps)
    return nn

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

#load target network, adapt target network directory            
L, width, wtl, btl, scale = target_net('PATH/model_ep40_it0.pth')
print(L)
print(width)
print(scale)
nbr = number_params(wtl, btl, 0.01)
print(nbr)

#orthogonal pruning
wprunedOrtho, bprunedOrtho, architectOrtho = prune_fc_ortho_relu(wtl, btl, 40, 0, 0.01)





