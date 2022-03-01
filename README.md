#######
Weak Lottery Pruning: This folder implements most of the pruning methods to obtain weak lottery tickets including Iterative Magnitude Pruning, Synaptic Flow, SNIP and GRASP. This code is taken from https://github.com/ganguli-lab/Synaptic-Flow.
Example run for iterative magnitude pruning, 
"python3 main.py --experiment multishot --expid construct-init-2 --model fc_mnist --dataset mnist --model-class default --train-batch-size 256 --prune-dataset-ratio 25 --test-batch-size 512 --pruner mag --prune-bias True --verbose --post-epochs 50 --pre-epochs 10 --level 10 --sparsity 0.01 --lr 0.01 --result-dir testFolder/ --mask-scope local --weight-decay 1e-3"

########
Edge Pop Up: This folder implements the edge popup and the scaled edge popup algorithm as detailed in our paper. We run experiments on CIFAR, MNIST, Autoencoders (CIFAR) and tabular data. This code is built on the work of the edge popup authors, from https://github.com/allenai/hidden-networks
Example run with,
"bash run_cifar_exp.sh"

########
Construction: This folder contains our implementation of the subset sum approximation to construct lottery tickets as per our proof, with different initializations.
Example run,
"python prune_2layers.py" for uniform and normal initialization
"python prune_2layers_orthogonal.py" for orthogonal initialization

