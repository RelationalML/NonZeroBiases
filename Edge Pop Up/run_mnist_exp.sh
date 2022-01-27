#! /bin/bash --
# for sparsity in 0.01 0.05 0.1 0.2

epochs=50
for lr in 1e-3
do
    for seed in 1 2 3
    do
        for levels in 10
        do
            for sparsity in 0.1
            do

                python ae_mnist.py --model AE --sparsity $sparsity --loss MSE --save_file runs-mnist-mse-ae-norm-modspar --dataset MNIST --anneal True --initBias kn-nonzero-bias --epochs $epochs --levels $levels --lr $lr --seed $seed |& tee log_kn_nonzerobias_seed${seed}_sparsity${sparsity}_levels${levels}_scaled.txt

                python ae_mnist.py --model AE --sparsity $sparsity --loss MSE --save_file runs-mnist-mse-ae-norm-modspar --dataset MNIST --anneal True --initBias kn-zero-bias --epochs $epochs --levels $levels --lr $lr --seed $seed |& tee log_kn_zerobias_seed${seed}_sparsity${sparsity}_levels${levels}_unscaled.txt

        done
        done
    done
done
