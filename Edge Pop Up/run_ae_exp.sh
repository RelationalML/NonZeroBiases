epochs=50
for lr in 1e-3
do
    for seed in 3 4 5
    do
        for levels in 10
        do
            for sparsity in 0.1 0.01 0.05
            do



                python ae_main.py --model AE --sparsity $sparsity --loss MSE --anneal True --save_file runs-cifar-ae --initBias kn-nonzero-bias --epochs $epochs --levels $levels --lr $lr --seed $seed |& tee log_kn_zerobias_seed${seed}_sparsity${sparsity}_levels${levels}_scaled.txt

                python ae_main.py --model AE --sparsity $sparsity --loss MSE --anneal True --save_file runs-cifar-ae --initBias kn-zero-bias --epochs $epochs --levels $levels --lr $lr --seed $seed |& tee log_kn_zerobias_seed${seed}_sparsity${sparsity}_levels${levels}_unscaled.txt

                
        done
        done
    done
done
