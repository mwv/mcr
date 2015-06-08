#!/bin/bash

monkeys=( blue titi colobus all )
for monkey in "${monkeys[@]}"
do
    echo $monkey
    data=resources/$monkey.isolated.train
    cfg=isolated.gridsearch.cfg
    out=results/$monkey.isolated.model.joblib.pkl
    python train_isolated.py $data $cfg $out -j 30 -v
    echo
done
