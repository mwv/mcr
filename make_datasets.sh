#!/bin/bash

monkeys=( blue titi colobus all )
modes=( isolated continuous )
for monkey in "${monkeys[@]}"
do
  for mode in "${modes[@]}"
  do
    echo $monkey $mode
    python make_dataset.py resources/annotation.csv $mode $monkey \
           resources/$monkey.$mode --split 0.2
    echo
  done
done
