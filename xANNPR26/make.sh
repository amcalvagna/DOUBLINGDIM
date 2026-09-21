#!/usr/bin/env bash

python3 ./reverse_check.py reverse.CIFAR10
python3 ./reverse_check.py reverse.CIFAR100

python3 ./makeANNPR26Plots.py WM38
python3 ./makeANNPR26Plots.py CIFAR10
python3 ./makeANNPR26Plots.py CIFAR100