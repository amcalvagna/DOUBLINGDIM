#!/usr/bin/env bash

python3 ./reverse_check.py reverse.CIFAR10
python3 ./reverse_check.py reverse.CIFAR100

#./makeall.sh test.ResNet18IMG.WM38K.2 --run
#./makeall.sh test.ResNet18IMG.Cifar100.2 --run
#./makeall.sh test.ResNet18IMG.Cifar10.0 --run
