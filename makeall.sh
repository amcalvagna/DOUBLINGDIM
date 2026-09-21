#!/usr/bin/env zsh
set -e   # stop if any command fails

if [[ "$*" == *"--reset"* ]]; then
    echo "resetting..."
    ./clean.sh $1 data coresets img log save   
    cp hyper_parameters.json $1

fi

if [[ "$*" == *"--profile"* ]]; then
    echo "profiling..."
    #DD_PROFILE=1 DD_PROFILE_DIR=./profiles python main.py $1
    DD_PROFILE=1 python main.py $1

fi

if [[ "$*" == *"--run"* ]]; then
    echo "Running experiments..."
    python main.py  $1
fi
