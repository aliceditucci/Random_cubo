#!/bin/bash

# Extract arguments

N=$1
r=$2
alpha=$3
shots=$4
ansatz_type=$5
initialization=$6

source /lustre/fs24/group/cqta/atucci/environment2/bin/activate

/lustre/fs24/group/cqta/atucci/environment2/bin/python3 /lustre/fs24/group/cqta/atucci/Random_cubo/VQE_largesize/VQE_QAOA/run_vqe.py --N "$N" --r "$r" --alpha "$alpha" --shots "$shots" --ansatz_type "$ansatz_type" --initialization "$initialization"
