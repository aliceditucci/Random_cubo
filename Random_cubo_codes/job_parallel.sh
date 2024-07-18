#!/bin/bash

# Extract arguments

N=$1
r=$2
alpha=$3
shots=$4
ansatz_type=$5
layer=$6
tau=$7
initialization=$8

source /lustre/fs24/group/cqta/atucci/environment/bin/activate

/lustre/fs24/group/cqta/atucci/environment/bin/python3 /lustre/fs24/group/cqta/atucci/QUBO/Random_cubo_codes/run_qubo.py --N "$N" --r "$r" --alpha "$alpha" --shots "$shots" --ansatz_type "$ansatz_type" --layer "$layer" --tau "$tau" --initialization "$initialization"
