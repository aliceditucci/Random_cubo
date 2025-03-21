#!/bin/bash

# Extract arguments

N=$1
r=$2
alpha=$3
shots=$4
tau=$5
layer=$6
graph_type=$7
if_adsorting=$8
if_analytic=$9
bond=$10
backend_method=$11

source /lustre/fs24/group/cqta/atucci/environment2/bin/activate

/lustre/fs24/group/cqta/atucci/environment2/bin/python3 /lustre/fs24/group/cqta/atucci/Random_cubo/codes/run_qubo_shots.py --N "$N" --r "$r" --alpha "$alpha" --shots "$shots" --layer "$layer" --tau "$tau" --graph_type "$graph_type" --if_adsorting "$if_adsorting" --if_analytic "$if_analytic" --bond "$bond" --backend_method "$backend_method"


