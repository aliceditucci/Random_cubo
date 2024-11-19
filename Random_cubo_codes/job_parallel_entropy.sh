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

source /lustre/fs24/group/cqta/atucci/environment/bin/activate

/lustre/fs24/group/cqta/atucci/environment/bin/python3 /lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/run_qubo_iterate_adap_sorting_entropy.py --N "$N" --r "$r" --alpha "$alpha" --shots "$shots" --layer "$layer" --tau "$tau" --graph_type "$graph_type" --if_adsorting "$if_adsorting"


