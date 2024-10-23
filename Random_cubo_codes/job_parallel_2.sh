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
sorting=$9
absolute=${10}
invert=${11}

source /lustre/fs24/group/cqta/atucci/environment/bin/activate

/lustre/fs24/group/cqta/atucci/environment/bin/python3 /lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/run_qubo_iterate_adap_sorting.py --N "$N" --r "$r" --alpha "$alpha" --shots "$shots" --ansatz_type "$ansatz_type" --layer "$layer" --tau "$tau" --initialization "$initialization" --sorting "$sorting" --absolute "$absolute" --invert "$invert"
