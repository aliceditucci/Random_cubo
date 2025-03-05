import numpy as np
import networkx as nx
from scipy.optimize import minimize
#import matplotlib.pyplot as plt

import argparse
import os
import sys
import pickle

import neal  # Simulated Annealing Sampler
from collections import OrderedDict

import itertools
import time
import pickle

sys.path.insert(0, '../')  #nel codice e' sys.path.insert(0, '../../') perche' sono due cartelle
from qubo_hamiltonian import *

# from VQE_CVaR import VQE, partition_N

from warm_start_paramenters import *
from warm_start_parameters_lightcone import *



def brute_force(n_qubits, edge_list, h_list, J_list):
    H = Hamiltonian_qubo(n_qubits, edge_list, h_list, J_list)   #This is changes from the old version of the code!!

    t1 = time.time()
    eigen_list = H.to_matrix(sparse=True).diagonal().real
    eigen_ids = np.argsort(eigen_list)
    eigen_idvalue_dict = {int(id): float(eigen_list[id]) for id in eigen_ids}
    t2 = time.time()
    print("\ntime for brute force: ", t2-t1)
    print("eigen_idvalue_dict", list(eigen_idvalue_dict.items())[:10])

    return eigen_idvalue_dict

def simulated_annealing(n_qubits, edge_list, h_list, J_list):
    Q = {}
    for i in range(n_qubits):
        Q[(i, i)] = -2 * h_list[i]
        for edge, J in zip(edge_list, J_list):
            if i == edge[0] or i == edge[1]:
                Q[(i, i)] -= 2 * J

    for edge, J in zip(edge_list, J_list):
        Q[edge] = 4*J

    const = sum(h_list) + sum(J_list)


    t1 = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    # Run the sampler with parameters
    sampleset = sampler.sample_qubo(Q, num_reads=100000)
    t2 = time.time()
    print("\ntime for simulated annealing: ", t2-t1)

    # Get all solutions sorted by energy (lowest first)
    sorted_samples = sorted(sampleset.data(), key=lambda x: x.energy)

    # Use OrderedDict to remove duplicates while preserving order
    unique_solutions = OrderedDict()
    for sample in sorted_samples:
        key = tuple(sample.sample.items())  # Convert dict to tuple (hashable)
        if key not in unique_solutions:
            unique_solutions[key] = sample
        
    # Extract the best 10 unique solutions
    best_unique_samples = list(unique_solutions.values())[:10]

    # Function to reverse bit order and convert to decimal
    def bitstring_to_decimal(sample_dict):
        sorted_bits = [sample_dict[q] for q in sorted(sample_dict.keys(), reverse=True)]  # Reverse order
        bitstring = ''.join(map(str, sorted_bits))  # Convert list to bitstring
        decimal_value = int(bitstring, 2)  # Convert to decimal
        return bitstring, decimal_value

    # Print the best 10 unique results with decimal conversion
    eigen_idvalue_dict = {}
    for i, sample in enumerate(best_unique_samples):
        bitstring, decimal_value = bitstring_to_decimal(sample.sample)
        eigen_idvalue_dict[decimal_value] =  round(sample.energy + const, 4)
        print(f"Rank {i+1}: Bitstring={bitstring}, Decimal={decimal_value}, Energy={round(sample.energy + const, 4)}")
    
    return eigen_idvalue_dict


def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=20)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.3)
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers", required=False, type=int, default=1)
    
    parser.add_argument("--backend_method", help="backend method for simulation, statevector or 'matrix_product_state'", required=False, type=str, default='matrix_product_state')
    parser.add_argument("--bond", help="bond dimension for matrix product state", required=False, type=int, default=100)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=10000)

    parser.add_argument("--graph_density", help="density of the graph, 1 for complete graph, 0 for 3reg graph", required=False, type=float, default=0.0)
    parser.add_argument("--if_adsorting", help="adaptive sorting or random order of coefficients", required=False, type=int, default=1)
    parser.add_argument("--if_analytic", help="if use analytic way to estimate parameters in product states", required=False, type=int, default=0)
   


    args = parser.parse_args()

    n_qubits = args.N
    r = args.r
    alpha = args.alpha
    tau = args.tau
    layer = args.layer

    backend_method = args.backend_method
    bond = args.bond
    shots = args.shots

    graph_density = args.graph_density
    if_adsorting = args.if_adsorting
    if_analytic = args.if_analytic
    
    if if_adsorting == 0:
        if_adsorting = False
    else:
        if_adsorting = True
        invert = True
    
    print("\n\n#############################################################################")

    print("\nn_qubits:            ", n_qubits)
    print("r:                   ", r)
    print("alpha:               ", alpha)
    print("tau:                 ", tau)
    print("layer:               ", layer)

    print("backend_method:      ", backend_method)
    print("bond:                ", bond)
    print("shots:               ", shots)

    print("graph_density:       ", graph_density)
    print("if_adsorting:        ", if_adsorting)
    print("if_analytic:         ", if_analytic)

    #endregion

    
    
    #make data dir
    if if_analytic == 1:
        dir_0 = './data_iter_adap_sorting_analytic/'
        
    else:
        dir_0 = './data_iter_adap_sorting_nomeasure/'
        
    
    os.makedirs(dir_0, exist_ok=True)

    #region load qubo instances, get Hamiltonian and edge_coeff_dict


    graph_file = f"./instances/{int(graph_density * 100):03}/N_{n_qubits}/QUBO_{n_qubits}V_r_{r}.gpickle"
    coeffs_file = f"./instances/{int(graph_density * 100):03}/N_{n_qubits}/QUBO_coeff_{n_qubits}V_r_{r}.txt"

    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    coeff_list = np.loadtxt(coeffs_file)
    edge_list = list(G.edges)
    

    print('\n\n#############################################################################')
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits : n_qubits + len(edge_list)]

    #get the exact solution
    if n_qubits <= 2:
        print('\n\nstart brute force')
        eigen_idvalue_dict = brute_force(n_qubits, edge_list, h_list, J_list)
        #print('eigen_idvalue_dict', list(eigen_idvalue_dict.items())[:10])
    else:
        print('\n\nstart simulated annealing')
        eigen_idvalue_dict = simulated_annealing(n_qubits, edge_list, h_list, J_list)
       # print('eigen_idvalue_dict', eigen_idvalue_dict)
   

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)}) #CHANGED COMPARED TO THE OLD CODE
    # print('edge_coeff_dict', edge_coeff_dict.items())

    #endregion


    if backend_method == 'matrix_product_state':
        backendoptions = {'method':backend_method, 'matrix_product_state_max_bond_dimension': bond, 'shots': shots}
    else:
        backendoptions = {'method':backend_method, 'shots': shots}

    #region get best sorting way from cvar value
    print("\n\n##start random order circuit")
    t1 = time.time()
    expz_array = np.array([0]*n_qubits)
    #expz_array = np.array([1, 1, 1, -1])   ## z0, z1 , ... , zn-1
    circ_init = initial_state_ry(n_qubits, expz_array)

    edge_params_dict, params_init, exp_poss_dict = get_good_initial_params_measure_iterate(\
    n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_idvalue_dict, shots, backendoptions, if_analytic)     #HO SPOSTATO IL SALVADATI A DOPO
    cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)
    t2 = time.time()
    print("time to run whole circuit: ", t2-t1)
    print('cvar of random order: ', cvar)
    print('exp_pos_dict', list(exp_poss_dict['l_1'].items())[:5])
    print('expz_array: ', expz_array)

    ############################################################
    best_cvar = cvar 
    sorting = False
    best_abs = False
    best_inv = False

    print("\n\n##start adaptive sorting")
    if if_adsorting:
        for Abs in [True, False]:
            for invert in [True, False]:

                expz_array = np.array([0]*n_qubits)
                sorted_edge_list, edges_columns = sorted_gates(J_list, edge_list, Abs, invert)

                circ_init = initial_state_ry(n_qubits, expz_array)
                edge_params_dict, params_init, exp_poss_dict = get_good_initial_params_measure_iterate(\
                n_qubits, tau, layer, circ_init, edge_coeff_dict, sorted_edge_list, eigen_idvalue_dict, shots, backendoptions, if_analytic)     #HO SPOSTATO IL SALVADATI A DOPO
                cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)

                print(f'\ncvar of Abs:{Abs}, invert:{invert} cvar: {cvar}')
                print('(exp, poss):', [(float(tp[0][0]), tp[1]) for tp in list(exp_poss_dict['l_1'].items())[:5]])
                # print('expz_array: ', expz_array)


                if cvar < best_cvar:
                    best_cvar = cvar
                    sorting = True
                    best_abs = Abs
                    best_inv = invert
            
    #endregion


    #region iterate and save data
    print('\n\n#############################################################################')
    print('start iterate')
    if sorting:
        edge_list, edges_columns = sorted_gates(J_list, edge_list, best_abs, best_inv)
        print('\nchanged: ', sorting, best_abs, best_inv)

    expz_array = np.array([0]*n_qubits)
    steps_edge_params_dict = {}
    steps_exp_poss_dict = {}
    steps_cvar_dict = {}
    for step in range(5):
        circ_init = initial_state_ry(n_qubits, expz_array)
        edge_params_dict, params_init, exp_poss_dict = get_good_initial_params_measure_iterate(\
        n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_idvalue_dict, shots, backendoptions, if_analytic)     #HO SPOSTATO IL SALVADATI A DOPO
        cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)
        #entropy = entanglement_entropy(state_all , range(round(n_qubits/2)), [2]*n_qubits, tol=1e-12)[0]

        steps_edge_params_dict['step_'+str(step)] = edge_params_dict
        steps_exp_poss_dict['step_'+str(step)] = list(exp_poss_dict['l_1'].items())[:100]
        steps_cvar_dict['step_'+str(step)] = cvar
        print('\nstep: ', step)
        print(f"cvar: {cvar:.6f}")
        print('(exp, poss):', [(float(tp[0][0]), tp[1]) for tp in list(exp_poss_dict['l_1'].items())[:5]])
        print(f"expz_array: {expz_array}")

    num_params = len(params_init)   

    dir = dir_0 + f"{int(graph_density * 100):03}" + '/N_' + str(n_qubits)
    os.makedirs(dir_0, exist_ok=True)
    dir_name =  dir + '/method_{}/shots_{}/layer_{}/alpha_{}/r_{}/'\
        .format(backend_method, shots, layer, alpha, r)
    os.makedirs(dir_name, exist_ok=True)

        
    gi_file_path = dir_name + 'ifadsorting_{}_tau_{}_bond_{}.pkl'.format(if_adsorting, tau, bond)

    save_data = {
                'edge_order': edge_list,
                'steps_edge_params_dict': steps_edge_params_dict,
                'params_list': params_init, ## just for good formula to run vqe
                'steps_exp_poss_dict': steps_exp_poss_dict,
                'steps_cvar_dict': steps_cvar_dict,
                'sorting' : sorting,
                'abs' : best_abs,
                'invert' : best_inv
                    }
    
    with open(gi_file_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print('Write sucessfully to ' + dir_name)

    # #endregion

if __name__ == "__main__":
    main()
    