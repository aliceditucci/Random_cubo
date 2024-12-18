import numpy as np
import networkx as nx
from scipy.optimize import minimize
#import matplotlib.pyplot as plt

import argparse
import os
import sys
import pickle

sys.path.insert(0, '../')  #nel codice e' sys.path.insert(0, '../../') perche' sono due cartelle

from codes.functions import * 

def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=6)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=0)
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.3)
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers", required=False, type=int, default=1)

    parser.add_argument("--graph_type", help="type of graph, complete or 3regular", required=False, type=str, default='complete')

    parser.add_argument("--if_adsorting", help="adaptive sorting or random order of coefficients", required=False, type=int, default=0)

    args = parser.parse_args()

    n_qubits = args.N
    r = args.r
    alpha = args.alpha
    shots = args.shots
    tau = args.tau
    graph_type = args.graph_type
    layer = args.layer

    if_adsorting = args.if_adsorting
    
    if shots == 0:# exact simulation
        shots = None
        approximation = True
    else:#simulation with finite shots
        approximation = False
    
    if if_adsorting == 0:
        if_adsorting = False
    else:
        if_adsorting = True
        invert = True

    shotsmulti = 10 #to use more shots to evaluate the final circuit

    print('\nN: {}, \nr: {}, \nalpha: {}, \nshots: {}, \ngraph_type: {},  \ntau: {}, \nif_adsorting: {}'\
        .format(n_qubits, r, alpha, shots, graph_type, tau, if_adsorting))
    
    
    #make data dir
    dir_0 = './data_iter_adap_sorting'
    os.makedirs(dir_0, exist_ok=True)

    instance_dir = '../instances/'+ graph_type + '/N_' + str(n_qubits )
    with open(instance_dir + '/QUBO_' + str(n_qubits ) + 'V_'+ 'r_'+ str(r)+ '.gpickle', 'rb') as f:
        G = pickle.load(f)
    coeff_list = np.loadtxt(instance_dir + '/QUBO_coeff_' + str(n_qubits) + 'V_'+ 'r_'+ str(r)+ '.txt')
    edge_list = list(itertools.chain.from_iterable(partition_graph(G)))

    print('\n\n#############################################################################')
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits :]

    #extrcact Hamiltonian, edge coefficients and eigen list
    H = Hamiltonian_qubo(n_qubits, edge_list, h_list, J_list)  
    # eigen_list = H.to_spmatrix().diagonal().real
    eigen_list = H.to_matrix(sparse=True).diagonal().real

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})            #Initialize dictionary for Pauli Z term with coefficient from from h_list
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)})   #Initialize dictionary for Pauli ZZ term with coefficient from from J_list

    expz_array = np.array([0]*n_qubits)
    circ_init = initial_state_ry(n_qubits, expz_array)
     
    edge_params_dict, params_init, circ = get_parameters(\
    n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_list, shots, approximation) 

    poss, exp_poss_dict, cvar, expz_array = compute_expectations(n_qubits, eigen_list, circ, shotsmulti*shots, alpha)

    best_cvar = cvar 
    sorting = False
    best_abs = False
    best_inv = False

    if if_adsorting:
        for Abs in [True, False]:
            for invert in [True, False]:

                expz_array = np.array([0]*n_qubits)
                sorted_edge_list, edges_columns = sorted_gates(J_list, edge_list, Abs, invert)

                circ_init = initial_state_ry(n_qubits, expz_array)
                edge_params_dict, params_init, circ = get_parameters(\
                n_qubits, tau, layer, circ_init, edge_coeff_dict, sorted_edge_list, eigen_list, shots, approximation)     
                poss, exp_poss_dict, cvar, expz_array = compute_expectations(n_qubits, eigen_list, circ, shotsmulti*shots, alpha)

                if cvar < best_cvar:
                    best_cvar = cvar
                    sorting = True
                    best_abs = Abs
                    best_inv = invert

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
    steps_entropy_dict = {}

    for step in range(5):
        circ_init = initial_state_ry(n_qubits, expz_array)

        edge_params_dict, params_init, circ = get_parameters(\
        n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_list, shots, approximation)
        
        if not shots:
            entropy = entanglement_entropy(circ , range(round(n_qubits/2)), [2]*n_qubits, tol=1e-12)[0]
        else:
            entropy = 0 

        poss, exp_poss_dict, cvar, expz_array = compute_expectations(n_qubits, eigen_list, circ, shotsmulti*shots, alpha)

        steps_edge_params_dict['step_'+str(step)] = edge_params_dict
        steps_exp_poss_dict['step_'+str(step)] = exp_poss_dict
        steps_cvar_dict['step_'+str(step)] = cvar
        steps_entropy_dict['entropy_' + str(step)] = entropy

        print('\nstep: ', step)
        print(f"fidelity: {poss},  cvar: {cvar:.6f},  entropy: {entropy}")
        print('exp_poss_dict', list(exp_poss_dict['l_1'].items())[:10])
        print(f"expz_array: {expz_array}")
    
    num_params = len(params_init)   

    dir_0 = './data_iter_adap_sorting'
    os.makedirs(dir_0, exist_ok=True)
    dir_name =  dir_0 + '/graph_{}/shots_{}/num_variables_{:03d}/params_{}_layer_{}/alpha_{}/r_{}/'\
        .format(graph_type,  shots, n_qubits, num_params, layer, alpha, r)
    os.makedirs(dir_name, exist_ok=True)

        
    gi_file_path = dir_name + 'ifadsorting_{}_tau_{}_entropy_saved.pkl'.format(if_adsorting, tau)
    #gi_file_path = dir_name + 'ifadsorting_{}_tau_{}_entro.pkl'.format(if_adsorting, tau)

    save_data = {
                'edge_order': edge_list,
                'steps_edge_params_dict': steps_edge_params_dict,
                'params_list': params_init, ## just for good formula to run vqe
                'steps_exp_poss_dict': steps_exp_poss_dict,
                'steps_cvar_dict': steps_cvar_dict,
                'steps_entropy_dict': steps_entropy_dict,
                # 'entropy_list':entropy_list,
                'sorting' : sorting,
                'abs' : best_abs,
                'invert' : best_inv
                    }
    
    with open(gi_file_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print('Write sucessfully to ' + gi_file_path)


if __name__ == "__main__":
    main()