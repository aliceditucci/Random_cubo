import numpy as np
import networkx as nx
from scipy.optimize import minimize
#import matplotlib.pyplot as plt

import argparse
import os
import sys
import pickle

sys.path.insert(0, '../')  #nel codice e' sys.path.insert(0, '../../') perche' sono due cartelle
from qubo_hamiltonian import *

# from VQE_CVaR import VQE, partition_N

from warm_start_paramenters import *
from warm_start_parameters_lightcone import *

import itertools
import time
import pickle


def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=4)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.3)
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers", required=False, type=int, default=1)
    
    parser.add_argument("--backend_method", help="backend method for simulation, statevector or 'matrix_product_state'", required=False, type=str, default='statevector')
    parser.add_argument("--bond", help="bond dimension for matrix product state", required=False, type=int, default=100)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=0)

    parser.add_argument("--graph_density", help="density of the graph, 1 for complete graph", required=False, type=float, default=1)
    parser.add_argument("--if_adsorting", help="adaptive sorting or random order of coefficients", required=False, type=int, default=0)
   


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
    
    if if_adsorting == 0:
        if_adsorting = False
    else:
        if_adsorting = True
        invert = True

    print("n_qubits:            ", n_qubits)
    print("r:                   ", r)
    print("alpha:               ", alpha)
    print("tau:                 ", tau)
    print("layer:               ", layer)

    print("backend_method:      ", backend_method)
    print("bond:                ", bond)
    print("shots:               ", shots)

    print("graph_density:       ", graph_density)
    print("if_adsorting:        ", if_adsorting)

    #endregion

    
    
    #make data dir
    dir_0 = './data_iter_adap_sorting/'
    os.makedirs(dir_0, exist_ok=True)

    #region load qubo instances, get Hamiltonian and edge_coeff_dict
    instance_dir = "./instances/" + f"{int(graph_density * 100):03}" + '/N_' + str(n_qubits)
    with open(instance_dir + '/QUBO_' + str(n_qubits) + 'V_'+ 'r_'+ str(r) + '.gpickle', 'rb') as f:
        G = pickle.load(f)
    coeff_list = np.loadtxt(instance_dir + '/QUBO_coeff_' + str(n_qubits) + 'V_'+ 'r_'+ str(r) + '.txt')
    edge_list = list(G.edges)
    

    print('\n\n#############################################################################')
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits : n_qubits + len(edge_list)]

    #extrcact Hamiltonian, edge coefficients and eigen list
    H = Hamiltonian_qubo(n_qubits, edge_list, h_list, J_list)   #This is changes from the old version of the code!!
    t1 = time.time()
    eigen_list = H.to_matrix(sparse=True).diagonal().real
    eigen_ids = np.argsort(eigen_list)
    eigen_idvalue_dict = {int(id): float(eigen_list[id]) for id in eigen_ids}
    t2 = time.time()
    print("\ntime to get eigen_list: ", t2-t1)
    print("eigen_idvalue_dict", list(eigen_idvalue_dict.items())[:10])

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)}) #CHANGED COMPARED TO THE OLD CODE

    #endregion
    if backend_method == 'matrix_product_state':
        backendoptions = {'method':backend_method, 'matrix_product_state_max_bond_dimension': bond, 'shots': shots}
    else:
        backendoptions = {'method':backend_method, 'shots': shots}

    #region get best sorting way from cvar value
    t1 = time.time()
    expz_array = np.array([0]*n_qubits)
    #expz_array = np.array([1, 1, 1, -1])   ## z0, z1 , ... , zn-1
    circ_init = initial_state_ry(n_qubits, expz_array)

    edge_params_dict, params_init, exp_poss_dict = get_good_initial_params_measure_iterate(\
    n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_idvalue_dict, shots, backendoptions)     #HO SPOSTATO IL SALVADATI A DOPO
    cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)
    t2 = time.time()
    print("\ntime to run whole circuit: ", t2-t1)
    print('cvar of random order: ', cvar)
    print('exp_pos_dict', list(exp_poss_dict['l_1'].items())[:10])
    # print('expz_array: ', expz_array)

    ############################################################
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
                edge_params_dict, params_init, exp_poss_dict = get_good_initial_params_measure_iterate(\
                n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_idvalue_dict, shots, backendoptions)     #HO SPOSTATO IL SALVADATI A DOPO
                cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)

                print(f'\ncvar of Abs:{Abs}, invert:{invert} order: {cvar}')
                print('exp_pos_dict', list(exp_poss_dict['l_1'].items())[:10])
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
        n_qubits, tau, layer, circ_init, edge_coeff_dict, edge_list, eigen_idvalue_dict, shots, backendoptions)     #HO SPOSTATO IL SALVADATI A DOPO
        cvar, expz_array = get_expz(n_qubits, exp_poss_dict['l_1'] , alpha)
        #entropy = entanglement_entropy(state_all , range(round(n_qubits/2)), [2]*n_qubits, tol=1e-12)[0]

        steps_edge_params_dict['step_'+str(step)] = edge_params_dict
        steps_exp_poss_dict['step_'+str(step)] = exp_poss_dict
        steps_cvar_dict['step_'+str(step)] = cvar
        print('\nstep: ', step)
        print(f"cvar: {cvar:.6f}")
        print('exp_poss_dict', list(exp_poss_dict['l_1'].items())[:10])
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

    #endregion

if __name__ == "__main__":
    main()
    