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

from VQE_CVaR import VQE, partition_N

from warm_start_paramenters import *
from warm_start_parameters_lightcone import *

import itertools

import pickle


def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=6)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=0)
  
    parser.add_argument("--ansatz_type", help="ansatz type", required=False, type=str, default='structure_like_qubo_YZ_2')
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers, be careful of different defination of layer in different ansatz type", required=False, type=int, default=1)
    
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.3)
    parser.add_argument("--initialization", help="method for initial paramters, warm_start_measure, warm_start_analy, zeros, or random", required=False, type=str, default='warm_start_measure')

    parser.add_argument("--sorting", help="sorting or random order of coefficients", required=False, type=int, default=0)
    parser.add_argument("--absolute", help="sort coefficients in absolute value if true", required=False, type=int, default=0)
    parser.add_argument("--invert", help="sort coefficients in inverse order if true", required=False, type=int, default=0)

    args = parser.parse_args()

    n_qubits = args.N
    r = args.r
    alpha = args.alpha
    shots = args.shots
    ansatz_type = args.ansatz_type
    layer = args.layer
    tau = args.tau
    initialization = args.initialization
    sorting = args.sorting
    absolute = args.absolute 
    invert = args.invert

    # print(sorting, absolute, invert)
    
    if shots == 0:# exact simulation
        shots = None
        approximation = True
    else:#simulation with finite shots
        approximation = False
    
    if sorting == 0:
        sorting = False
    else:
        sorting = True

    if absolute == 0:
        absolute = False
    else:
        absolute = True

    if invert == 0:
        invert = False
    else:
        invert = True


    # optimizer = 'COBYLA'

    print('\nN: {}, \nr: {}, \nalpha: {}, \nshots: {}, \nansatz: {}, \nlayer: {}, \ntau: {}, \ninitialization: {}, \nsorting: {}, \nabsolute: {}, \ninvert: {}'\
        .format(n_qubits, r, alpha, shots, ansatz_type, layer, tau, initialization, sorting, absolute, invert))

    #region load qubo instances, get Hamiltonian and edge_coeff_dict
    instance_dir = '../instances/complete/N_' + str(n_qubits )
    with open(instance_dir + '/QUBO_' + str(n_qubits ) + 'V_comp_'+ '.gpickle', 'rb') as f:
        G = pickle.load(f)

    # edge_list = G.edges()
    pairs_all = list(itertools.chain.from_iterable(partition_N(n_qubits)))
    edges_columns = partition_N(n_qubits)  

    # print('\nedge columns', edges_columns)

    #lightcone_dict = find_light_cone(edges_columns)

    print('\n\n#############################################################################')

    coeff_list = np.loadtxt(instance_dir + '/QUBO_coeff_' + str(n_qubits) + 'V_comp_'+ 'r_'+ str(r)+ '.txt')
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits :]

    #extrcact Hamiltonian, edge coefficients and eigen list
    H = Hamiltonian_qubo(n_qubits, pairs_all, h_list, J_list)   #This is changes from the old version of the code!!
    eigen_list = H.to_spmatrix().diagonal().real

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    #Initialize dictionary for Pauli ZZ term with coefficient from from J_list
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(pairs_all, J_list)}) #CHANGED COMPARED TO THE OLD CODE

    ## order for two-qubit gate in circuit
    num_pairs = len(pairs_all)

    if sorting:
        # print(f'Coefficients are sorted according to absolut {absolute} and invert {invert}')    
        pairs_all, edges_columns = sorted_gates(coeff_list[n_qubits :], pairs_all, absolute, invert)
        #lightcone_dict = find_light_cone(edges_columns)
        # print('\nedge columns', edges_columns)
    else:
        print("Order of gates is random")

    # print('num pairs', num_pairs, 'layers:', layer, 'for new ansatz,', layer*(1 +2*num_pairs/n_qubits), 'for old ansatz')
    eff_layers = 0
 
    #region get initial parameters
    if (ansatz_type) == 'structure_like_qubo_YZ_2':

        #print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ITE')
        #exp_poss_dict_ite, state_ite = ITE(n_qubits, edge_coeff_dict, tau, eigen_list)
        # ITE_poss.append(list(exp_poss_dict.items())[0][1])
        # print('exp_poss_dict', list(exp_poss_dict.items())[0])

        steps_edge_params_dict = {}
        steps_exp_poss_dict = {}
        steps_cvar_dict = {}
        expz_array = np.array([0]*n_qubits)

        if initialization == 'warm_start_measure':
            
            step = 0

            circ_init = initial_state_ry(n_qubits, expz_array)
            edge_params_dict, params_init, exp_poss_dict, state_all = get_good_initial_params_measure_iterate(\
            n_qubits, tau, layer, circ_init, edge_coeff_dict, pairs_all, eigen_list, shots, approximation)     #HO SPOSTATO IL SALVADATI A DOPO
            poss, cvar, expz_array = get_expz(n_qubits, state_all, alpha, eigen_list)

            best_cvar = cvar 
            sorting = False
            best_abs = False
            best_inv = False

            # print(f"fidelity: {poss},  cvar: {cvar:.6f},  sorting: {sorting}")
            for Abs in [True, False]:
                for invert in [True, False]:

                    expz_array = np.array([0]*n_qubits)
                    pairs_all = list(itertools.chain.from_iterable(partition_N(n_qubits)))
                    pairs_all, edges_columns = sorted_gates(coeff_list[n_qubits :], pairs_all, Abs, invert)

                    circ_init = initial_state_ry(n_qubits, expz_array)
                    edge_params_dict, params_init, exp_poss_dict, state_all = get_good_initial_params_measure_iterate(\
                    n_qubits, tau, layer, circ_init, edge_coeff_dict, pairs_all, eigen_list, shots, approximation)     #HO SPOSTATO IL SALVADATI A DOPO
                    poss, cvar, expz_array = get_expz(n_qubits, state_all, alpha, eigen_list)
                    
                    # print(cvar, best_cvar, sorting)
                    # print("fidelity: ", poss, ",  cvar: ", cvar, ",  expz_array: ", expz_array)

                    if cvar < best_cvar:
                        best_cvar = cvar
                        sorting = True
                        best_abs = Abs
                        best_inv = invert
                    
                    #     print(cvar, best_cvar, sorting, best_abs, best_inv)
                    # print(f"fidelity: {poss},  cvar: {cvar:.6f},  expz_array: {expz_array}")

            if sorting:
                pairs_all = list(itertools.chain.from_iterable(partition_N(n_qubits)))
                pairs_all, edges_columns = sorted_gates(coeff_list[n_qubits :], pairs_all, best_abs, best_inv)
                # print('changed: ', sorting, best_abs, best_inv)

            expz_array = np.array([0]*n_qubits)

            for step in range(5):
                #print(f'# step: {step}')
                circ_init = initial_state_ry(n_qubits, expz_array)
                edge_params_dict, params_init, exp_poss_dict, state_all = get_good_initial_params_measure_iterate(\
                n_qubits, tau, layer, circ_init, edge_coeff_dict, pairs_all, eigen_list, shots, approximation)     #HO SPOSTATO IL SALVADATI A DOPO
                poss, cvar, expz_array = get_expz(n_qubits, state_all, alpha, eigen_list)

                steps_edge_params_dict['step_'+str(step)] = edge_params_dict
                steps_exp_poss_dict['step_'+str(step)] = exp_poss_dict
                steps_cvar_dict['step_'+str(step)] = cvar
                # print('exp_poss_dict', list(exp_poss_dict['l_1'].items())[:10])
                # print("fidelity: ", poss, ",  cvar: ", cvar, ",  expz_array: ", expz_array)
                # print(f"fidelity: {poss},  cvar: {cvar:.6f},  expz_array: {expz_array}")

            num_params = len(params_init)   

        else:
            raise ValueError('initialization method not found')
    #endregion

    # print('\ninitial parameters: ', params_init)
    #print('num params', num_params)

    #make data dir
    dir_0 = './data_iter_adap_sorting'
    os.makedirs(dir_0, exist_ok=True)
    dir_name =  dir_0 + '/ansatz_type_{}/shots_{}/num_variables_{:03d}/params_{}_layer_{}/alpha_{}/initial_{}/r_{}/'\
        .format(ansatz_type,  shots, n_qubits, num_params, layer*(1 + eff_layers), alpha, initialization, r)
    os.makedirs(dir_name, exist_ok=True)

    if initialization == 'warm_start_measure':
        
        gi_file_path = dir_name + 'tau_{}.pkl'.format(tau)

        save_data = {
                    'edge_order': pairs_all,
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

        
    if initialization == 'warm_start_measure_lightcone':
        
        if sorting:
            gi_file_path2 = dir_name + 'tau_{}_{}_{}.pkl'.format(tau, absolute, invert)
        else:
            gi_file_path2 = dir_name + 'tau_{}_random.pkl'.format(tau)

        save_data2 = {
                    'edge_order': edges_columns,
                    'edge_params_dict': edge_params_dict,
                    'params_list': params_init, ## just for good formula to run vqe
                    'exp_poss_dict': exp_poss_dict,
                    # 'exp_poss_dict_ite': exp_poss_dict_ite,
                    # 'ite_overlap' : overlap
                        }

        with open(gi_file_path2, 'wb') as f:
            pickle.dump(save_data2, f)

        print('Write sucessfully to ' + dir_name)

if __name__ == "__main__":
    main()