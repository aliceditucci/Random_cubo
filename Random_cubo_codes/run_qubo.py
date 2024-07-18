import numpy as np
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=10000)
  
    parser.add_argument("--ansatz_type", help="ansatz type", required=False, type=str, default='structure_like_qubo_YZ_2')
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers, be careful of different defination of layer in different ansatz type", required=False, type=int, default=1)
    
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.2)
    parser.add_argument("--initialization", help="method for initial paramters, warm_start_measure, warm_start_analy, zeros, or random", required=False, type=str, default='warm_start_measure')

    args = parser.parse_args()

    n_qubits = args.N
    r = args.r
    alpha = args.alpha
    shots = args.shots
    ansatz_type = args.ansatz_type
    layer = args.layer
    tau = args.tau
    initialization = args.initialization

    if shots == 0:# exact simulation
        shots = None
        approximation = True
    else:#simulation with finite shots
        approximation = False

    optimizer = 'COBYLA'

    print('\nN: {}, \nr: {}, \nalpha: {}, \nshots: {}, \nansatz: {}, \nlayer: {}, \ntau: {}, \ninitialization: {}'\
        .format(n_qubits, r, alpha, shots, ansatz_type, layer, tau, initialization))

    #region load qubo instances, get Hamiltonian and edge_coeff_dict
    instance_dir = '../instances/complete/N_' + str(n_qubits )
    with open(instance_dir + '/QUBO_' + str(n_qubits ) + 'V_comp_'+ '.gpickle', 'rb') as f:
        G = pickle.load(f)
    edge_list = G.edges()
    coeff_list = np.loadtxt(instance_dir + '/QUBO_coeff_' + str(n_qubits) + 'V_comp_'+ 'r_'+ str(r)+ '.txt')
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits :]

    #extrcact Hamiltonian, edge coefficients and eigen list
    H = Hamiltonian_qubo(n_qubits, edge_list, h_list, J_list)
    eigen_list = H.to_spmatrix().diagonal().real
    #print('Hamiltonian', H)
    #print('eigen list', eigen_list[:13])

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    #Initialize dictionary for Pauli ZZ term with coefficient from from J_list
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)})
    print('edge_coeff_dict' , edge_coeff_dict)
    #endregion

    ## order for two-qubit gate in circuit
    pairs_all = list(itertools.chain.from_iterable(partition_N(n_qubits)))
    num_pairs = len(pairs_all)
    # print('num pairs', num_pairs)
    # print('pairs', pairs_all)

    edges_columns = partition_N(n_qubits)
    # print('edges_columns', edges_columns)
    lightcone_dict = find_light_cone(edges_columns)
    # print('lightcone_dict', lightcone_dict)



    # print('num pairs', num_pairs, 'layers:', layer, 'for new ansatz,', layer*(1 +2*num_pairs/n_qubits), 'for old ansatz')
    eff_layers = 0
 
    #region get initial parameters
    if (ansatz_type) == 'structure_like_qubo_YZ_2':
        if initialization == 'warm_start_measure':

            layers_edge_params_dict, params_init, layers_exp_poss_dict = get_good_initial_params_measure(\
            n_qubits, tau, layer, edge_coeff_dict, pairs_all, eigen_list, shots, approximation)     #HO SPOSTATO IL SALVADATI A DOPO
            print('\nwarm start fidelity', list(layers_exp_poss_dict['l_'+str(layer)].items())[0])
            # print(' params_init',  params_init)
            # print('layers_edge_params_dict', layers_edge_params_dict)
            # print('layers_exp_poss_dict', layers_exp_poss_dict)
        
        elif initialization == 'warm_start_measure_lightcone':

            edge_params_dict, params_init, exp_poss_dict = warm_start_parameters_lightcone(\
            n_qubits, tau, edge_coeff_dict, edges_columns, eigen_list, lightcone_dict)
            print('\nwarm start fidelity lightcone', list(exp_poss_dict.items())[0])
            # print(' params_init',  params_init)
            # print('layers_edge_params_dict', layers_edge_params_dict)
            # print('layers_exp_poss_dict', layers_exp_poss_dict)

        elif initialization == 'zeros':
            params_init = np.zeros((n_qubits + 2*num_pairs) * layer)
        elif initialization == 'random':
            params_init = np.random.uniform(-np.pi, np.pi, (n_qubits + 2*num_pairs) * layer)
        else:
            raise ValueError('initialization method not found')
        num_params = len(params_init)   

    elif (ansatz_type) == 'R_y': # for efficient su2 ansatz, 
        if layer == 3: 
                        #layer should be (1 + 2*len(edge_list)/N) times more than ansatz 'structure_like_qubo_YZ_2' to have the same number of parameters
            if initialization == 'zeros':
                params_init = np.zeros((n_qubits) * layer)
                #params_init = np.zeros(n_qubits * layer) 
            elif initialization == 'random':
                params_init = np.random.uniform(-0.1, 0.1, (n_qubits) * layer)
                #params_init = np.random.uniform(-np.pi, np.pi, n_qubits * layer)
            else:
                raise ValueError('initialization method not found')

            num_params = len(params_init)  

        else:
            #layer should be (1 + 2*len(edge_list)/N) times more than ansatz 'structure_like_qubo_YZ_2' to have the same number of parameters
            if initialization == 'zeros':
                params_init = np.zeros((n_qubits + 2*num_pairs) * layer)
                #params_init = np.zeros(n_qubits * layer) 
            elif initialization == 'random':
                params_init = np.random.uniform(-0.1, 0.1, (n_qubits + 2*num_pairs) * layer)
                #params_init = np.random.uniform(-np.pi, np.pi, n_qubits * layer)
            else:
                raise ValueError('initialization method not found')

            num_params = len(params_init)  
            eff_layers= num_params/n_qubits   
    #endregion

    # print('\ninitial parameters: ', params_init)
    print('num params', num_params)

    #make data dir
    dir_0 = './data'
    os.makedirs(dir_0, exist_ok=True)
    dir_name =  dir_0 + '/ansatz_type_{}/shots_{}/num_variables_{:03d}/params_{}_layer_{}/alpha_{}/initial_{}/r_{}/'\
        .format(ansatz_type,  shots, n_qubits, num_params, layer*(1 + eff_layers), alpha, initialization, r)
    os.makedirs(dir_name, exist_ok=True)

    if initialization == 'warm_start_measure':

        gi_file_path = dir_name + 'tau_{}.pkl'.format(tau)
        save_data = {
                    'edge_order': pairs_all,
                    'layers_edge_params_dict': layers_edge_params_dict,
                    'params_list': params_init, ## just for good formula to run vqe
                    'layers_exp_poss_dict': layers_exp_poss_dict
                        }
        
        with open(gi_file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print('Write sucessfully to ' + dir_name)

        
    if initialization == 'warm_start_measure_lightcone':

        gi_file_path2 = dir_name + 'tau_{}.pkl'.format(tau)
        save_data2 = {
                    'edge_order': edges_columns,
                    'edge_params_dict': edge_params_dict,
                    'params_list': params_init, ## just for good formula to run vqe
                    'exp_poss_dict': exp_poss_dict
                        }

        with open(gi_file_path2, 'wb') as f:
            pickle.dump(save_data2, f)

        print('Write sucessfully to ' + dir_name)



    
#     vqe = VQE(Hamiltonian=H, n_qubits = n_qubits) 
#     print('minimal exp: ', vqe.exp_min)
#     print('ground_id_list: ', vqe.ground_id_list)
#     print('n_qubits: ', vqe.n_qubits)
#     vqe.edge_list = pairs_all
#     vqe.alpha = alpha
#     vqe.ansatz_type = ansatz_type 
#     vqe.shots = shots

#     vqe.cvar_eval = []
#     vqe.r_eval = []
#     vqe.poss_eval = []

#     final = minimize(vqe.CVaR_expectation,
#                         params_init,
#                         jac=False,
#                         bounds=None,
#                         method=optimizer,
#                         callback=None,
# #                               tol=1e-5,
# #                                       options={'maxiter': 50 * n_qubits})  
#                         options={'maxiter': len(params_init)*5}) 

#     save_list = np.array([vqe.cvar_eval, vqe.r_eval, vqe.poss_eval])
#     save_dir_name = dir_name 
#     os.makedirs(save_dir_name, exist_ok=True)
#     np.savetxt(save_dir_name + '/result.txt', save_list.T)
#     np.savetxt(save_dir_name + '/param.txt', final.x)
#         ### check the result
#     print('final fidelity: ', max(vqe.poss_eval))
#     print('cvar:', vqe.cvar_eval[(vqe.poss_eval).index(max(vqe.poss_eval))])
#     print('final', final.fun)
#     print('Write sucessfully to ' + save_dir_name)


if __name__ == "__main__":
    main()