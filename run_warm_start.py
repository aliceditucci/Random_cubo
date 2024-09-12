#!/lustre/fs23/group/nic/yahuichai/package/miniconda3/envs/py37/bin/python3.7
# coding: utf-8

import sys
import os
import argparse
import numpy as np
import networkx as nx
import pickle

from Functions import *



def main():
  
    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=6)
    parser.add_argument("--r", help="random instances index", required=False, type=int, default=0)
    parser.add_argument("--ws_method", help="warm start method type: single_term, lightcone, or block", required=False, type=str, default='single_term')
    parser.add_argument("--numpara", help="number of parameters for each ZZ term, 2, 4, or 6", required=False, type=int, default=2)
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.2)

    args = parser.parse_args()
    N = args.N
    r = args.r
    ws_method = args.ws_method
    numpara = args.numpara
    tau = args.tau

 
    print('\nN: {}, \nr: {}, \nws_method: {}, \nnumpara:{}, \ntau: {}'\
        .format(N, r, ws_method, numpara, tau))
    #endregion

    ### folder to save result
    data_dir = './data/N_{}/r_{}/{}/'.format(N, r, ws_method)
    os.makedirs(data_dir, exist_ok=True)
    file_path = data_dir + 'numpara_{}_tau_{}.pkl'.format(numpara, tau)

    #region load qubo instances, get Hamiltonian and edge_coeff_dict
    if N < 16:  # for test
        G = nx.complete_graph(N)
        coeff_list = [1] * (N + N*(N-1)//2)
    else:
        instance_dir = '/lustre/fs23/group/nic/yahuichai/code/COP/code_qiskit_1/QUBO/instances/complete/N_' + str(N)
        G = nx.read_gpickle(instance_dir + '/QUBO_' + str(N) + 'V_comp_'+ '.gpickle')
        coeff_list = np.loadtxt(instance_dir + '/QUBO_coeff_' + str(N) + 'V_comp_'+ 'r_'+ str(r)+ '.txt')
    edge_list = G.edges()
    h_list = coeff_list[:N]
    J_list = coeff_list[N:]
    
    H = Hamiltonian_qubo(N, edge_list, h_list, J_list)
    eigen_list = H.to_spmatrix().diagonal().real

    # Initialize dictionary for single Pauli Z term with coefficient from h_list
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    #Initialize dictionary for Pauli ZZ term with coefficient from from J_list
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)})
    #endregion

    ## order for two-qubit gate in circuit
    edges_columns = partition_N(N)
    pairs_all = list(itertools.chain.from_iterable(partition_N(N)))


    #region warm start
    if ws_method == 'single_term':
        edge_params_dict, params_init, exp_poss_dict, state_all = warm_start_parameters_adaptlightcone(N, tau, numpara, 0.0, edge_coeff_dict, edges_columns, eigen_list)
    elif ws_method == 'lightcone':
        edge_params_dict, params_init, exp_poss_dict, state_all = warm_start_parameters_adaptlightcone(N, tau, numpara, 0.95, edge_coeff_dict, edges_columns, eigen_list)
    elif ws_method == 'block':
        edge_params_dict, params_init, exp_poss_dict, state_all = warm_start_parameters_block(N, tau, numpara, edge_coeff_dict, edges_columns, eigen_list)
    else:
        raise ValueError('warm start method not found')

    print('\nexp_poss_dict', exp_poss_dict)
    save_data = {
                'edge_order': pairs_all,
                'edge_params_dict': edge_params_dict,
                'exp_poss_dict': list(exp_poss_dict.items())[:1000]
    }

    with open(file_path, 'wb') as f:
        pickle.dump(save_data, f)

if __name__ == "__main__":
    main()




