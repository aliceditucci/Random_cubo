#!/lustre/fs23/group/nic/yahuichai/package/miniconda3/envs/py37/bin/python3.7
# coding: utf-8

import sys
import os
import numpy as np
from math import cos, sin, cosh, sinh, atan, exp, pi
import copy
import pickle

from qiskit import *
from qiskit.quantum_info import Pauli, SparsePauliOp
#from qiskit_aer.primitives import Estimator

import scipy.sparse.linalg
from scipy.sparse.linalg import expm
from scipy.optimize import minimize
#from qiskit.opflow import PauliSumOp
import matplotlib.pyplot as plt

import networkx as nx
import itertools
from itertools import combinations, permutations

def ITE(N, qi, coeff, tau, vec):
    """
    Caculating the statevector after imiginary time evolution, 
    i.e, e^{- tau * coeff * Z_i} * vec, or e^{- tau * coeff * Z_jZ_i} * vec
    
    Args:
        N: int, number of qubits
        qi: qubit index list, [i] for single-qubit term, [i,j] for two-qubit term
        coeff: float, coeffieicnt of the terms acting on qubit i or (i, j)
        tau: float, time for imaginary time evolution
        vec:numpy.array,  the statevector of current state
    Return:
        vec_tau / norm: numpy.array, normalized statevector after imiginary time evolution
    """
    
    if len(qi) == 1:  ## single Z term
        i = qi[0]
        Z_op = Pauli('I'*(N-1-i) + 'Z' + 'I'*i)
    elif len(qi) == 2: ## two body term Z_jZ_i
        i = min(qi)
        j = max(qi)
        Z_op = Pauli('I'*(N-1-j) + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*i)
        
    Z_op_diag = Z_op.to_matrix(sparse=True).diagonal().real
    vec_z = np.multiply(Z_op_diag, vec)  ## multiplying each element
    vec_tau = np.cosh(coeff * tau) * vec - np.sinh(coeff * tau) * vec_z
    
    norm = np.linalg.norm(vec_tau)
    
    return vec_tau / norm

def quant_circ_update(N, circ, qi, params_in):
    """get the updated quantum circuit by adding exp(-i param * pauli_op) in quantum circuit
    know opstrs by the length of params_in
    Args:
        N: number of qubits
        circ: QauntumCircuit
        qi: list of related qubit index
        params_in: list of parameters
    Return:
        circ: updated circuit
    """
    
    if len(qi) == 1:    ## only ry for e^{-tau * h_i * Z_i}
        circ.ry(params_in[0], qi[0])
    else:               ## quantum gate inspired by e^{ -tau * J_ij * Z_iZ_j}
        i = min(qi)
        j = max(qi)
        
        if len(params_in) == 6:  ## IY, ZY, YZ, YI, XY, YX
            params = params_in
        elif len(params_in) == 4:  ## IY, ZY, YZ, YI
            params = list( copy.deepcopy(params_in) )
            params.extend([0, 0])
        elif len(params_in) == 2:
            params = np.zeros(6)  ## ZY, YZ
            params[2] = params_in[0]
            params[3] = params_in[1]
            
        ### exp{-i/2 params[0] * YI}
        if len(params_in) > 2:   
            circ.ry(params[0], i)
        
        ### exp{-i/2 ( params[2]*ZiYj + params[3]*YiZj )}
        circ.rx(-np.pi/2, i)
        circ.rz(-np.pi/2, j)

        circ.cx(i, j)
        circ.ry(params[2], i)
        circ.rz(-params[3], j)
        circ.cx(i, j)

        circ.rx(np.pi/2, i)
        circ.rz(np.pi/2, j)

        if len(params_in) > 2:
            ### exp{-i/2 params[1] * IY}  
            circ.ry(params[1], j)
            
            ### exp{-i/2 ( params[4]*XiYj + params[5]*YiXj )}
            circ.rx(-np.pi/2, i)
            circ.rz(-np.pi/2, j)
            circ.rx(-np.pi/2, j)

            circ.cx(i,j)
            circ.rx(params[4], i)
            circ.rz(-params[5], j)
            circ.cx(i,j)

            circ.rx(np.pi/2, i)
            circ.rx(np.pi/2, j)
            circ.rz(np.pi/2, j)
    
    return circ

def optimal_params(N, qi, coeff, tau, circ, opstrs):
    """get the warm start parameters corresboding to qubits in list qi,
    by maxmizing the overlap between statevectors from ITE and quantum circuit.
    This is a function to double check with the warm start by measurement-based approach
    Args:
        N: number of qubits
        qi: qubit index list
        circ: current circuit to be updated
    Return:
        final.x: warm start parameters for qubits in qi
        final.fun: minimal value of 1-overlap
    """
    
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(circ).result()
    vec = np.array( result.get_statevector()).real

    vec_tau = ITE(N, qi, coeff, tau, vec) ## get the statevector after imaginary time evolution
    
    ## define the cost function in minimization: minimizing the 1-overlap
    def cost(params):
        circ_0 = quant_circ_update(N, circ.copy(), qi, params)  ## circ.copy(), deep copy
        result = backend.run(circ_0).result()
        vec_q = np.array( result.get_statevector()).real
        
        amp = np.dot(vec_tau, vec_q)
        inf = 1 - abs(amp)**2
        return inf

    params_init = [0] * len(opstrs)
    final = minimize(cost,
                      params_init,
                      jac=False,
                      bounds=None,
                      method='L-BFGS-B',
                      callback=None,
    #                               tol=1e-5,
                      options={'maxiter': 1000})
    
    return final.x, final.fun


# def Hamiltonian_qubo(N, edge_list, h_list, J_list):
#     """Hamiltonian defined by a N vertex graph with connected edge in edge_list
#     Args:
#         N: number of qubits
#         edge_list: list of edges(qubit index pairs)
#         h_list: coefficients of single Pauli Z term
#         J_list: coefficients of ZZ term
#     Return:
#         H: PauliSumOp, Hamiltonian

#     """
#     pauli_list = []
#     for i in range(N):
#         pauli_str = (N-i-1)*'I' + 'Z' + i*'I'
#         op = Pauli(pauli_str)
#         pauli_list.append((op.to_label(), h_list[i]))
        
#     for k, (i, j) in enumerate(edge_list):
#         x_p = np.zeros(N, dtype = bool)
#         z_p = np.zeros(N, dtype = bool)
#         z_p[i] = True
#         z_p[j] = True
#         op = Pauli((z_p, x_p))
#         pauli_list.append((op.to_label(), J_list[k]))
        
#     H = PauliSumOp.from_list(pauli_list)
    
#     return H


def partition(N:int):
    '''do the partition of a complete graph with N vertex, to find the optimal orders for edges to run circuit in parallel
    Args:
        N: number of qubits
    Return:
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
    '''
    indexs = range(N)
    pairs_all = []  

    ## swap indexes of even layer [0,1,2,3,4] -> [1,0,3,2,4]
    swap_even = [i + pow(-1, i) for i in range(N - (N%2))]  
    if (N%2) == 1:
        swap_even.append(N-1)
    ## swap indexes of even layer [0,1,2,3,4] -> [0,2,1,4,3]
    swap_odd = [0]
    swap_odd.extend([i + pow(-1, i+1) for i in range(1,N-(N+1)%2)])
    if (N%2) == 0:
        swap_odd.append(N-1)
    
    ## qubit pairs need to be implemented in layer 0
    pairs_even = [(i, i+1) for i in range(0, N-1, 2)]  
    pairs_all.append(pairs_even)
    indexs = np.array(indexs)[swap_even]   ### indexs after swap even

    for i in range(1, N):
        if (i%2)==1: ## odd layer
            pair_odd = [(indexs[i], indexs[i+1]) for i in range(1, N-1, 2)]
            pairs_all.append(pair_odd)
            indexs = np.array(indexs)[swap_odd]   ### indexs after swap odd

        elif (i%2)==0: ## even layer
            pair_even = [(indexs[i], indexs[i+1]) for i in range(0, N-1, 2)]
            pairs_all.append(pair_even)
            indexs = np.array(indexs)[swap_even]   ### indexs after swap even

    return pairs_all

def cost_mimic_1op(para:list, *args:tuple):
    """Caculate the cost function to find a good initial parameters for 1-qubit gates"""
    exp_dict = args[0] ## dict, {pauli string: expectation of the corresbonding pauli op}
    tauc = args[1]  ## tau * coeff

    theta = para[0]
    cost = 0  ## overlap between ITE and mimic QC
    cost += cos(theta/2) * (cosh(tauc) - sinh(tauc)*exp_dict['Z'])
    cost += -1j * sin(theta/2) * (cosh(tauc)*exp_dict['Y'] + 1j*sinh(tauc)*exp_dict['X'])

    return -abs(cost)

def get_initial_para_1op_Y(N:int, qi:list, coeff:float, tau:float, circ:QuantumCircuit, shots:int, approximation:bool):
    """Get the good initial parameters for operations in qubit [i] by mimic the ITE e^{-tau*coeff*Zi}
    Args:
        N: number of qubits
        qi: list of corresboding qubit index, only one element in this case
        coeff: coefficient of qi term in Hamiltonian
        tau: time step for imaginary time evolution
        circ: current quantum circuit
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                             Otherwise, it calculates expectation values with sampling.
        approximation:
    Returns:
        init_params: warm start parameters for gates corresboding qubit in qi
    """
    i = qi[0] # qubit index
    tauc = tau * coeff

    estimator = Estimator(approximation =  approximation, run_options={"shots": shots})

    op_dict = {}
    op_dict['Z'] = SparsePauliOp.from_sparse_list([('Z', [i], 1)], N)
    op_dict['Y'] = SparsePauliOp.from_sparse_list([('Y', [i], 1)], N)
    op_dict['X'] = SparsePauliOp.from_sparse_list([('X', [i], 1)], N)

    ## measure the expectations of operators in op_dict
    exp_dict = {}
    for op_str in op_dict.keys():
        op = op_dict[op_str]
        exp = estimator.run(circ, op).result().values[0]
        exp_dict[op_str] = exp
        #print(estimator.run(circ, op).result())

    para_init = [0]
    final = minimize(cost_mimic_1op,
                      para_init,
                      args = (exp_dict, tauc),
                      jac=False,
                      bounds=None,
                      method='L-BFGS-B',
                      callback=None,
    #                               tol=1e-5,
                      options={'maxiter': 10000})
    return final.x
    
def cost_mimic_2op(para:list, *args:tuple):
    """Caculate the cost function to find a good initial parameters for 2-qubit gates, only for YZ_2 ansatz"""
    exp_dict = args[0] ## dict, {pauli string: expectation of the corresbonding pauli op}
    tauc = args[1]  ## tauc = tau * coeff

    theta0 = para[0]
    theta1 = para[1]
    cost = 0  ## overlap between ITE and mimic QC
    cost += cos(theta0/2) * cos(theta1/2) * (cosh(tauc) - sinh(tauc)*exp_dict['ZZ'])
    cost += -1j * cos(theta0/2) * sin(theta1/2) * (cosh(tauc)*exp_dict['ZY'] + 1j*sinh(tauc)*exp_dict['Xi'])
    cost += -1j * sin(theta0/2) * cos(theta1/2) * (cosh(tauc)*exp_dict['YZ'] + 1j*sinh(tauc)*exp_dict['Xj'])
    cost += -1 * sin(theta0/2) * sin(theta1/2) * (cosh(tauc)*exp_dict['XX'] + sinh(tauc)*exp_dict['YY'])

    return -abs(cost)

def get_initial_para_2op_YZ(N:int, qi:list, coeff:float, tau:float, circ:QuantumCircuit, shots:int, approximation:bool):
    """Get the good initial parameters for gates acting on qubits [i,j] by mimic the ITE e^{-tau*coeff*ZiZj}.
       Only for structure_like_qubo_YZ_2 ansatz 
    Args:
        N: number of qubits
        qi: list of qubits index, should be two elements [i,j]
        coeff: coefficient of ZiZj term in Hamiltonian
        tau: time step for imaginary time evolution
        circ: current quantum circuit
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                            Otherwise, it calculates expectation values with sampling.
        approximation:
    Returns:
        init_params: warm start parameters corresbonding to qubits i and j
    """
    i = min(qi)
    j = max(qi)
    tauc = tau * coeff

    estimator = Estimator(approximation =  approximation, run_options={"shots": shots})

    op_dict = {}
    op_dict['ZZ'] = SparsePauliOp.from_sparse_list([('ZZ', [j, i], 1)], N)
    op_dict['ZY'] = SparsePauliOp.from_sparse_list([('ZY', [j, i], 1)], N)
    op_dict['YZ'] = SparsePauliOp.from_sparse_list([('YZ', [j, i], 1)], N)
    op_dict['XX'] = SparsePauliOp.from_sparse_list([('XX', [j, i], 1)], N)
    op_dict['YY'] = SparsePauliOp.from_sparse_list([('YY', [j, i], 1)], N)

    op_dict['Xi'] = SparsePauliOp.from_sparse_list([('X', [i], 1)], N)
    op_dict['Xj'] = SparsePauliOp.from_sparse_list([('X', [j], 1)], N)

    exp_dict = {}
    for op_str in op_dict.keys():
        op = op_dict[op_str]
        exp = estimator.run(circ, op).result().values[0]
        exp_dict[op_str] = exp
        #print(estimator.run(circ, op).result())

    para_init = [0, 0]
    final = minimize(cost_mimic_2op,
                      para_init,
                      args = (exp_dict, tauc),
                      jac=False,
                      bounds=None,
                      method='L-BFGS-B',
                      callback=None,
                      options={'maxiter': 10000})
    return final.x


def get_good_initial_params_measure(N:int, tau:float, layer:int, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_list:list, shots:int, approximation:bool, file_path:str):
    """get the warm start parameters by measurement-based approach
    Args:
        N: number of qubits
        tau: time step for imaginary time evolution
        layer: number of layers in the ansatz
        edge_coeff_dict: dict, {edge: coeff}, coefficients of edges (or vertexes, i as edge (i,)) in the graph
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
        eigen_list: list of eigenvalues of Hamiltonian
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                            Otherwise, it calculates expectation values with sampling.
        approximation:
        file_path: str, path to save the warm start parameters
    Return:
        layers_edge_params_dict: dict, {layer: {edge: params}}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """
    
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])

    layers_edge_params_dict = {}  ## save the warm start parameters for each edge in the graph from l=1 to maximal layer
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    layers_exp_poss_dict = {} ## save probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    for l in range(1, layer+1):
        edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
        exp_poss_dict = {}  ### record the {exp:poss} information after the excution of l'th layer

        # Z term
        for i in range(N):
            para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0]
            edge_params_dict[(i,)] = para
            params_list.append(para)
            circ.ry(para, i)

        # ZZ term
        for edge in pairs_all:
            para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, shots, approximation)
            edge_params_dict[edge] = para
            params_list.extend(para)
            circ =  quant_circ_update(N, circ, edge, para)
        
        layers_edge_params_dict['l_'+str(l)] = edge_params_dict

        # run the l layer circuit
        backend = Aer.get_backend('statevector_simulator')
        result = backend.run(circ).result()
        vec_final = np.array( result.get_statevector() ).real

        for id in eigens_ids:
            eigen = eigen_list[id]
            poss = abs(vec_final[id])**2
            exp_poss_dict[eigen] = poss


        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict


    save_data = {
                'edge_order': pairs_all,
                'layers_edge_params_dict': layers_edge_params_dict,
                'params_list': params_list, ## just for good formula to run vqe
                'layers_exp_poss_dict': layers_exp_poss_dict
    }

    with open(file_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    return layers_edge_params_dict, params_list, layers_exp_poss_dict

def get_good_initial_params_analy(N:int, tau:float, layer:int, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_list:list, file_path:str):
    """get the warm start parameters by analitical approach, it should be finished by analytical calculation,
    here the exact simulation is used to get the same result using the functions above
    Args:
        N: number of qubits
        tau: time step for imaginary time evolution
        layer: number of layers in the ansatz
        edge_coeff_dict: dict, {edge: coeff}, coefficients of edges (or vertexes, i as edge (i,)) in the graph
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
        eigen_list: list of eigenvalues of Hamiltonian
        file_path: str, path to save the warm start parameters
    Return:
        edge_params_dict: dict, {edge: params}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """
    ## exact simulation in analytical approach
    shots = None
    approximation = True
    
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])
    ## save the warm start parameters for each edge in the graph, and the parameters are the same for each layer
    edge_params_dict = {}  
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    # Z term
    for i in range(N):
        para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0]
        edge_params_dict[(i,)] = para
        params_list.append(para)
    # ZZ term
    for edge in pairs_all:
        para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, shots, approximation)
        edge_params_dict[edge] = para
        params_list.extend(para)

    ## run circuit with warm parameters, save probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    layers_exp_poss_dict = {} 
    for l in range(1, layer+1):
        exp_poss_dict = {}  # record the {exp:poss} information after the excution of l'th layer
        
        for i in range(N):
            para = edge_params_dict[(i,)]
            circ.ry(para, i)
        for edge in pairs_all:
            para = edge_params_dict[edge]
            circ =  quant_circ_update(N, circ, edge, para)

        # run the l layer circuit
        backend = Aer.get_backend('statevector_simulator')
        result = backend.run(circ).result()
        vec_final = np.array( result.get_statevector() ).real

        for id in eigens_ids:
            eigen = eigen_list[id]
            poss = abs(vec_final[id])**2
            exp_poss_dict[eigen] = poss

        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict

    save_data = {
                'edge_order': pairs_all,
                'edge_params_dict': edge_params_dict,
                'params_list': params_list, ## just for good formula to run vqe
                'layers_exp_poss_dict': layers_exp_poss_dict
    }

    with open(file_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    return edge_params_dict, params_list*layer, layers_exp_poss_dict