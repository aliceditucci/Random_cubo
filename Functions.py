from qiskit import *
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp

from math import cos, sin, cosh, sinh, atan, exp, pi
from scipy.optimize import minimize
import sys
import copy
import pickle
import itertools
import numpy as np
import networkx as nx

def Hamiltonian_qubo(N, edge_list, h_list, J_list):
    """Hamiltonian defined by a N vertex graph with connected edge in edge_list
    Args:
        N: number of qubits
        edge_list: list of edges(qubit index pairs)
        h_list: coefficients of single Pauli Z term
        J_list: coefficients of ZZ term
    Return:
        H: PauliSumOp, Hamiltonian

    """
    pauli_list = []
    for i in range(N):
        pauli_str = (N-i-1)*'I' + 'Z' + i*'I'
        op = Pauli(pauli_str)
        pauli_list.append((op.to_label(), h_list[i]))
        
    for k, (i, j) in enumerate(edge_list):
        x_p = np.zeros(N, dtype = bool)
        z_p = np.zeros(N, dtype = bool)
        z_p[i] = True
        z_p[j] = True
        op = Pauli((z_p, x_p))
        pauli_list.append((op.to_label(), J_list[k]))
        
    H = PauliSumOp.from_list(pauli_list)
    
    return H

def partition_N(N:int):
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

def find_light_cone(edges_columns):
    """find the lightcone of all qubit pairs in previous one layer"""
    lightcone_dict = {}
    for index, pair_list in enumerate(edges_columns):
        for pair in pair_list:
            qi, qj = pair
            relevent_pairs = []  ##  qubit pairs in the previous layer that in the lightcone of the current pair
            if index > 0:
                for pair_layerm1 in edges_columns[index-1]: ## qubit pairs in the previous layer
                    if (qi in pair_layerm1) or (qj in pair_layerm1):
                        relevent_pairs.append(pair_layerm1)
            lightcone_dict[pair] = relevent_pairs
    return lightcone_dict

def find_light_cone2(edges_columns, pair_now, lnow, L):
    """find the pairs in lightcone including previous L layers for a specific qubit pair at layer lnow 
    """
    lightcone = []
    qubit_list = list(pair_now) ## qubit index included, which is used to find the pair in lightcone in previous layer
    if L == 0:
        lightcone.append(pair_now)
    else:
        for index in range(lnow-1, max(0, lnow-L)-1, -1):
            pair_list = edges_columns[index]
            for pair in pair_list:
                if pair[0] in qubit_list or pair[1] in qubit_list:
                    lightcone.append(pair)
                    qubit_list.extend(pair)
        lightcone.append(pair_now)
    return lightcone

def find_block(N : int, lnow : int, edges_columns : list):
    '''find the block of qubit pairs in the layer lnow and lnow-1'''
    if lnow%2 != 1:
        print('something is wrong with the layer index')
        sys.exit()

    block_list = []
    for ei in range(0, len(edges_columns[lnow]), 3):
        edge_list = []
        edge_list.append(edges_columns[lnow-1][ei])
        edge_list.append(edges_columns[lnow-1][ei+1])
        edge_list.append(edges_columns[lnow][ei])
        block_list.append(edge_list)

    for ei in range(1, len(edges_columns[lnow]), 3):
        edge_list = []
        
        edge_list.append(edges_columns[lnow-1][ei+1])
        if (N%3 == 2) and (ei == len(edges_columns[lnow])-2):
            edge_list.append(edges_columns[lnow-1][-1])

        edge_list.append(edges_columns[lnow][ei])
        if ei+1 < len(edges_columns[lnow]):
            edge_list.append(edges_columns[lnow][ei+1])
        
        block_list.append(edge_list)

    return block_list

def ITE(N:int, edge_coeff_dict:dict, tau:float, eigen_list:np.array):
    
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????
        
    edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
    exp_poss_dict = {}   ## save probalities of eigenvalues using warm start circuit

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])

    ## Z term 
    for i in range(N):
        #para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0] #use this to extimate para from min expectation value
        tauc = tau * edge_coeff_dict[(i,)] 
        para = 2*atan( -exp(-2*tauc) ) + pi/2 #use this to use analytic formula (only valid for 1 layer)
        edge_params_dict[(i,)] = para
        circ.ry(para, i)
            
    ## ZZ term 
    state = Statevector(circ)
    for edge, coeff in edge_coeff_dict.items():
        if len(edge) == 2:
            tauc = tau * coeff
            state = circuit_update_zz(edge, state, tauc, N)

    state = np.array(state)
    for id in eigens_ids:
        eigen = eigen_list[id]
        poss = abs(state[id])**2
        # print('eigen', eigen, 'poss', poss)
        exp_poss_dict[eigen] = poss
    return exp_poss_dict, state

def circuit_update_zz(Edge,State, Tauc, N):
    '''update the state vector by applying exp(-tauc * ZZ) term to the current state vector'''
    i = min(Edge)
    j = max(Edge)

    zzop = SparsePauliOp.from_sparse_list([('ZZ', [j, i], -sinh(Tauc) )], N)
    zzop += SparsePauliOp.from_list([("I"*N, cosh(Tauc))])
    
    State = State.evolve(zzop)

    # Normalize the state vector manually
    norm = np.linalg.norm(State.data)
    State = State / norm

    # Print the normalized state vector
    # print("Normalized state:", State)

    return State

def circuit_update_theta(Edge, State, paras, N):
    
    j = min(Edge)    #I SWAPPED i AND j TO MATCH WITH YAHUI CIRCUIT
    i = max(Edge)

    zyop = SparsePauliOp.from_sparse_list([('ZY', [j, i], -1j*sin(paras[0]/2))], N)
    zyop += SparsePauliOp.from_list([("I"*N, cos(paras[0]/2))])

    yzop = SparsePauliOp.from_sparse_list([('YZ', [j, i], -1j*sin(paras[1]/2))], N)
    yzop += SparsePauliOp.from_list([("I"*N, cos(paras[1]/2))])

    op = zyop.compose(yzop)

    if len(paras) == 4:
        iyop = SparsePauliOp.from_sparse_list([('IY', [j, i], -1j*sin(paras[2]/2))], N)
        iyop += SparsePauliOp.from_list([("I"*N, cos(paras[2]/2))])
        op = op.compose(iyop)

        yiop = SparsePauliOp.from_sparse_list([('YI', [j, i], -1j*sin(paras[3]/2))], N)
        yiop += SparsePauliOp.from_list([("I"*N, cos(paras[3]/2))])
        op = op.compose(yiop)
    elif len(paras) == 6:
        xyop = SparsePauliOp.from_sparse_list([('XY', [j, i], -1j*sin(paras[4]/2))], N)
        xyop += SparsePauliOp.from_list([("I"*N, cos(paras[4]/2))])
        op = op.compose(xyop)

        yxop = SparsePauliOp.from_sparse_list([('YX', [j, i], -1j*sin(paras[5]/2))], N)
        yxop += SparsePauliOp.from_list([("I"*N, cos(paras[5]/2))])
        op = op.compose(yxop)
    
    State = State.evolve(op)  ## e^{-i op * t}

    # print('states2', State)

    return State

def square_modulus_cost_light_cone(Paras : list, *args):

    # para_init = np.zeros((len(lightcone_dict[edge]) + 1, 2))

    Edge_list = args[0]
    State = args[1]
    Tauc = args[2]
    Updated_state = args[3]
    N = args[4]

    State_zz = circuit_update_zz(Edge_list[-1], Updated_state, Tauc, N)
    
    if len(Paras) == 2*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[2*index], Paras[2*index + 1]]
            State = circuit_update_theta(edge, State, Parameters, N)
    elif len(Paras) == 4*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[4*index], Paras[4*index + 1], Paras[4*index + 2], Paras[4*index + 3]]
            State = circuit_update_theta(edge, State, Parameters, N)
    elif len(Paras) == 6*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[6*index], Paras[6*index + 1], Paras[6*index + 2], Paras[6*index + 3], Paras[6*index + 4], Paras[6*index + 5]]
            State = circuit_update_theta(edge, State, Parameters, N)

    # Compute the scalar product (inner product) between the two state vectors
    inner_product = State_zz.inner(State)

    # Compute the square modulus of the inner product
    square_modulus = abs(inner_product)**2

    return - square_modulus

def square_modulus_cost_light_cone_block(Paras : list, *args):

    # para_init = np.zeros((len(lightcone_dict[edge]) + 1, 2))

    Edge_list = args[0]
    State = args[1]
    Tauc_list = args[2]
    N = args[3]

    State_zz = copy.deepcopy(State)
    for ei, edge in enumerate(Edge_list):
        State_zz = circuit_update_zz(edge, State_zz, Tauc_list[ei], N)
    
    if len(Paras) == 2*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[2*index], Paras[2*index + 1]]
            State = circuit_update_theta(edge, State, Parameters, N)
    elif len(Paras) == 4*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[4*index], Paras[4*index + 1], Paras[4*index + 2], Paras[4*index + 3]]
            State = circuit_update_theta(edge, State, Parameters, N)
    elif len(Paras) == 6*len(Edge_list):
        for index, edge in enumerate(Edge_list):
            Parameters = [Paras[6*index], Paras[6*index + 1], Paras[6*index + 2], Paras[6*index + 3], Paras[6*index + 4], Paras[6*index + 5]]
            State = circuit_update_theta(edge, State, Parameters, N)

    # Compute the scalar product (inner product) between the two state vectors
    inner_product = State_zz.inner(State)

    # Compute the square modulus of the inner product
    square_modulus = abs(inner_product)**2

    return - square_modulus

def warm_start_parameters_adaptlightcone(N : int, tau:float, numpara:int, ampth:float, edge_coeff_dict : dict,
    edges_columns :list,  eigen_list:list, Lmax = 1, ifprint = False):
    '''' warm start the parameters for the circuit using the lightcone adaptivly
    Args:
        N: number of qubits
        tau: imaginary time
        numpara: number of parameters for each edge, 2, 4, or 6
        ampth: amplitude threshold, if the amplitude between optimized state and ITE state is larger than this value, the optimization stops; \
               otherwise continue with a larger lightcone including more previous layers
        edge_coeff_dict: coefficients for each edge
        edges_columns: list of edges in each layer
        eigen_list: list of eigenvalues
        Lmax: maximum number of previous layers included in the lightcone
        ifprint: print the optimization process
    Return:
        edge_params_dict: parameters for each edge
        params_list: list of parameters
        exp_poss_dict: probabilities of eigenvalues using warm start circuit
        state: final state vector 
    '''
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????
    
    edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
    exp_poss_dict = {}   ## save probalities of eigenvalues using warm start circuit
    edge_coeff_dict_ite = {} ## record the edges has been executed, to get the ite state for each step

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])

    ## Z term 
    for i in range(N):
        tauc = tau * edge_coeff_dict[(i,)] 
        para = 2*atan( -exp(-2*tauc) ) + pi/2 #use this to use analytic formula (only valid for 1 layer)
        edge_params_dict[(i,)] = para
        circ.ry(para, i)

        edge_coeff_dict_ite[(i,)] = edge_coeff_dict[(i,)]
   
    
    ## ZZ term 
    state = Statevector(circ)
    for column_index, column in enumerate(edges_columns):
        if ifprint:
            print('\n### column index', column_index, 'column', column)
        if column_index == 0: 
            first_column_state = copy.deepcopy(state)
            for edge in column:
                edge_coeff_dict_ite[edge] = edge_coeff_dict[edge]

                tauc = tau * edge_coeff_dict[edge]

                para_init = [0]* numpara
                final = minimize(square_modulus_cost_light_cone,
                        para_init,
                        args = ([edge], first_column_state, tauc, first_column_state, N),
                        jac=False,
                        bounds=None,
                        method='SLSQP',
                        callback=None,
                        options={'maxiter': 1000})
                para = final.x
                if ifprint:
                    print('\nedge: ', edge,  ',  coeff: ', edge_coeff_dict[edge],  ',     final square modulus: ', final.fun)

                edge_params_dict[edge] = para

                #region check the overlap between the ite state and the optimized state
                state_opt = copy.deepcopy(state)
                for old_edge in edge_coeff_dict_ite.keys():
                    if len(old_edge) == 2:
                        state_opt = circuit_update_theta(old_edge, state_opt, edge_params_dict[old_edge], N)
                state_opt = np.array(state_opt)
                if ifprint and (N <= 12):
                    _, state_ite = ITE(N, edge_coeff_dict_ite, tau, eigen_list)
                    print('overlap between ite and optimized state: ', state_ite.T.conj() @ state_opt)
                
        else:
            for ei, edge in enumerate(column):  
                if ifprint:
                    print('\nedge:', edge, ',  coeff:', edge_coeff_dict[edge])
                edge_coeff_dict_ite[edge] = edge_coeff_dict[edge]
                ## optimize the gate in the lightcone including previous L layers
                for L in range(0, min(Lmax, column_index) +1):
                    lightcone = find_light_cone2(edges_columns, edge, column_index, L)  ## current edge is included in the lightcone, which is the last element in the list
                    if ifprint:
                        print('~~L:', L, 'lightcone:', lightcone)

                    updated_state = copy.deepcopy(state)
                    optimized_state = copy.deepcopy(state)

                    para_init = []
                    edge_list = []  ## to save the edges in the lightcone that will be optimized
                    for ci in range(column_index-L):## previours layer before lightcone
                        for old_edge in edges_columns[ci]:
                            updated_state = circuit_update_theta(old_edge, updated_state, edge_params_dict[old_edge], N)  
                            optimized_state = circuit_update_theta(old_edge, optimized_state, edge_params_dict[old_edge], N)
                    for ci in range(column_index-L, column_index):## layer in lightcone      
                        for old_edge in edges_columns[ci]:
                            if old_edge in lightcone:
                                edge_list.append(old_edge)
                                para_init.extend(list(edge_params_dict[old_edge]))
                                updated_state = circuit_update_theta(old_edge, updated_state, edge_params_dict[old_edge], N) 
                    
                    tauc = tau * edge_coeff_dict[edge]

                    edge_list = edge_list + [edge]
                    para_init = para_init + [0]*numpara
                    #print('optimize edge list', edge_list, ',  para_init: ', para_init)
                    final = minimize(square_modulus_cost_light_cone,
                            para_init,
                            args = (edge_list, optimized_state, tauc, updated_state, N),
                            jac=False,
                            bounds=None,
                            method='SLSQP',
                            callback=None,
                            options={'maxiter': 1000})

                    para = final.x
                    if ifprint:
                        print('final square modulus: ', final.fun)

                    ## check the overlap between the ite state and the optimized state
                    para = (np.array(para)).reshape(-1, numpara)
                    edge_para_dict = {edge:para[index] for index, edge in enumerate(edge_list)}
                    for ci in range(column_index-L, column_index+1):## layer in lightcone      
                        for old_edge in edges_columns[ci]:
                            if old_edge in edge_coeff_dict_ite.keys():
                                if old_edge in lightcone:
                                    optimized_state = circuit_update_theta(old_edge, optimized_state, edge_para_dict[old_edge], N)
                                else:
                                    optimized_state = circuit_update_theta(old_edge, optimized_state, edge_params_dict[old_edge], N)
                    optimized_state = np.array(optimized_state)
                    if ifprint and (N <= 12):
                        _, state_ite = ITE(N, edge_coeff_dict_ite, tau, eigen_list)
                        print('overlap between ite and optimized state: ', state_ite.T.conj() @ optimized_state)

                    if final.fun < -ampth:  ##
                        break

                ## update the parameters for the optimized edge
                for index, edge in enumerate(edge_list):
                    edge_params_dict[edge] = para[index]
                ## check the overlap between the ite state and the optimized state
                state_opt = copy.deepcopy(state)
                for old_edge in edge_coeff_dict_ite.keys():
                    if len(old_edge) == 2:
                        state_opt = circuit_update_theta(old_edge, state_opt, edge_params_dict[old_edge], N)
                state_opt = np.array(state_opt)
                #print('overlap between ite and final optimized state: ', state_ite.T.conj() @ state_opt)
                 
                

    ## get the final state vector
    for ci in range(len(edges_columns)):
        for edge in edges_columns[ ci]:
            state = circuit_update_theta(edge, state, edge_params_dict[edge], N)

    #generate params_list
    values_as_arrays = [np.atleast_1d(value) for value in edge_params_dict.values()]
    # Concatenate and flatten all arrays into a single array
    flattened_array = np.concatenate(values_as_arrays)
    # Convert the flattened array to a list if needed
    params_list = flattened_array.tolist()
    # print(' params_list' , params_list )

    state = np.array(state)

    for id in eigens_ids:
        eigen = eigen_list[id]
        poss = abs(state[id])**2
        # print('eigen', eigen, 'poss', poss)
        exp_poss_dict[eigen] = poss
    return edge_params_dict, params_list, exp_poss_dict, state

def warm_start_parameters_block(N : int, tau:float, numpara:int, edge_coeff_dict : dict, edges_columns :list,                            
    eigen_list:list, ifprint = False):
    '''' warm start the parameters for the circuit using the lightcone adaptivly
    Args:
        N: number of qubits
        tau: imaginary time
        numpara: number of parameters for each edge, 2, 4, or 6
        edge_coeff_dict: coefficients for each edge
        edges_columns: list of edges in each layer
        eigen_list: list of eigenvalues
    Return:
        edge_params_dict: parameters for each edge
        params_list: list of parameters
        exp_poss_dict: probabilities of eigenvalues using warm start circuit
        state: final state vector 
    '''
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????
    
    edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
    exp_poss_dict = {}   ## save probalities of eigenvalues using warm start circuit
    edge_coeff_dict_ite = {} ## record the edges has been executed, to get the ite state for each step

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])

    ## Z term 
    for i in range(N):
        tauc = tau * edge_coeff_dict[(i,)] 
        para = 2*atan( -exp(-2*tauc) ) + pi/2 #use this to use analytic formula (only valid for 1 layer)
        edge_params_dict[(i,)] = para
        circ.ry(para, i)

        edge_coeff_dict_ite[(i,)] = edge_coeff_dict[(i,)]
   
    
    ## ZZ term 
    ## now suppose N is even
    state = Statevector(circ)
    for column_index in range(1, N, 2):
        if ifprint:
            print('\n### column index', column_index)
        block_list = find_block(N, column_index, edges_columns)
        for bi, edge_list in enumerate(block_list): 
            if ifprint:
                print('\nbi:', bi, 'edge_list:', edge_list)
            for ei, edge in enumerate(edge_list):
                edge_coeff_dict_ite[edge] = edge_coeff_dict[edge]
            

            tauc_list = [tau*edge_coeff_dict[edge] for edge in edge_list]
            para_init = [0] * numpara * len(edge_list)
            if ifprint:
                print('optimize edge list', edge_list, ',  para_init: ', para_init)
            final = minimize(square_modulus_cost_light_cone_block,
                    para_init,
                    args = (edge_list, state, tauc_list, N),
                    jac=False,
                    bounds=None,
                    method='SLSQP',
                    callback=None,
                    options={'maxiter': 1000})

            para = final.x
            if ifprint:
                print('final square modulus: ', final.fun)
            para = (np.array(para)).reshape(-1, numpara)
            
            ## update the circuit for the optimized edge
            for ei, edge in enumerate(edge_list):
                edge_params_dict[edge] = para[ei]
                state = circuit_update_theta(edge, state, para[ei], N)
            if ifprint and (N <= 12):
                _, state_ite = ITE(N, edge_coeff_dict_ite, tau, eigen_list)
                print('overlap between ite and optimized state: ', state_ite.T.conj() @ np.array(state))
            

        

    #generate params_list
    values_as_arrays = [np.atleast_1d(value) for value in edge_params_dict.values()]
    # Concatenate and flatten all arrays into a single array
    flattened_array = np.concatenate(values_as_arrays)
    # Convert the flattened array to a list if needed
    params_list = flattened_array.tolist()
    # print(' params_list' , params_list )

    state = np.array(state)
    for id in eigens_ids:
        eigen = eigen_list[id]
        poss = abs(state[id])**2
        # print('eigen', eigen, 'poss', poss)
        exp_poss_dict[eigen] = poss
    return edge_params_dict, params_list, exp_poss_dict, state

