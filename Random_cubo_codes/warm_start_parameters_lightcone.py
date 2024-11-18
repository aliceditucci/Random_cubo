from qiskit import *
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from math import cos, sin, cosh, sinh, atan, exp, pi
from scipy.optimize import minimize

import sys
import copy

import itertools


def find_light_cone(pairs):
    lightcone_dict = {}
    for index, list in enumerate(pairs):
        for pair in list:
            qi, qj = pair
            relevent_pairs = []  ##  qubit pairs in the previous layer that in the lightcone of the current pair
            if index > 0:
                for pair_layerm1 in pairs[index-1]: ## qubit pairs in the previous layer
                    if (qi in pair_layerm1) or (qj in pair_layerm1):
                        relevent_pairs.append(pair_layerm1)
            lightcone_dict[pair] = relevent_pairs
    return lightcone_dict



def circuit_update_zz(Edge,State, Tauc, N):
    j = min(Edge)
    i = max(Edge)

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
    
    State = State.evolve(op)

    # print('states2', State)

    return State


def circuit_update_theta_Yahui(Edge, Circ, paras):

    qcirc = Circ.copy()
    
    i = min(Edge)
    j = max(Edge)

    ### exp{-i/2 ( params[2]*ZiYj + params[3]*YiZj )}
    qcirc.rx(-np.pi/2, i)
    qcirc.rz(-np.pi/2, j)

    qcirc.cx(i, j)
    qcirc.ry(paras[0], i)
    qcirc.rz(-paras[1], j)
    qcirc.cx(i, j)

    qcirc.rx(np.pi/2, i)
    qcirc.rz(np.pi/2, j)
    
    State = Statevector(qcirc)

    # print('states2 yahui', State)

    return State

def square_modulus_cost(Paras : list, *args):

    Edge = args[0]
    State = args[1]
    Tauc = args[2]
    N = args[3]

    State_zz = circuit_update_zz(Edge,State, Tauc, N)
    State_theta = circuit_update_theta(Edge,State, Paras, N)

    # Compute the scalar product (inner product) between the two state vectors
    inner_product = State_zz.inner(State_theta)

    # Compute the square modulus of the inner product
    #square_modulus = abs(inner_product)**2
    square_modulus = abs(inner_product)   #I CHANGED THE SQUARE MODULUS TO NORMAL MODULUS BECAUSE ITS LIKE THIS IN THE OLD WARM START MEASURE


    # Print the inner product and its square modulus
    # print("Inner product:", inner_product)
    # print("Square modulus of the inner product:", square_modulus)

    return - square_modulus

def square_modulus_cost_Yahui(Paras : list, *args):

    Edge = args[0]
    Circ = args[1]
    Tauc = args[2]
    N = args[3]

    State =  Statevector(Circ)

    State_zz = circuit_update_zz(Edge,State, Tauc, N)
    State_theta = circuit_update_theta_Yahui(Edge, Circ, Paras)

    # Compute the scalar product (inner product) between the two state vectors
    inner_product = State_zz.inner(State_theta)

    # Compute the square modulus of the inner product
    square_modulus = abs(inner_product)**2

    # Print the inner product and its square modulus
    # print("Inner product:", inner_product)
    # print("Square modulus of the inner product:", square_modulus)

    return - square_modulus

def square_modulus_cost_light_cone(Paras : list, *args):

    # para_init = np.zeros((len(lightcone_dict[edge]) + 1, 2))

    Edge_list = args[0]
    State = args[1]
    Tauc = args[2]
    Updated_state = args[3]
    N = args[4]

    State_zz = circuit_update_zz(Edge_list[-1], Updated_state, Tauc, N)
    
    for index, edge in enumerate(Edge_list):
        Parameters = [Paras[2*index], Paras[2*index + 1]]
        State = circuit_update_theta(edge,State, Parameters, N)

    # Compute the scalar product (inner product) between the two state vectors
    inner_product = State_zz.inner(State)

    # Compute the square modulus of the inner product
    #square_modulus = abs(inner_product)**2
    square_modulus = abs(inner_product)  #I REMOVED THE SQUARE MODULUS TO JUST MODULUS

    # Print the inner product and its square modulus
    # print("Inner product:", inner_product)
    # print("Square modulus of the inner product:", square_modulus)

    return - square_modulus

def warm_start_parameters_lightcone(N : int, tau:float, edge_coeff_dict : dict, edges_columns :list,  eigen_list:list, lightcone_dict: dict):    

    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????
    
    edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
    exp_poss_dict = {}   ## save probalities of eigenvalues using warm start circuit

    q = QuantumRegister(N, name = 'q')
    circ = QuantumCircuit(q)
    circ.clear()
    circ.h(q[::])

    # Z term
    for i in range(N):

        #para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0] #use this to extimate para from min expectation value
        tauc = tau * edge_coeff_dict[(i,)] 
        para = 2*atan( -exp(-2*tauc) ) + pi/2 #use this to use analytic formula (only valid for 1 layer)
        edge_params_dict[(i,)] = para
        circ.ry(para, i)
        
    ## ZZ term 
    state = Statevector(circ)
    #updated_state = state 

    # print('\nnumber of columns is:', len(edges_columns))

    for column_index, column in enumerate(edges_columns):
        # print('\n##################################################')
        # print('column index', column_index, 'column', column)

        if column_index == 0: 

            first_column_state = copy.deepcopy(state)

            for edge in column:
                
                if len(lightcone_dict[edge]) != 0:
                    sys.stderr.write('something is wrong with the lightcones')
                    sys.exit()
            
                # print('\nedge', edge)  

                tauc = tau * edge_coeff_dict[edge]

                para_init = [0,0]
                #para_init = np.zeros(2)
                #para_init = np.random.uniform(-0.1, 0.1, 2)

                final = minimize(square_modulus_cost,
                                    para_init,
                                    args = (edge, first_column_state, tauc, N),
                                    jac=False,
                                    bounds=None,
                                    # method='L-BFGS-B',
                                    method='SLSQP',
                                    callback=None,
                                    options={'maxiter': 10000})

                para = final.x

                # print('opt paramenters', para)
                # print('final square modulus', final.fun)

                edge_params_dict[edge] = para
                # print('edge_params_dict:', edge_params_dict)

                # print('old state', state)
                first_column_state = circuit_update_theta(edge, first_column_state, para, N)  #THIS LINE SHOULDN'T BE HERE BUT SOMEHOW THERE IS A TINY NUMERICAL ERROR IN THE THETAS IF I DON'T UPDATE THE CIRCUIT

                # print('first column state - new state', first_column_state)
            
        else:
            
            for edge in column: 
                # if len(lightcone_dict[edge]) == 0:
                #         sys.stderr.write('something is wrong with the lightcones')
                #         sys.exit()

                # print('\nedge', edge)  

                # print('len light cone:', len(lightcone_dict[edge]), ', light cone edges:', lightcone_dict[edge])

                # print('initial state', state)

                updated_state = copy.deepcopy(state)
                
                for old_edge in lightcone_dict[edge]:
                    # print('old_edge', old_edge, 'param', edge_params_dict[old_edge])
                    updated_state = circuit_update_theta(old_edge, updated_state, edge_params_dict[old_edge], N)    
                
                # print('updated_state', updated_state)

                edge_list = lightcone_dict[edge] + [edge]
                # print('edge list', edge_list)
                
                tauc = tau * edge_coeff_dict[edge]

                para_init = np.zeros(2 + 2*len(lightcone_dict[edge]))
                #para_init = np.random.uniform(-0.1, 0.1, 2 + 2*len(lightcone_dict[edge]))
                # print('para init', para_init)

                final = minimize(square_modulus_cost_light_cone,
                        para_init,
                        args = (edge_list, state, tauc, updated_state, N),
                        jac=False,
                        bounds=None,
                        # method='L-BFGS-B',
                        method='SLSQP',
                        callback=None,
                        options={'maxiter': 10000})

                para = final.x
                # print('opt paramenters', para)
                # print('final square modulus', final.fun)

                para = (np.array(para)).reshape(-1, 2)
                # print('param list reshaped', para)
                for index, edge in enumerate(edge_list):
                    edge_params_dict[edge] = para[index] 
                # print('edge_params_dict', edge_params_dict)
            
            # print('\n######### previous column update ########')
            # print('column index', column_index, 'column', column)
            # print('previous column is:', edges_columns[column_index -1])

            for edge in edges_columns[column_index -1]:
                # print('edge', edge, 'parameter', edge_params_dict[edge])
                state = circuit_update_theta(edge, state, edge_params_dict[edge], N)

            # print('circuit updated ad the previous column i.e. column', column_index -1)
            # print('updated statevector', state)

    # print('\n######### last column update ########')
    # print('last column is:', edges_columns[ -1])

    for edge in edges_columns[ -1]:
        # print('edge', edge, 'parameter', edge_params_dict[edge])
        state = circuit_update_theta(edge, state, edge_params_dict[edge], N)

    # print('circuit updated at the last column i.e.', len(edges_columns) - 1)
    # print('updated statevector', state)

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

    print('tau:', tau)

    return edge_params_dict, params_list, exp_poss_dict, state

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

def sorted_gates(coefficients, edges, Abs, invert):

    if Abs:
        coefficients = np.abs(coefficients)

    sorted_index = np.argsort(coefficients)

    if invert:
        sorted_index = sorted_index[::-1]

    sorted_edges = [edges[x] for x in sorted_index]

    grouped = []
    group = []

    for t in sorted_edges:
        qi, qj = t
        if any(qi in pair for pair in group) or any(qj in pair for pair in group):
            grouped.append(group)
            group = []
        group.append(t)  
    grouped.append(group)

    if len(list(itertools.chain.from_iterable(grouped))) != len(edges):
        sys.stderr.write('something is wrong with the new columns')
        sys.exit()

    return sorted_edges, grouped 


def sorted_columns(edge_coeff_dict, edges_columns, Abs, invert):
    sum_jcol = []

    if Abs:
        for column in edges_columns:
            abs_sum_j = 0
            for edge in column:
                abs_sum_j += abs(edge_coeff_dict[edge])
            sum_jcol.append(abs_sum_j)

    else:
        for column in edges_columns:
            sum_j = 0 
            for edge in column:
                sum_j += edge_coeff_dict[edge]
            sum_jcol.append(sum_j)

    new_index = np.argsort(sum_jcol)
    
    if invert:
        new_index = new_index[::-1]

    sorted_edge_columns = copy.deepcopy(np.array(edges_columns,  dtype=object)[new_index]) 
    return sorted_edge_columns.tolist()