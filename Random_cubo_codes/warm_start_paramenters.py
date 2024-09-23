import sys

import numpy as np 
import math
from math import cos, sin, cosh, sinh, atan, exp, pi, sqrt

from qiskit import *
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.primitives import Estimator

from scipy.optimize import minimize
import more_itertools as mit

import itertools

def compute_coeffedge_list(hamiltonian):

    ''' Buils a dictionary {(i,j) : coeff} for the operators in the hamiltonian. coeff are h_i and J_ij. (i,j) are qubits locations, (i,) for h_i'''

    coeff_dict = {}

    # pauli_op_h = hamiltonian.to_pauli_op() 
    # max_coeff = np.max(np.abs(pauli_op_h.coeffs))
    # for k in pauli_op_h:
    #     string = (k.primitive).to_label()
    #     locations = tuple(mit.locate(string[::-1], lambda x: x == "Z"))
    #     coeff_dict[locations] = k.coeff/max_coeff

    # if () in coeff_dict:
    #     del coeff_dict[()]

    pauli_op_h = hamiltonian.to_pauli_op() 

    for k in pauli_op_h:
        string = (k.primitive).to_label()
        locations = tuple(mit.locate(string[::-1], lambda x: x == "Z"))
        coeff_dict[locations] = k.coeff
    if () in coeff_dict:
        del coeff_dict[()]

    # Get the maximum absolute value
    max_abs_value = max(coeff_dict.values(), key=lambda x: abs(x))
    max_abs_value = abs( max_abs_value)

    # Divide all elements by the maximum absolute value using dictionary comprehension
    coeff_dict = {key: value / max_abs_value for key, value in coeff_dict.items()}

    return coeff_dict

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

    ## quantum gate inspired by e^{ -tau * J_ij * Z_iZ_j}
    i = min(qi)
    j = max(qi)
    
    if len(params_in) == 2:
        params = np.zeros(6)  ## ZY, YZ
        params[2] = params_in[0]
        params[3] = params_in[1]
        # print(len(params_in), params[2], params[3])
        
        ### exp{-i/2 ( params[2]*ZiYj + params[3]*YiZj )}
        circ.rx(-np.pi/2, i)
        circ.rz(-np.pi/2, j)

        circ.cx(i, j)
        circ.ry(params[2], i)
        circ.rz(-params[3], j)
        circ.cx(i, j)

        circ.rx(np.pi/2, i)
        circ.rz(np.pi/2, j)
    
    else: 
        sys.stderr.write('\n!!! Something strange with wart start parameters is happenig !!!')
        sys.exit()

    return circ


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
        exp = estimator.run(circ, op).result().values[0]    #Computes the expectation value of the given operator
        exp_dict[op_str] = exp
        #print(estimator.run(circ, op).result())

    para_init = [0]
    final = minimize(cost_mimic_1op,
                      para_init,
                      args = (exp_dict, tauc),
                      jac=False,
                      bounds=None,
                    #   method='L-BFGS-B',    #use this optimizers because can use derivatives, I think
                      method='SLSQP',
                      callback=None,
    #                               tol=1e-5,
                      options={'maxiter': 10000})
    
    # print('target', 2*atan(-math.exp(-2*tauc)) + pi/2)
    return final.x

def cost_mimic_2op(para:list, *args:tuple):
    """Caculate the cost function to find a good initial parameters for 2-qubit gates, only for YZ_2 ansatz"""
    exp_dict = args[0] ## dict, {pauli string: expectation of the corresbonding pauli op}
    tauc = args[1]  ## tauc = tau * coeff

    # print('exp_dict[ZY]', exp_dict['ZY']) #dovrebbero essere zero???
    # print('exp_dict[YY]', exp_dict['YZ'])

    theta0 = para[0]
    theta1 = para[1]
    cost = 0  ## overlap between ITE and mimic QC
    cost += cos(theta0/2) * cos(theta1/2) * (cosh(tauc) - sinh(tauc)*exp_dict['ZZ'])
    cost += -1j * cos(theta0/2) * sin(theta1/2) * (cosh(tauc)*exp_dict['ZY'] + 1j*sinh(tauc)*exp_dict['Xi'])
    cost += -1j * sin(theta0/2) * cos(theta1/2) * (cosh(tauc)*exp_dict['YZ'] + 1j*sinh(tauc)*exp_dict['Xj'])
    cost += -1 * sin(theta0/2) * sin(theta1/2) * (cosh(tauc)*exp_dict['XX'] + sinh(tauc)*exp_dict['YY'])

    # #ADD NORMALIZATION ??
    # cost = cost/(sqrt(cosh(2*tauc) - sinh(2*tauc)*exp_dict['ZZ']))
    # print(abs(cost))

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

    # print('tauc', tauc)

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
                    #   method='L-BFGS-B',
                      method='SLSQP',
                      callback=None,
                      options={'maxiter': 10000})
    return final.x

def get_good_initial_params_measure(N:int, tau:float, layer:int, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_list:list, shots:int, approximation:bool):
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
    
    eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????
    # print('eigens_ids', eigens_ids)
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

            #para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0] #use this to extimate para from min expectation value
            tauc = tau * edge_coeff_dict[(i,)] 
            para = 2*atan( -exp(-2*tauc) ) + pi/2 #use this to use analytic formula (only valid for 1 layer)

            edge_params_dict[(i,)] = para
            params_list.append(para)
            circ.ry(para, i)

        # ZZ term
        for edge in pairs_all:

            if edge[0] >= edge[1]:
                sys.stderr.write('wrong edge in pairs_all')
                sys.exit()

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
            # print('eigen', eigen, 'poss', poss)
            exp_poss_dict[eigen] = poss


        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict
    
    return layers_edge_params_dict, params_list, layers_exp_poss_dict, vec_final


def find_light_cone(pairs):
    lightcone_dict = {}
    for index, list in enumerate(pairs):
        print(index, list)
        for pair in list:
            qi, qj = pair
            relevent_pairs = []  ##  qubit pairs in the previous layer that in the lightcone of the current pair
            if index > 0:
                for pair_layerm1 in pairs[index-1]: ## qubit pairs in the previous layer
                    if (qi in pair_layerm1) or (qj in pair_layerm1):
                        relevent_pairs.append(pair_layerm1)
            lightcone_dict[pair] = relevent_pairs
    return lightcone_dict