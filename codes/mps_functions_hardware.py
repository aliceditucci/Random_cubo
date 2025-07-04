from qiskit import *
# from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np 
import itertools

import sys
sys.path.insert(0, '../')

import copy

from math import cos, sin, cosh, sinh, atan, exp, pi, sqrt, acos
from qiskit import *

# from qiskit_aer.primitives import Estimator, Sampler
from qiskit_ibm_runtime import  Estimator

from qiskit_ibm_runtime import SamplerV2 as HardSampler 

# from qiskit_aer import StatevectorSimulator
from scipy.optimize import minimize

import warnings

import time
import neal  # Simulated Annealing Sampler
from collections import OrderedDict

# Suppress specific DeprecationWarning related to Sampler
warnings.filterwarnings(
    "ignore",
    message="Sampler has been deprecated as of Aer 0.15, please use SamplerV2 instead."
)


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
        
    # H = PauliSumOp.from_list(pauli_list)
    H = SparsePauliOp.from_list(pauli_list)
    
    return H

#Function to sort gates for sortin option
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

import sys

import numpy as np 
import math
from math import cos, sin, cosh, sinh, atan, exp, pi, sqrt, acos

from qiskit import *
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator

from scipy.optimize import minimize
import more_itertools as mit

import itertools
import time
import copy

def partition_N(n):
    '''do the partition of a complete graph'''
    indexs = range(n)
    pairs_all = []

    swap_even = [i + pow(-1, i) for i in range(n)]

    swap_odd = [0]
    swap_odd.extend([i + pow(-1, i+1) for i in range(1,n-1)])
    swap_odd.append(n-1)

    pairs_even = [(i, i+1) for i in range(0, n, 2)]
    indexs = np.array(indexs)[swap_even]   ### indexs after swap even
    #     print('\nindexs after swap {}: {}'.format(0, indexs))
    pairs_all.append(pairs_even)
    for i in range(1, n):
        if (i%2)==1:
            pair_odd = [(indexs[i], indexs[i+1]) for i in range(1, n-1, 2)]
            pairs_all.append(pair_odd)
            indexs = np.array(indexs)[swap_odd]   ### indexs after swap even
    #             print('\nindexs after swap {}: {}'.format(i, indexs))

        elif (i%2)==0:
            pair_even = [(indexs[i], indexs[i+1]) for i in range(0, n-1, 2)]
            pairs_all.append(pair_even)
            indexs = np.array(indexs)[swap_even]   ### indexs after swap even
    #             print('\nindexs after swap {}: {}'.format(i, indexs))

    return pairs_all


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

def get_initial_para_1op_Y(N:int, qi:list, coeff:float, tau:float, circ:QuantumCircuit, backendoptions:dict):
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

    options = copy.deepcopy(backendoptions)
    options['shots'] = 1000
    estimator = Estimator(backend_options=options)

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

def get_initial_para_2op_YZ(N:int, qi:list, coeff:float, tau:float, circ:QuantumCircuit, backendoptions:dict):
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
    options = copy.deepcopy(backendoptions)
    options['shots'] = 1000
    estimator = Estimator(backend_options=options)

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


def get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict):
    """get the eigenvalue of Hamiltonian from the bitstring"""
    # Convert bitstring to Ising spins (s_i = 1 - 2x_i)
    s = np.array([1 - 2 * int(b) for b in reversed(bitstr)])
    eigen = 0
    for edge, coeff in edge_coeff_dict.items():
        if len(edge) == 1:
            i = edge[0]
            eigen += coeff * s[i]
        elif len(edge) == 2:
            i, j = edge
            eigen += coeff * s[i] * s[j]
        
    return eigen


def get_good_initial_params_measure_iterate(N:int, tau:float, layer:int, circ:QuantumCircuit, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_idvalue_dict:dict, shots:int, backendoptions:str, if_analytic:int):
    """get the warm start parameters by measurement-based approach
    Args:
        N: number of qubits
        tau: time step for imaginary time evolution
        layer: number of layers in the ansatz
        edge_coeff_dict: dict, {edge: coeff}, coefficients of edges (or vertexes, i as edge (i,)) in the graph
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
        eigen_idvalue_dict: dict, {id: eigenvalue}, eigenvalues of Hamiltonian from smallest to largest, id is the index of eigenvalue, i.e. decimel representation of the binary string
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                            Otherwise, it calculates expectation values with sampling.
        approximation:
        file_path: str, path to save the warm start parameters
        if_analytic: int 1 or 0, 1: use the analytic formula (estimate all parameters from product state) to get the initial parameters; 0: use the measurement-based approach to get the initial parameters
    Return:
        layers_edge_params_dict: dict, {layer: {edge: params}}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """

    layers_edge_params_dict = {}  ## save the warm start parameters for each edge in the graph from l=1 to maximal layer
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    layers_exp_poss_dict = {} ## save probalities of eigenvalues using warm start circuit with l=1 to maximal layer

    for l in range(1, layer+1): 

        edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
        exp_poss_dict = {(eigen, id):0 for (id, eigen) in list(eigen_idvalue_dict.items())[:10]}  ### initialize the dict with the smallest 10 eigenvalues
        # print('\nexp_poss_dict', exp_poss_dict)

        # Z term
        for i in range(N):
            para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, backendoptions)[0] #use this to extimate para from min expectation value
            edge_params_dict[(i,)] = para
            params_list.append(para)
            circ.ry(para, i)

        # ZZ term
        if if_analytic == 1:
            ## got all the initial parameters from product state first, then update it in circuit
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, backendoptions)
                edge_params_dict[edge] = para
                params_list.extend(para)
            for edge in pairs_all:
                para = edge_params_dict[edge]
                circ =  quant_circ_update(N, circ, edge, para)
        else:
            ## got the initial parameters for each edge in the graph by measurement-based approach
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, backendoptions)
                edge_params_dict[edge] = para
                params_list.extend(para)
                circ =  quant_circ_update(N, circ, edge, para)
        
        layers_edge_params_dict['l_'+str(l)] = edge_params_dict


        # run the l layer circuit
        if backendoptions['method'] == 'statevector':
            simulator = AerSimulator(method='statevector')
            shots = backendoptions['shots']

            if shots == 0:
                circ.save_statevector() 
                tcirc = transpile(circ, simulator)
                result = simulator.run(tcirc).result()

                vec_final = np.array( result.get_statevector() ).real
                for id in list(eigen_idvalue_dict.keys())[:min(1000, 2**N)]:
                    eigen = eigen_idvalue_dict[id]
                    poss = abs(vec_final[id])**2
                    exp_poss_dict[(eigen, id)] = float(poss)
            else:
                circ.measure_all()
                tcirc = transpile(circ, simulator)
                result = simulator.run(tcirc, shots = shots).result()
                
                counts= result.get_counts()
                for bitstr, count in counts.items():
                    id = int(bitstr, 2)  ## q_{N-1}..... q1 q0, in statevector backend
                    eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                    exp_poss_dict[(eigen, id)] = count/shots    
        elif backendoptions['method'] == 'matrix_product_state':
            maxbond = backendoptions['matrix_product_state_max_bond_dimension']
            shots = backendoptions['shots']
            circ.measure_all()
            simulator = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=maxbond, shots=shots, max_memory_mb=10**12)
            # tcirc = transpile(circ, simulator)
            result = simulator.run(circ, shots = shots).result()
            counts= result.get_counts()
            # print(counts)
            for bitstr, count in counts.items():
                # print('\ncount', count)
                # print('\nbistr', bitstr)
                id = int(bitstr, 2)   ## q_{N-1}..... q1 q0, in mps backend
                # print('\nid', id)
                eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                # print('\neigen', eigen)
                eigen = round(eigen, 4)  ## round the eigenvalue to 4 decimal places, because all coefficients are rounded to 4 decimal places
                poss = count/shots
                # print('\nposs', poss)
                exp_poss_dict[(eigen, id)] = poss
            
        exp_poss_dict = dict(sorted(exp_poss_dict.items(), key=lambda item: item[0][0]))  ## sort the dict by eigen values in ascending order
        
        # print('\nprint to check!!', list(exp_poss_dict.items())[:5])

        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict

    return layers_edge_params_dict, params_list, layers_exp_poss_dict


def initial_state_ry(N:int, z_list):
    """initialize the quantum state of each qubit n as: ry(a)|0> = (cos(a/2)|0> + sin(a/2) |1>), <Z> = cos(a/2)^2 - sin(a/2)^2 = cos(a)
    Args:
        N: number of qubits
        z_list: list of expectation values of sigma^z for each qubit
    Returns:
        circ_init: QuantumCircuit for the initial state
    
    """
    q = QuantumRegister(N)
    circ_init = QuantumCircuit(q)

    for n in range(N):
        # find the angle theta such that sin(theta/2)**2 * C = k_list[n]
        expz = z_list[n]
        theta = acos(expz)
        circ_init.ry(theta, q[n])

    return circ_init


def get_expz(N, exp_poss_dict, alpha):
    """get the expectation value of <Z> for each qubit
    Args:
        N: number of qubits
        exp_poss_dict: dict, {(exp, id): poss}, probalities of eigenvalues by measurement results
        alpha: CVaR coefficient
    Returns:
        z_list: list of expectation values of sigma^z for each qubit
    """
    # backend = Aer.get_backend('statevector_simulator')
    # job = backend.run(circ)
    # result = job.result()
    # outputstate = np.array(result.get_statevector(circ))   ### amplitude
    # prob_list = [abs(outputstate[i]) ** 2 for i in range(len(outputstate))]
    # prob_list = np.abs(np.array(state_vector)) ** 2

    exp_poss_dict = dict(sorted(exp_poss_dict.items(), key=lambda item: item[0][0]))  ## sort the dict by eigen values in ascending order

    cvar = 0
    total_prob = 0
    expz_array = np.zeros(N)
    for (v, id), p in exp_poss_dict.items():
        if p >= alpha - total_prob:
            p = alpha - total_prob
        total_prob += p
        cvar += p * v

        bits = format(id, '0' + str(N) + 'b')[::-1] # bitstring of the eigen state, (id, binary) = q_{N-1}..... q1 q0, revers it to q0 q1 ... q_{N-1}
        for j, bit in enumerate(bits):
            expz_array[j] += pow(-1, int(bit)) * p

        if abs(total_prob - alpha) < 1e-8:
            break

    cvar /= total_prob
    expz_array /= total_prob
   
    return cvar, expz_array

def entanglement_entropy(state, trace_indices, dof_list, tol=1e-12):
    """

    Inputs:

    state = numpy array statevector

    trace_indices = list of indices of sites to be traced out

    dof_list = list of number of degrees of freedom per site for all sites

    tol = any eigenvalue of the reduced density matrix that is smaller than this tolerence will be neglected

    Outputs:

    ee, rho_reduced = entanglement entropy and reduced density matrix

    """

    # Make sure input is in the right type form and state is normalized to 1
    state, trace_indices, dof_list = np.array(state) / np.linalg.norm(np.array(state)), list(trace_indices), list(
        dof_list)

    # Just a simple list containing the indices from 0 to N - 1 where N is the total number of sites
    site_indices = np.arange(len(dof_list), dtype='int')

    # The dimension of each index to be traced
    trace_dof = [dof_list[i] for i in trace_indices]

    # List containing the indices of the sites not to be traced
    untraced_indices = [idx for idx in site_indices if idx not in trace_indices]

    # The dimension of each index in the list of untraced indices
    untraced_dof = [dof_list[i] for i in untraced_indices]

    # Reshape statevector into tensor of rank N with each index having some degrees of freedom specified by the dof_list
    # for example if it is a spin-1/2 chain then each site has 2 degrees of freedom and the dof_list should be [2]*N = [2, 2, 2, 2, ..., 2]
    state = np.reshape(state, dof_list)
    state = np.transpose(state, axes=site_indices[::-1])   ## to have the smame index order as the qiskit

    # Permute the indices of the rank N tensor so the untraced indices are placed on the left and the ones to be traced on the right
    state = np.transpose(state, axes=untraced_indices + trace_indices)

    # Reshape the rank N tensor into a matrix where you merge the untraced indices into 1 index and you merge the traced indices into 1 index
    # if the former index is called I and the latter J then we have state_{I, J}
    state = np.reshape(state, (np.prod(untraced_dof), np.prod(trace_dof)))

    # The reduced density matrix is given by state_{I, J}*state_complex_conjugated_{J, K}, so we see from here that the indices to be
    # traced out ie the ones contained in the merged big index J are summed over in the matrix multiplication
    rho_reduced = np.matmul(state, state.conjugate().transpose())

    evals = np.linalg.eigh(rho_reduced)[0]
    # Calculate the 'from Newman' (von Neumann) entropy
    ee = 0
    for eval in evals:
        if eval < tol:
            continue
        ee += -eval * np.log2(eval)

    return ee, rho_reduced  # return both the entanglement entropy and the reduced density matrix


def partition_graph(G):
    """
    Partition the edges of a given graph.
    G: Input graph (not necessarily complete)
    Returns: List of edge partitions
    """
    edges = list(G.edges())  # Get the edges of the graph
    n = G.number_of_nodes()  # Number of nodes in the graph
    pairs_all = []

    # Swapping indices for even and odd iterations
    swap_even = [i + pow(-1, i) for i in range(n)]
    swap_odd = [0]
    swap_odd.extend([i + pow(-1, i + 1) for i in range(1, n - 1)])
    swap_odd.append(n - 1)

    # Initial indices and first partition
    indexs = list(range(n))
    pairs_even = [(i, i + 1) for i in range(0, n, 2) if (i, i + 1) in edges or (i + 1, i) in edges]
    indexs = np.array(indexs)[swap_even]  # Apply initial swap
    pairs_all.append(pairs_even)

    # Iterate to create partitions
    for i in range(1, n):
        if i % 2 == 1:
            pair_odd = [(indexs[j], indexs[j + 1]) for j in range(1, n - 1, 2)
                        if (indexs[j], indexs[j + 1]) in edges or (indexs[j + 1], indexs[j]) in edges]
            pairs_all.append(pair_odd)
            indexs = np.array(indexs)[swap_odd]  # Swap for odd iteration
        else:
            pair_even = [(indexs[j], indexs[j + 1]) for j in range(0, n - 1, 2)
                         if (indexs[j], indexs[j + 1]) in edges or (indexs[j + 1], indexs[j]) in edges]
            pairs_all.append(pair_even)
            indexs = np.array(indexs)[swap_even]  # Swap for even iteration

    return pairs_all


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

    return eigen_idvalue_dict  # Return both sets






def get_good_initial_params_measure_iterate_hardware(N:int, tau:float, layer:int, circ:QuantumCircuit, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_idvalue_dict:dict, shots:int, backendoptions:str, if_analytic:int, hardware_backend=None, layout=None):
    """get the warm start parameters by measurement-based approach
    Args:
        N: number of qubits
        tau: time step for imaginary time evolution
        layer: number of layers in the ansatz
        edge_coeff_dict: dict, {edge: coeff}, coefficients of edges (or vertexes, i as edge (i,)) in the graph
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
        eigen_idvalue_dict: dict, {id: eigenvalue}, eigenvalues of Hamiltonian from smallest to largest, id is the index of eigenvalue, i.e. decimel representation of the binary string
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                            Otherwise, it calculates expectation values with sampling.
        approximation:
        file_path: str, path to save the warm start parameters
        if_analytic: int 1 or 0, 1: use the analytic formula (estimate all parameters from product state) to get the initial parameters; 0: use the measurement-based approach to get the initial parameters
    Return:
        layers_edge_params_dict: dict, {layer: {edge: params}}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """

    layers_edge_params_dict = {}  ## save the warm start parameters for each edge in the graph from l=1 to maximal layer
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    layers_exp_poss_dict = {} ## save probalities of eigenvalues using warm start circuit with l=1 to maximal layer

    for l in range(1, layer+1): 

        edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
        exp_poss_dict = {(eigen, id):0 for (id, eigen) in list(eigen_idvalue_dict.items())[:10]}  ### initialize the dict with the smallest 10 eigenvalues
        # print('\nexp_poss_dict', exp_poss_dict)

        # Z term
        for i in range(N):
            para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, backendoptions)[0] #use this to extimate para from min expectation value
            edge_params_dict[(i,)] = para
            params_list.append(para)
            circ.ry(para, i)

        # ZZ term
        if if_analytic == 1:
            ## got all the initial parameters from product state first, then update it in circuit
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, backendoptions)
                edge_params_dict[edge] = para
                params_list.extend(para)
            for edge in pairs_all:
                para = edge_params_dict[edge]
                circ =  quant_circ_update(N, circ, edge, para)
        else:
            ## got the initial parameters for each edge in the graph by measurement-based approach
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, backendoptions)
                edge_params_dict[edge] = para
                params_list.extend(para)
                circ =  quant_circ_update(N, circ, edge, para)
        
        layers_edge_params_dict['l_'+str(l)] = edge_params_dict

        # # run the l layer circuit
        # if backendoptions['method'] == 'statevector':
        #     simulator = AerSimulator(method='statevector')
        #     shots = backendoptions['shots']

        #     if shots == 0:
        #         circ.save_statevector() 
        #         tcirc = transpile(circ, simulator)
        #         result = simulator.run(tcirc).result()

        #         vec_final = np.array( result.get_statevector() ).real
        #         for id in list(eigen_idvalue_dict.keys())[:min(1000, 2**N)]:
        #             eigen = eigen_idvalue_dict[id]
        #             poss = abs(vec_final[id])**2
        #             exp_poss_dict[(eigen, id)] = float(poss)
        #     else:
        #         circ.measure_all()
        #         tcirc = transpile(circ, simulator)
        #         result = simulator.run(tcirc, shots = shots).result()
                
        #         counts= result.get_counts()
        #         for bitstr, count in counts.items():
        #             id = int(bitstr, 2)  ## q_{N-1}..... q1 q0, in statevector backend
        #             eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
        #             exp_poss_dict[(eigen, id)] = count/shots    
        # elif backendoptions['method'] == 'matrix_product_state':
        #     maxbond = backendoptions['matrix_product_state_max_bond_dimension']
        #     shots = backendoptions['shots']
        #     circ.measure_all()
        #     simulator = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=maxbond, shots=shots, max_memory_mb=10**12)
        #     # tcirc = transpile(circ, simulator)
        #     result = simulator.run(circ, shots = shots).result()
        #     counts= result.get_counts()
        #     # print(counts)
        #     for bitstr, count in counts.items():
        #         # print('\ncount', count)
        #         # print('\nbistr', bitstr)
        #         id = int(bitstr, 2)   ## q_{N-1}..... q1 q0, in mps backend
        #         # print('\nid', id)
        #         eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
        #         # print('\neigen', eigen)
        #         eigen = round(eigen, 4)  ## round the eigenvalue to 4 decimal places, because all coefficients are rounded to 4 decimal places
        #         poss = count/shots
        #         # print('\nposs', poss)
        #         exp_poss_dict[(eigen, id)] = poss
        
        print('\n analytic part done, start hardware')
        t1 = time.time()

        if hardware_backend is None:
            sys.stderr.write('something is wrong with hardware run')
            sys.exit()
        else:
            shots = backendoptions['shots']
            circ.measure_all()
            # transpiled_circ = transpiler.run(circ)
            transpiled_circ = transpile(circ, backend=hardware_backend, initial_layout=layout, optimization_level=3)

            sampler = HardSampler(mode=hardware_backend) 

            # DD
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"

            #PT
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.num_randomizations = 200 #16
            sampler.options.twirling.shots_per_randomization = 500 #125000
            # sampler.options.twirling.enable_measure = False
            # sampler.options.twirling.enable_gates = False

            # Experimental
            sampler.options.experimental = {"execution_path": "gen3-turbo"}

            job = sampler.run([transpiled_circ], shots=shots)  

            result = job.result()
            # counts= result.get_counts()
            # counts = dict(result.quasi_dists[0]) 
            counts = result[0].data.meas.get_counts()

            t2 = time.time()
            print('\n hardware done', t2-t1)
            
            # print(counts)
            for bitstr, count in counts.items():
                # print('\ncount', count)
                # print('\nbistr', bitstr)
                id = int(bitstr, 2)   ## q_{N-1}..... q1 q0, in mps backend
                # print('\nid', id)
                eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                # print('\neigen', eigen)
                eigen = round(eigen, 4)  ## round the eigenvalue to 4 decimal places, because all coefficients are rounded to 4 decimal places
                poss = count/shots
                # print('\nposs', poss)
                exp_poss_dict[(eigen, id)] = poss

                
            
        exp_poss_dict = dict(sorted(exp_poss_dict.items(), key=lambda item: item[0][0]))  ## sort the dict by eigen values in ascending order
        
        # print('\nprint to check!!', list(exp_poss_dict.items())[:5])

        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict

    return layers_edge_params_dict, params_list, layers_exp_poss_dict

################ functions for warm start parameters by approximated(analytic) approach ################
def cost_approximate_1op(para:list, *args:tuple):
    """Caculate the cost function to find a good initial parameters for 1-qubit gates in the approximated approach, which do not need do the measure"""
    phi= args[0]  ## angle in the initial state
    tauc = args[1]  ## tau * coeff

    theta = para[0]
    cost = 0  ## overlap between ITE and mimic QC
    cost += cos(theta/2) * (cosh(tauc) - sinh(tauc)*cos(phi))
    cost += sin(theta/2) * sinh(tauc) * sin(phi)
    return -abs(cost)

def get_initial_para_1op_Yappro(N, i, phi, tauc):
    """
    Get the approximated warm start parameters for all the single-qubit term,
    Args:
        N: number of qubits
        i: the qubit index
        phi: the angle in the initial state qubit i, i.e., initial state is |phi_i> = R_y(phi_i) |0>. A special case is |+> = R_y(pi/2) |0>
        tauc: the multication of tau and coefficient of the single-qubit term, i.e., tauc = tau * h_i
    Returns:
        para: the warm start parameter for the single-qubit term
    """

    para_init = [0]
    final = minimize(cost_approximate_1op,
                para_init,
                args = (phi, tauc),
                jac=False,
                bounds=None,
                method='SLSQP',
                callback=None,
                options={'maxiter': 10000})
 
    return float(final.x[0])

def cost_approximate_2op(para:list, *args:tuple):
    """Caculate the cost function to find a good initial parameters for 2-qubit gates in a approximated approach, which do not need do the measure"""
    phi0= args[0]  ## angle of qubit i, in the initial state
    phi1= args[1]  ## angle of qubit j, in the initial state
    tauc = args[2]  ## tau * coeff

    theta0 = para[0]
    theta1 = para[1]
    cost = 0  ## overlap between ITE and mimic QC
    cost += cos(theta0/2) * cos(theta1/2) * ( cosh(tauc) - sinh(tauc) * cos(phi0) * cos(phi1) )
    cost += -1j * cos(theta0/2) * sin(theta1/2) * ( 1j * sinh(tauc) * sin(phi0))
    cost += -1j * sin(theta0/2) * cos(theta1/2) * ( 1j*sinh(tauc) * sin(phi1))
    cost += -1 * sin(theta0/2) * sin(theta1/2) * (cosh(tauc) * sin(phi0) * sin(phi1) )
    return -abs(cost)

def get_initial_para_2op_YZappro(N, edge, phi_list, tauc):
    """Get the approximated warm start parameters for a two-qubit term without measurements,
    Args:
        N: number of qubits
        edge: list of two qubits, i.e., [i, j], which means the two-qubit term is acting on qubit i and qubit j
        phi_list: list of angles in the product state for each qubit,  |phi_0, phi_1, ..., phi_{N-1}> = \Prod_{i=0}^{N-1} R_y(phi_i) |0>. A special case is |+> = R_y(pi/2) |0>
        tauc: multication of tau and coefficient of the two-qubit term, i.e., tauc = tau * J_{ij}
    Returns:
        para_list: warm start parameters for this two-qubit term"""
    
    i = min(edge)
    j = max(edge)

    phi0 = phi_list[i]
    phi1 = phi_list[j]

    para_init = [0, 0]
    final = minimize(cost_approximate_2op,
                para_init,
                args = (phi0, phi1, tauc),
                jac=False,
                bounds=None,
                method='SLSQP',
                callback=None,
                options={'maxiter': 10000})
    
    para = [ float( final.x[0]), float( final.x[1])]
    return para

##########################################################################################################


def MimicITE_iterate(N:int, tau:float, layer:int, expz_array:np.array, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_idvalue_dict:dict, shots:int, backendoptions:str, if_analytic:int, hardware_backend=None, session=None, transpiler=None):
    """get the warm start parameters by measurement-based approach
    Args:
        N: number of qubits
        tau: time step for imaginary time evolution
        layer: number of layers in the ansatz
        expz_list: list of expectation values of sigma^z for each qubit in the initial state
        edge_coeff_dict: dict, {edge: coeff}, coefficients of edges (or vertexes, i as edge (i,)) in the graph
        pairs_all: list of qubit index pairs (edges) in a order to parallel the circuit
        eigen_idvalue_dict: dict, {id: eigenvalue}, eigenvalues of Hamiltonian from smallest to largest, id is the index of eigenvalue, i.e. decimel representation of the binary string
        shots (None or int): The number of shots. If None and approximation is True, it calculates the exact expectation values. 
                            Otherwise, it calculates expectation values with sampling.
        if_analytic: int 1 or 0, 1: use the analytic formula (estimate all parameters from product state) to get the initial parameters; 0: use the measurement-based approach to get the initial parameters
    Return:
        layers_edge_params_dict: dict, {layer: {edge: params}}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """

    layers_edge_params_dict = {}  ## save the warm start parameters for each edge in the graph from l=1 to maximal layer
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    layers_exp_poss_dict = {} ## save probalities of eigenvalues using warm start circuit with l=1 to maximal layer

    circ = initial_state_ry(N, expz_array)
    for l in range(1, layer+1): 

        edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
        exp_poss_dict = {(eigen, id):0 for (id, eigen) in list(eigen_idvalue_dict.items())[:10]}  ### initialize the dict with the smallest 10 eigenvalues

        if if_analytic == 1: 
            ## got the initial parameters for each edge in the graph by approximated approach
            #single qubit term
            phi_list = [acos(expz_array[i]) for i in range(N)]  ## angles of RY gate in the initial state
            for i in range(N):
                phi = phi_list[i]
                tauc = tau * edge_coeff_dict[(i,)]
                para = get_initial_para_1op_Yappro(N, i, phi, tauc)
                edge_params_dict[(i,)] = para
                params_list.append(para)
                phi_list[i] += para  ## update the angle in the initial state, so when approximating the params of two-qubit term, the effects of single-qubit terms are already counted
                circ.ry(para, i)
            
            #two qubit term, got the approximated params for all terms
            for edge in pairs_all:
                para = get_initial_para_2op_YZappro(N, edge, phi_list, tau * edge_coeff_dict[edge])
                edge_params_dict[edge] = para
                params_list.extend(para)

            # update the circuit
            for edge in pairs_all:
                para = edge_params_dict[edge]
                circ =  quant_circ_update(N, circ, edge, para)
        else:
            ## got the initial parameters for each edge in the graph by measurement-based approach
            for i in range(N):
                para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, backendoptions)[0] #use this to extimate para from min expectation value
                edge_params_dict[(i,)] = para
                params_list.append(para)
                circ.ry(para, i)
        
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, backendoptions)
                edge_params_dict[edge] = para
                params_list.extend(para)
                circ =  quant_circ_update(N, circ, edge, para)
        
        layers_edge_params_dict['l_'+str(l)] = edge_params_dict

        print('backend:', backendoptions['method'])

        # run the l layer circuit
        if backendoptions['method'] == 'statevector':
            simulator = AerSimulator(method='statevector')
            shots = backendoptions['shots']

            if shots == 0:
                circ.save_statevector() 
                tcirc = transpile(circ, simulator)
                result = simulator.run(tcirc).result()

                vec_final = np.array( result.get_statevector() ).real
                for id in list(eigen_idvalue_dict.keys())[:min(1000, 2**N)]:
                    eigen = eigen_idvalue_dict[id]
                    poss = abs(vec_final[id])**2
                    exp_poss_dict[(eigen, id)] = float(poss)
            else:
                circ.measure_all()
                tcirc = transpile(circ, simulator)
                result = simulator.run(tcirc, shots = shots).result()
                
                counts= result.get_counts()
                for bitstr, count in counts.items():
                    id = int(bitstr, 2)  ## q_{N-1}..... q1 q0, in statevector backend
                    eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                    exp_poss_dict[(eigen, id)] = count/shots    
        elif backendoptions['method'] == 'matrix_product_state':
            maxbond = backendoptions['matrix_product_state_max_bond_dimension']
            shots = backendoptions['shots']
            circ.measure_all()
            simulator = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=maxbond, shots=shots, max_memory_mb=10**12)
            # tcirc = transpile(circ, simulator)
            result = simulator.run(circ, shots = shots).result()
            counts= result.get_counts()
       
            for bitstr, count in counts.items():
                id = int(bitstr, 2)   ## q_{N-1}..... q1 q0, in mps backend
                eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                eigen = round(eigen, 4)  ## round the eigenvalue to 4 decimal places, because all coefficients are rounded to 4 decimal places
                poss = count/shots
                exp_poss_dict[(eigen, id)] = poss

        elif backendoptions['method'] == 'hardware':
            print('\n analytic part done, start hardware')
            t1 = time.time()

            shots = backendoptions['shots']
            circ.measure_all()
            transpiled_circ = transpiler.run(circ)
            # transpiled_circ = transpile(circ, backend=hardware_backend, initial_layout=layout, optimization_level=3)

            sampler = HardSampler(mode=session) 

            # DD
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"

            #PT
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.num_randomizations = 200 #80 #200 #16
            sampler.options.twirling.shots_per_randomization = 500 #125 #500 #125000
            # sampler.options.twirling.enable_measure = False
            # sampler.options.twirling.enable_gates = False

            # Experimental
            sampler.options.experimental = {"execution_path": "gen3-turbo"}

            job = sampler.run([transpiled_circ], shots=shots)  

            result = job.result()
            # counts= result.get_counts()
            # counts = dict(result.quasi_dists[0]) 
            counts = result[0].data.meas.get_counts()

            t2 = time.time()
            print('\n hardware done in sec:', t2-t1)

                        # print(counts)
            for bitstr, count in counts.items():
                # print('\ncount', count)
                # print('\nbistr', bitstr)
                id = int(bitstr, 2)   ## q_{N-1}..... q1 q0, in mps backend
                # print('\nid', id)
                eigen = get_eigenvalue_from_bitstring(bitstr, edge_coeff_dict)
                # print('\neigen', eigen)
                eigen = round(eigen, 4)  ## round the eigenvalue to 4 decimal places, because all coefficients are rounded to 4 decimal places
                poss = count/shots
                # print('\nposs', poss)
                exp_poss_dict[(eigen, id)] = poss        

        exp_poss_dict = dict(sorted(exp_poss_dict.items(), key=lambda item: item[0][0]))  ## sort the dict by eigen values in ascending order


        layers_exp_poss_dict['l_'+str(l)] = exp_poss_dict

    return layers_edge_params_dict, params_list, layers_exp_poss_dict