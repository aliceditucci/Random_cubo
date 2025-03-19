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
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_aer import StatevectorSimulator
from scipy.optimize import minimize

import warnings

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

#Initialize circuit at every iteration

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

#Optimization Iterations

def get_parameters(N:int, tau:float, layer:int, circ:QuantumCircuit, edge_coeff_dict:dict, pairs_all:list, \
                                    eigen_list:list, shots:int, approximation:bool, if_analytic:int):
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
        if_analytic: int 1 or 0, 1: use the analytic formula (estimate all parameters from product state) to get the initial parameters; 0: use the measurement-based approach to get the initial parameters
    Return:
        layers_edge_params_dict: dict, {layer: {edge: params}}, warm start parameters for each edge in the graph from l=1 to maximal layer
        layers_exp_poss_dict: dict, {layer: {exp: poss}}, probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    """
    
    # eigens_ids = np.argsort(eigen_list)[:100]  ## return the id of the lowest 100 eigenvalues    ????

    layers_edge_params_dict = {}  ## save the warm start parameters for each edge in the graph from l=1 to maximal layer
    params_list = []  ## save the warm start parameters for each layer, just for good formula to run vqe
    layers_exp_poss_dict = {} ## save probalities of eigenvalues using warm start circuit with l=1 to maximal layer
    
    for l in range(1, layer+1): 

        edge_params_dict = {} ## to save the initial parameters for each vertex or edge in l'th layer
        # exp_poss_dict = {}  ### record the {exp:poss} information after the excution of l'th layer

        # Z term
        for i in range(N):

            para = get_initial_para_1op_Y(N, [i], edge_coeff_dict[(i,)], tau, circ, shots, approximation)[0] #use this to extimate para from min expectation value
            edge_params_dict[(i,)] = para
            params_list.append(para)
            circ.ry(para, i)
        
            # print('parameters', para)

        # ZZ term
        if if_analytic == 1:
            ## got all the initial parameters from product state first, then update it in circuit
            for edge in pairs_all:
                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, shots, approximation)
                edge_params_dict[edge] = para
                params_list.extend(para)
            for edge in pairs_all:
                para = edge_params_dict[edge]
                circ =  quant_circ_update(N, circ, edge, para)
        else:
            for edge in pairs_all:
                if edge[0] >= edge[1]:
                    sys.stderr.write('wrong edge in pairs_all')
                    sys.exit() 

                para = get_initial_para_2op_YZ(N, edge, edge_coeff_dict[edge], tau, circ, shots, approximation)
                edge_params_dict[edge] = para
                params_list.extend(para)
                circ =  quant_circ_update(N, circ, edge, para)

            # print('quadratic parameters', para)
        
        layers_edge_params_dict['l_'+str(l)] = edge_params_dict

    return layers_edge_params_dict, params_list, circ

#Cost function linear terms
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

    return final.x

#Cost function quadratic terms
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
        # print(estimator.run(circ, op).result())

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



def compute_expectations(N, eigen_list, circ, shots, alpha):
    
    sampler = Sampler(run_options={"shots": shots})

    circ.measure_all()
    # exp = sampler.run(circ).result().quasi_dists[0]   #Computes the expectation value of the given operator
    exp = sampler.run(circ).result().quasi_dists[0] 

    # if shots:
    #     exp_padded = {key: exp.get(key, 0) for key in range(2**N)}
    #     print(exp_padded)
    #     print(len(exp_padded))
  

    # sorted_indices = np.argsort(eigen_list)
    sorted_indices = np.argsort(eigen_list)[:100]

    cvar = 0
    total_prob = 0
    expz_array = np.zeros(N)
    exp_poss_dict = {}

    for index in sorted_indices:

        value = float(eigen_list[index])
        p = exp.get(index, 0)
        exp_poss_dict[value] = p

        if p >= alpha - total_prob:
            p = alpha - total_prob
        total_prob += p
        cvar += p * value

        bits = format(index, '0' + str(N) + 'b')[::-1] # bitstring of the eigen state
        # print(index, value, p, bits, int(bits,2), cvar, total_prob, 'index, eigen, proba, bits, cvar, totalprob')
        for j, bit in enumerate(bits):
            expz_array[j] += pow(-1, int(bit)) * p

        if abs(total_prob - alpha) < 1e-8:
            break

    cvar /= total_prob
    expz_array /= total_prob
    # print('final', total_prob, cvar,expz_array)

    layers_exp_poss_dict = {}
    layers_exp_poss_dict['l_1'] = exp_poss_dict   #USELESS STEP REMOVE LAYERS EVERYWHERE

    return exp.get(sorted_indices[0], 0), layers_exp_poss_dict, cvar, expz_array


def entanglement_entropy(circ, trace_indices, dof_list, tol=1e-12):
    """

    Inputs:

    state = numpy array statevector

    trace_indices = list of indices of sites to be traced out

    dof_list = list of number of degrees of freedom per site for all sites

    tol = any eigenvalue of the reduced density matrix that is smaller than this tolerence will be neglected

    Outputs:

    ee, rho_reduced = entanglement entropy and reduced density matrix

    """

    new_circ = copy.deepcopy(circ)
    # backend = AerSimulator.from_backend('statevector_simulator')
    backend = StatevectorSimulator()
    result = backend.run(new_circ).result()
    state = np.array( result.get_statevector() ).real

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