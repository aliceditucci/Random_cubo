from qiskit import *
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import ParameterVector
import argparse

from VQE_and_QAOA import VQE_and_QAOA

import networkx as nx
import time
import neal  # Simulated Annealing Sampler
from collections import OrderedDict

from Function_gi_params import *

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
        pauli_list.append(('Z', [i], h_list[i]))
        
    for k, (i, j) in enumerate(edge_list):
        pauli_list.append(('ZZ', [i, j], J_list[k]))
        
    H = SparsePauliOp.from_sparse_list(pauli_list, num_qubits = N)
    
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
    
    return eigen_idvalue_dict


def main(): 
    #the eigenvalue are round to 4 decimal places, be careful if you need a higher precision

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=6)
    parser.add_argument("--r", help="instances index", required=False, type=int, default=0)
    parser.add_argument("--alpha", help="CVaR coeffcient", required=False, type=float, default=0.01)
    parser.add_argument("--ansatz_type", help="type of ansatz, qaoa or 'parallel_cz and so on'", required=False, type=str, default='parallel_cz')
    parser.add_argument("--tau", help="imiginary time evolution parameter if using warm start", required=False, type=float, default=0.0)
    parser.add_argument("--layer", help="Number of repetions of the ansatz layers", required=False, type=int, default=1)

    
    parser.add_argument("--backend_method", help="backend method for simulation, statevector or 'matrix_product_state'", required=False, type=str, default='matrix_product_state')
    parser.add_argument("--bond", help="bond dimension for matrix product state", required=False, type=int, default=1000)
    parser.add_argument("--shots", help="number of shots, 0 for exact simulation (inifinite shots)", required=False, type=int, default=10000)

    parser.add_argument("--graph_density", help="density of the graph, 1 for complete graph, 0 for 3reg graph", required=False, type=float, default=0.0)

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

    ansatz_type = args.ansatz_type

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\nn_qubits: {}, \nr: {}, \nalpha: {}, \nshots: {}, \nansatz: {}, \nlayer: {}, \ntau: {}'\
        .format(n_qubits, r, alpha, shots, ansatz_type, layer, tau))

    ### folder to save result
    data_dir = './data/data_shots_{}_layer_{}/alpha_{}/N_{}/r_{}/ansatz_type_{}/'\
        .format(shots, layer, alpha, n_qubits, r, ansatz_type)
    os.makedirs(data_dir, exist_ok=True)

    #region load qubo instances, get Hamiltonian and edge_coeff_dict
    coeffs_file = f"../instances/{int(graph_density * 100):03}/N_{n_qubits}/QUBO_coeff_{n_qubits}V_r_{r}.txt"
    coeff_list = np.loadtxt(coeffs_file)

    graph_file = f"../instances/{int(graph_density * 100):03}/N_{n_qubits}/QUBO_{n_qubits}V_r_{r}.gpickle"
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    edge_list = list(itertools.chain.from_iterable(partition_graph(G)))  ## used in bond dimension more than 100, same with Alice
    h_list = coeff_list[:n_qubits ]
    J_list = coeff_list[n_qubits : n_qubits + len(edge_list)]
    edge_coeff_dict = {}
    edge_coeff_dict.update({(i,): h_val for i, h_val in enumerate(h_list)})
    edge_coeff_dict.update({edge: J_val for edge, J_val in zip(edge_list, J_list)}) #CHANGED COMPARED TO THE OLD CODE
    
    H = Hamiltonian_qubo(n_qubits, edge_list, h_list, J_list)
    #endregion

    ## order for two-qubit gate in circuit
    pairs_all = list(itertools.chain.from_iterable(partition(n_qubits)))


    if backend_method == 'matrix_product_state':
        backendoptions = {'method':backend_method, 'matrix_product_state_max_bond_dimension': bond, 'shots': shots}
    else:
        backendoptions = {'method':backend_method, 'shots': shots}


    vqe = VQE_and_QAOA(Hamiltonian = H, n_qubits = n_qubits, ansatz_type = ansatz_type, alpha = alpha, backendoptions=backendoptions,circuit_show = False, shots = shots, edge_coeff_dict = edge_coeff_dict)
    vqe.edge_coeff_dict = edge_coeff_dict
    vqe.edge_list = pairs_all

    if n_qubits <= 5:
        E_min, E_max, ground_id_list = vqe.Get_minimun_from_H_mat()
        print('\nE_min from ED: ', E_min)
        for id in ground_id_list:
            print('ground state: ', np.binary_repr(id, n_qubits))
    else:
        eigen_idvalue_dict = simulated_annealing(n_qubits, edge_list, h_list, J_list)
        vqe.exp_min = min(eigen_idvalue_dict.values())
        print('\nE_min from simulated annealing: ', vqe.exp_min)
    

    ## set the gate order in circuit
    opt_cvar = 0
    opt_poss = 0
    for rand_params in range(10):
        params_init = np.random.uniform(-np.pi, np.pi, 2 * layer)
        print("\n\n################## Start optimization with initial parameters:", params_init)
        
        vqe.r_eval = []
        vqe.poss_eval= []
        vqe.cvar_eval = []
        vqe.std_eval = []
        final = minimize(vqe.CVaR_expectation,
                        params_init,
                        jac=False,
                        bounds=None,
                        method='COBYLA',
                        callback=None,
                        options={'maxiter': 1000})

        min_cvar = min(vqe.cvar_eval)
        max_poss = max(vqe.poss_eval)
        
        print('\nvqe.cvar_eval[0]:', vqe.cvar_eval[0], ', min_cvar:', min_cvar, ", opt_cvar:", opt_cvar)
        print("\nvqe.poss_eval[0]:", vqe.poss_eval[0], ' , max_poss:', max_poss, ", opt_poss:", opt_poss)

        if min_cvar < opt_cvar:
            opt_cvar = min_cvar
            opt_poss = max_poss

            save_list = np.array([vqe.cvar_eval, vqe.r_eval, vqe.poss_eval, vqe.std_eval])
            np.savetxt(data_dir + '/result_{}_tau_{}.txt'.format("random", tau), save_list.T)
            np.savetxt(data_dir + '/final_params_{}_tau_{}.txt'.format("random", tau), final.x)
            print('Write sucessfully to ' + data_dir)


if __name__ == "__main__":
    main()
