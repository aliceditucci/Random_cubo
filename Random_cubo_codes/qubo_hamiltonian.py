from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp
import numpy as np 

import sys
sys.path.insert(0, '../')

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
