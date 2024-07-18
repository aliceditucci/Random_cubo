from qiskit import Aer, QuantumRegister, QuantumCircuit
import sys
import numpy as np

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

class VQE:
    """
    Class to represent a VQE run

    Attributes
    H : PauliSumOp
        Hamiltonian represented as weighted Pauli sum
    eigen_list : list(Real)
        The eigenvalues of the Hamiltonian
    n_qubits : Int
        The number of qubits
    ansatz_type : String
        String encoding the ansatz type (currently R_xyz, R_y)
    shots : Int
        The number of shots that are executed during the measurement
    alpha : Real
        Coefficient for the conditional value at risk ranging from 0 to 1
    exp_min : Real
        Minimum eigenvalue of the Hamiltonian
    exp_max : 
        Maximum eigenvalue of the Hamiltonian
    ground_id_list : list[Int]
        List with the indices corresponding to the ground state(s)
    cvar_eval: list[Real]
        List for saving the conditional values at risk during a VQE
    r_eval: list[Real]
        List of relative deviations of the solution compared to he difference between maximum
        and minimum eigenvalue for every iteration
    poss_eval:
        List of probabilities to measure the ground state for every iteration 
    n_iter : Int
        Counter for the number of iterations of the classical optimizer takes during the VQE
    cnot_type : 
        String indicating the type of entangling layers used
    """

    def __init__(self, Hamiltonian=None, n_qubits=None):
        """
        Args:
            Hamiltonian (PauliSumOp): Problem Hamiltonian
            n_qubits (Int): Number of qubits            
            ansatz_type (String): VQE ansatz type
            shots (Int): The number of shots
            alpha (Real): The coefficient for the conditional value at risk
        """

        self.H = Hamiltonian.reduce().to_spmatrix().real
        self.eigen_list = self.H.diagonal()
        self.n_qubits = n_qubits
        self.ansatz_type = None
        self.shots = None
        self.alpha = None  # CVaR coefficient

        self.exp_min, self.exp_max, self.ground_id_list = self.Get_minimun_from_H_mat()

        self.cvar_eval = []  # list of the cvar for every evaluation
        # list of the ratio for everay evaluation, r = (exp - exp_min) / (exp_max - exp_min)
        self.r_eval = []
        self.poss_eval = []  # list of the possibility of the optimal solution for every evaluation

        self.n_iter = 0  # count the iteration step of the classical optimization
        self.edge_list = None   ### edge_list of the graph related to the QUBO Hamiltonian, will decide the order of two-qubit gate in circuit

    def Get_minimun_from_H_mat(self):
        """
        Determine the minimum and the maximum eigenvalue of the Hamiltonian as well as the indices 
        of the ground states

        Returns:
            exp_min (Real): Minimum eigenvalue
            exp_max (Real): Maximum eigenvalue
            ground_id_list (list[Int]): List containing the indices of the ground states
        """
        exp_list = self.eigen_list

        exp_min = min(exp_list)
        exp_max = max(exp_list)

        ground_id_list = []
        for i in range(len(exp_list)):
            if abs(exp_list[i] - exp_min) < 1e-6:
                ground_id_list.append(i)

        return exp_min.real, exp_max.real, ground_id_list

    def R_0(self, q, circ, param):
        """
        Add the initial rotation layer of the ansatz for a given set of parameters to a given circuit

        Args:
            q (QuantumRegister): Quantum register for the qubits
            circ (QuantumCircuit): The quantum circuit
            param (np.array): A list of parameters for the rotation gates in the initial layer

        Returns:
            (Int): returns 0 if the function ran successfully
        """

        if self.ansatz_type == 'R_xzy':
            # Case that we have an ansatz utilizing R_x, R_z and R_y gates, for the first layer R_x and R_z are sufficient
            if len(param) != 2*self.n_qubits:
                sys.stderr.write(
                    '!!! ansatz R_xzy, Error of the parameters in R_0 !!!')
                sys.exit()
            for i in range(self.n_qubits):
                circ.rx(param[2*i], q[i])
                circ.rz(param[2*i+1], q[i])
        elif self.ansatz_type == 'R_y':
            # Case that we only have R_y gates
            if len(param) != 1*self.n_qubits:
                sys.stderr.write(
                    '!!! ansatz R_y, Error of the parameters in R_0 !!!')
                sys.exit()
            for i in range(self.n_qubits):
                circ.ry(param[i], q[i])
        else:
            # Unknown ansatz type
            sys.stderr.write(
                '!!! Wrong ansatz type, can only be R_xzy or R_y !!!')
            sys.exit()

        return 0

    def R_j(self, q, circ, param):
        """
        Add the rotation layer of the ansatz for a given set of parameters to a given circuit

        Args:
            q (QuantumRegister): Quantum register for the qubits
            circ (QuantumCircuit): The quantum circuit
            param(np.array): A list of parameters for the rotation gates in the initial layer

        Returns:
            (Int): returns 0 if the function ran successfully
        """

        if len(param) != 1*self.n_qubits:
            sys.stderr.write(
                '!!! Error of the parameters in R_j !!!')
            sys.exit()
        for i in range(self.n_qubits):
            circ.ry(param[i], q[i])

        return 0

    def CNOT_circular(self, q, circ):
        """
        Add a circular CNOT layer to the ansatz

        Args:
            q (QuantumRegister): Quantum register for the qubits
            circ (QuantumCircuit): The quantum circuit

        Returns:
            (Int): returns 0 if the function ran successfully
        """
        for i in range(self.n_qubits):
            if i < (self.n_qubits - 1):
                circ.cx(i, i+1)
            else:
                circ.cx(i, 0)

        return 0

    def CNOT_linear(self, q, circ):
        """
        Add a linear CNOT layer to the ansatz

        Args:
            q (QuantumRegister): Quantum register for the qubits
            circ (QuantumCircuit): The quantum circuit

        Returns:
            (Int): returns 0 if the function ran successfully
        """
        for i in range(self.n_qubits):
            if i < (self.n_qubits - 1):
                circ.cx(i, i+1)

        return 0
    
    def Ansatz_structure_like_qubo_YZ_2(self, q, circ, params):
        """Generate the ansatz YZ+ZY following the problem structure for qubo problem, but first the single ry layer then the YZ+ZY layer"""
        n_para = 2*len(self.edge_list) + self.n_qubits  ## number of parameters each layer
        self.layer = round( len(params) / n_para ) ## number of layers of ansatz

        for l in range(1, self.layer+1):
            params_l = params[(l-1)*n_para : l*n_para]
            ### first single rotation layer
            self.R_j(q, circ, params_l[0: self.n_qubits])

            ### layer for e^{-i* (theta_0^{ij} * Z_iY_j + theta_1^{ij} * Y_iZ_j)} for the vertex pairs (i, j) in self.edge_list
            for k, (i,j) in enumerate(self.edge_list):
                ### exp{-i/2 ( params_l[2k]*ZiYj + params_l[2k+1]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                
                circ.cx(q[i],q[j])
                circ.ry(params_l[self.n_qubits + 2*k], q[i])
                circ.rz(-params_l[self.n_qubits + 2*k+1], q[j])
                circ.cx(q[i],q[j])
                
                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                circ.barrier(q[:])

    def Ansatz_R_y(self, q, circ, params):
        """generate the Efficient-SU(2) quantum circuit with linear or circular cnot, or parallel_cz
        Args:
        params: np.array of parameters of VQE

        """
        n_layer = int(len(params)/self.n_qubits)
        # print('n_layer: ', n_layer)

        self.R_j(q, circ, params[0: self.n_qubits])
        circ.barrier(q)

        for l in range(1,n_layer):
            self.CNOT_linear(q, circ)
            self.R_j(q, circ, params[ l * self.n_qubits : (l+1) * self.n_qubits ])
            circ.barrier(q)

    def qcircuit(self, params):
        """
        Generate the ansatz circuit for the given parameters and prepare a backend to 
        simulate it on

        Args:
            params (np.array):  Parameters for VQE ansatz

        Returns: 
            circ (QuantumCircuit): The ansatz circuit with a given number of parameters
            backend (AerSimulator): The backend we use for simulation
        """

        # Prepare the circuit and the simulator we use
        backend = Aer.get_backend('aer_simulator')
        q = QuantumRegister(self.n_qubits, name='q')
        circ = QuantumCircuit(q)

        circ.clear()
        circ.h(q[:])
        circ.barrier(q[:])

        if self.ansatz_type == 'R_y':
            self.Ansatz_R_y(q, circ, params)

        elif self.ansatz_type == 'structure_like_qubo_YZ_2':
            self.Ansatz_structure_like_qubo_YZ_2(q, circ, params)
        
        else:
            sys.stderr.write(
                '!!! Wrong ansatz type, can only be SIA or R_y !!!')
            sys.exit()

        return circ, backend
    
    def shot_probs_from_circuit(self, Circ, Backend):

        Circ.measure_all()

        # Execute the circuit and retrieve the results
        job = Backend.run(Circ, shots=self.shots)
        counts_dict = job.result().get_counts(0)  # key bitstring : q_n ....q_0

        prob_list = []
        val_list = []
        poss = 0
        for bitstr, count in counts_dict.items():
            index = int(bitstr, 2)  # convert binary number to decimal
            val_list.append(self.eigen_list[index])
            prob_list.append(count / self.shots)
            if index in self.ground_id_list:
                poss += count / self.shots
        
        return val_list, prob_list, poss
    
    def probs_from_circuit(self, Circ, Backend):

        ### exact simulation with infinate shots
        Circ.save_statevector()
        job = Backend.run(Circ)
        result = job.result()
        outputstate = np.array(result.get_statevector(Circ))   ### amplitude
        prob_list = [abs(outputstate[i]) ** 2 for i in range(len(outputstate))]
        val_list = self.eigen_list
        poss = 0
        for i in self.ground_id_list:
            poss += prob_list[i]
        
        return val_list, prob_list, poss


    def CVaR_expectation(self, params):
        """
        Get the conditional value at risk from the measurement results

        Args:
            params (np.array): The parameters for the ansatz

        Returns:
            cvar (Real): The conditional value at risk
        """

        # Get the circuit and add measurements
        circ, backend = self.qcircuit(params)

        if self.shots:
            val_list, prob_list, poss = self.shot_probs_from_circuit(circ, backend)

        else: 
            val_list, prob_list, poss = self.probs_from_circuit(circ, backend)

        # Get the probability for measuring the ground state
        self.poss_eval.append(poss) ### record the probability of optimal solution in each iteration

        # Get the conditional value at risk and save it
        cvar = self.compute_cvar(prob_list, val_list, self.alpha)
        self.cvar_eval.append(cvar.real) ### record the cvar of each iteration during VQE

        # Get the ratio between the difference of the VQE maximum range the spectrum spans and save it
        r = (cvar.real - self.exp_min) / (self.exp_max - self.exp_min)
        self.r_eval.append(r)

        return cvar.real

    def compute_cvar(self, probabilities, values, alpha):
        """
        Auxilliary method to compute the conditional value at risk.

        Args:
            probabilities (List[Real]): The probabilities for measuring a bit string
            values (List[Real]): The corresponding energy values
            alpha (Real): Confidence level for the conditional value at risk

        Returns:
            cvar (Real): The conditional value at risk
        """
        
        sorted_indices = np.argsort(values)
        probs = np.array(probabilities)[sorted_indices]
        vals = np.array(values)[sorted_indices]

        cvar = 0
        total_prob = 0
        for i, (p, v) in enumerate(zip(probs, vals)):
            if p >= alpha - total_prob:
                p = alpha - total_prob
            total_prob += p
            cvar += p * v
            if abs(total_prob - alpha) < 1e-8:
                break

        cvar /= total_prob

        return cvar

    def call_back(self, params):
        """
        Call back function to monitor the number of iterations and to get some output

        Args:
            params (List[Real]): 
        
        Returns:
            (Bool): Return false
        """

        self.n_iter += 1
        print('cvar: ', self.cvar_eval[-1], '  poss: ', self.poss_eval[-1])

        return False
