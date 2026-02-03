#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import *
import sys
import numpy as np
import matplotlib.pyplot as plt


class VQE_and_QAOA:
    def __init__(self, Hamiltonian = None, n_qubits = None, ansatz_type = None, alpha = None, circuit_show = False, shots = False):
        """Args:
            Hamiltonian: PauliSumOp in qiskit
            n_qubits: int, number of qubits
            ansatz_type: string, type of ansatz
            alpha: float, CVaR coefficient
            circuit_show: bool, show circuit or not
            shots: False for exact simulation; int number for finite shots simulation
        """
        
    
        
        if Hamiltonian:
            self.H = Hamiltonian.to_matrix(sparse=True)
            self.eigen_list = self.H.diagonal()
            self.exp_min, self.exp_max, self.ground_id_list = self.Get_minimun_from_H_mat()

        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type  ### linear_cnot, circular_cnot, parallel_cz, sstructure_like_qubo_YZ_2 .。。。
        self.alpha = alpha  ### CVaR coefficient
        
        
        
        self.cvar_eval = []       ### list of the cvar for every evaluation
        self.std_eval = []       ### list of the cvar standard deviation for every evaluation
        self.r_eval = []         ### list of the approxiamation ratio for everay evaluation, r = (exp) / (exp_min)
        self.poss_eval = []      ### list of the possibility of the optimal solution for every evaluation
        
        self.n_iter = 0         ### count the iteration step of the classical optimization
        
        self.edge_list = None   ### edge_list of the graph related to the QUBO Hamiltonian, will decide the order of two-qubit gate in circuit
        self.edge_coeff_dict = None ### {edge: coeff} of qubo problem for Multi-angle QAOA 
        self.circuit_show = circuit_show   ### False or True, To decide if we need save the circuit as a .png file
        
        self.shots = shots   ### number of shots, if it is false, do the exact simulation without shot noise; if it is a certain number, do the measurement with sample
        self.params_print = False
        self.params_record = False
        self.params_eval = []   ### list of the parameters for every evaluation
        
    def Get_minimun_from_H_mat(self):
        exp_list = self.eigen_list
        
        exp_min = min(exp_list)
        exp_max = max(exp_list)
        
        ground_id_list = []
        for i in range(len(exp_list)):
            if abs(exp_list[i] - exp_min) < 1e-8:
                ground_id_list.append(i)
        
        return exp_min.real, exp_max.real, ground_id_list

        
    #region define ansatzes
    def R_j(self, q, circ, param):
        """ Rotation layer consists of single Ry(theta_i, q_i) 
        Args:
            q: QuantumRegister, qubits
            circ: QuantumCircuit
            param: numpy.array or list
        """
        
        if len(param) != self.n_qubits:
            sys.stderr.write('!!! Error of the parameters in R_j !!!')
            sys.exit()
        for i in range(self.n_qubits):
            circ.ry(param[i], q[i])
        return 0
        
    def CNOT_circular(self, q, circ):
        for i in range(self.n_qubits):
            if i < (self.n_qubits - 1):
                circ.cx(i, i+1)
            else:
                circ.cx(i, 0)

        return 0
    
    def CNOT_linear(self, q, circ):
        for i in range(self.n_qubits):
            if i < (self.n_qubits - 1):
                circ.cx(i, i+1)

        return 0
    
    def CZ_parallel(self, q, circ):
        """generate the circuit that has parallel CZ gate, only for the even number qubits"""
        for i in range(0, self.n_qubits-1, 2):
            circ.cz(i, i+1)
        for i in range(1, self.n_qubits-1, 2):
            circ.cz(i, i+1)

        return 0

    def Efficient_su2(self, q, circ, params):
        """generate the Efficient-SU(2) quantum circuit with linear or circular cnot, or parallel_cz
        Args:
        params: np.array of parameters of VQE

        """
        n_layer = int(len(params)/self.n_qubits)
        # print('n_layer: ', n_layer)
        for l in range(n_layer):
         
            self.R_j(q, circ, params[ l * self.n_qubits : (l+1) * self.n_qubits ])
            circ.barrier(q)
            
            if l != n_layer-1:
                if self.ansatz_type == 'circular_cnot':
                    self.CNOT_circular(q, circ)
                elif self.ansatz_type == 'linear_cnot':
                    self.CNOT_linear(q, circ)
                elif self.ansatz_type == 'parallel_cz':
                    self.CZ_parallel(q, circ)
                else:
                    sys.stderr.write('!!! Error, wrong entangle type in Efficient_su2!!!')
                    sys.exit()
            circ.barrier(q)
                
        
        return 0
    
    def Ansatz_single_qubit_rotation(self, q, circ, params):
        """generate the circuit that only has single-qubit rotation Ry gate and T gate
        Args:
        params: np.array of parameters
        
        """
        n_layer = int(len(params)/self.n_qubits)
        for l in range(n_layer):
            self.R_j(q, circ, params[ l * self.n_qubits : (l+1) * self.n_qubits ])
            
            if l != n_layer - 1:
                for i in range(self.n_qubits):
                    circ.t(q[i])
                    
        return 0
    
    def Ansatz_structure_like_qubo_YZ(self, q, circ, params):
        """Generate the ansatz YZ+ZY following the problem structure for qubo problem"""
        
            
        if len(params) != 2*len(self.edge_list) + self.n_qubits:
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()
            
        else:
            for k, (i,j) in enumerate(self.edge_list):
                ### exp{-i/2 ( params[2k]*ZiYj + params[2k+1]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                
                circ.cx(q[i],q[j])
                circ.ry(params[2*k], q[i])
                circ.rz(-params[2*k+1], q[j])
                circ.cx(q[i],q[j])
                
                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                
        self.R_j(q, circ, params[ -self.n_qubits: ])
        
    def Ansatz_structure_like_qubo_YZ_2(self, q, circ, params):
        """Generate the ansatz YZ+ZY following the problem structure for qubo problem, but firstly the single ry layer then the YZ+ZY layer"""
        n_para = 2*len(self.edge_list) + self.n_qubits  ## number of parameters each layer
        self.layer = round( len(params) / n_para ) ## number of layers of ansatz

        for l in range(1, self.layer+1):
            params_l = params[(l-1)*n_para : l*n_para]
            ### firstly single-qubit rotation layer
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
                
    def Ansatz_structure_like_qubo_YX_2(self, q, circ, params):
        """Generate the ansatz YX+XY following the problem structure for qubo problem, but first the single ry layer then the YX+XY layer"""
        self.R_j(q, circ, params[0: self.n_qubits])
            
        if len(params) != 2*len(self.edge_list) + self.n_qubits:
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()
            
        else:
            for k, (i,j) in enumerate(self.edge_list):
                
                ### exp{-i/2 ( params[6k+2]*XiYj + params[6k+3]*YiXj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                circ.rx(-np.pi/2, q[j])

                circ.cx(q[i],q[j])
                circ.rx(params[self.n_qubits + 2*k], q[i])
                circ.rz(-params[self.n_qubits + 2*k+1], q[j])
                circ.cx(q[i],q[j])

                circ.rx(np.pi/2, q[i])
                circ.rx(np.pi/2, q[j])
                circ.rz(np.pi/2, q[j])
                
    def Ansatz_structure_like_qubo_YZ_YX_2(self, q, circ, params):
        """Generate the ansatz YZ+ZY following the problem structure for qubo problem, but first the single ry layer then the YZ+ZY layer"""
        self.R_j(q, circ, params[0: self.n_qubits])
            
        if len(params) != 4*len(self.edge_list) + self.n_qubits:
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()
            
        else:
            for k, (i,j) in enumerate(self.edge_list):
                ### exp{-i/2 ( params[4k]*ZiYj + params[4k+1]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                
                circ.cx(q[i],q[j])
                circ.ry(params[self.n_qubits + 4*k], q[i])
                circ.rz(-params[self.n_qubits + 4*k+1], q[j])
                circ.cx(q[i],q[j])
                
                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                
                ### exp{-i/2 ( params[4k+2]*XiYj + params[4k+3]*YiXj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                circ.rx(-np.pi/2, q[j])

                circ.cx(q[i],q[j])
                circ.rx(params[self.n_qubits + 4*k+2], q[i])
                circ.rz(-params[self.n_qubits + 4*k+3], q[j])
                circ.cx(q[i],q[j])

                circ.rx(np.pi/2, q[i])
                circ.rx(np.pi/2, q[j])
                circ.rz(np.pi/2, q[j])
                 
    def Ansatz_structure_like_qubo_YZ_Y(self, q, circ, params):
        """Generate the ansatz IY, YI, YZ, ZY following the problem structure for qubo problem"""
            
        if len(params) != (4*len(self.edge_list) + self.n_qubits):
            print('len(params): ', len(params))
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()
            
        else:
            self.R_j(q, circ, params[ 0 : self.n_qubits ])
            for k, (i,j) in enumerate(self.edge_list):
                ### exp{-i/2 ( params[2k]*ZiYj + params[2k+1]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                
                circ.cx(q[i],q[j])
                circ.ry(params[self.n_qubits + 4*k], q[i])
                circ.rz(-params[self.n_qubits + 4*k+1], q[j])
                circ.cx(q[i],q[j])
                
                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                
                circ.ry(params[self.n_qubits + 4*k+2], q[i])
                circ.ry(params[self.n_qubits + 4*k+3], q[j])
                
    def Ansatz_structure_like_qubo_YZ_Y_2(self, q, circ, params):
        """Generate the ansatz IY, YI, YZ, ZY following the problem structure for qubo problem"""
       ### change the order of params to match the good initial params
            
        if len(params) != (4*len(self.edge_list) + self.n_qubits):
            print('len(params): ', len(params))
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()
            
        else:
            self.R_j(q, circ, params[ 0 : self.n_qubits ])
            for k, (i,j) in enumerate(self.edge_list):
                #### ry
                circ.ry(params[self.n_qubits + 4*k+0], q[i])
                
                ### exp{-i/2 ( params[4*k+2]*ZiYj + params[4*k+3]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                
                circ.cx(q[i],q[j])
                circ.ry(params[self.n_qubits + 4*k+2], q[i])
                circ.rz(-params[self.n_qubits + 4*k+3], q[j])
                circ.cx(q[i],q[j])
                
                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                
                #### ry
                circ.ry(params[self.n_qubits + 4*k+1], q[j])
                
    def Ansatz_structure_like_qubo(self, q, circ, params):

        """Generate the ansatz following the problem structure for QUBO problem .
        for kth two-body interaction ZiZj in Hamiltonian, the quantum approxiamtion circuit will be : 
        exp{ -i/2 (params[6k]*Yi + params[6k+1]*Yj + params[6k+2]*XiYj + params[6k+3]*YiXj + params[6k+4]*ZiYj + params[6k+5]*YiZj }
        """

        if len(params) != (6*len(self.edge_list)):
            print('len(params): ', len(params))
            sys.stderr.write('\n!!! Error about the parameter number !!!')
            sys.exit()

        else:
            for k, (i,j) in enumerate(self.edge_list):
                ### exp{-i/2 ( params[2k]*ZiYj + params[2k+1]*YiZj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])

                circ.cx(q[i],q[j])
                circ.ry(params[6*k], q[i])
                circ.rz(-params[6*k+1], q[j])
                circ.cx(q[i],q[j])

                circ.rx(np.pi/2, q[i])
                circ.rz(np.pi/2, q[j])
                
                ### exp{-i/2 ( params[6k+2]*XiYj + params[6k+3]*YiXj )}
                circ.rx(-np.pi/2, q[i])
                circ.rz(-np.pi/2, q[j])
                circ.rx(-np.pi/2, q[j])

                circ.cx(q[i],q[j])
                circ.rx(params[6*k+2], q[i])
                circ.rz(-params[6*k+3], q[j])
                circ.cx(q[i],q[j])

                circ.rx(np.pi/2, q[i])
                circ.rx(np.pi/2, q[j])
                circ.rz(np.pi/2, q[j])
                
                ### RY(q_i) RY(q_j)
                circ.ry(params[6*k+4], q[i])
                circ.ry(params[6*k+5], q[j])




        return 0
    
    def QAOA(self, q, circ, params):
        p = int(round(len(params)/2))
        for l in range(p):
            gamma = params[2*l]
            beta = params[2*l+1]
            ## Uc(gamma)
            for edge, coeff in self.edge_coeff_dict.items():
                if len(edge) == 1:
                    i = edge[0]
                    circ.rz(gamma*coeff, q[i])
                elif len(edge) == 2:
                    i = edge[0]
                    j = edge[1]
                    circ.cx(q[i],q[j])
                    circ.rz(gamma*coeff, q[j])
                    circ.cx(q[i],q[j])
                else:
                    sys.stderr.write('!!! Wrong edge!!!')
                    sys.exit()
            ## Ub(beta)
            for i in range(self.n_qubits):
                circ.rx(beta, q[i])

    def Ma_QAOA(self, q, circ, params):
        """Multi-angle QAOA"""
        params_layer = self.n_qubits + len(list(self.edge_coeff_dict.keys()))
        p = int(round(len(params)/params_layer))
        para_id = 0
        for l in range(p):
            ## Uc(gamma)
            for edge, coeff in self.edge_coeff_dict.items():
                para = params[para_id]
                para_id += 1
                if len(edge) == 1:
                    i = edge[0]
                    circ.rz(para*coeff, q[i])
                    #print('edge:{}, para:{}'.format(edge, para))
                elif len(edge) == 2:
                    i = edge[0]
                    j = edge[1]
                    circ.cx(q[i],q[j])
                    circ.rz(para*coeff, q[j])
                    circ.cx(q[i],q[j])
                    #print('edge:{}, para:{}'.format(edge, para))
                else:
                    sys.stderr.write('!!! Wrong edge!!!')
                    sys.exit()
            ## Ub(beta)
            for i in range(self.n_qubits):
                para = params[para_id]
                para_id += 1
                circ.rx(para, q[i])
                # print('i:{}, para:{}'.format(i, para))
                
    #endregion
    

    def Quantum_circuit(self, params):
        """generate the quantum circuit with the input parameter
        Args:
            params: list of parameters in the ansatz
        Return:
            val_list: list of the measured eigen value if having finite shots;
                      list of eigen values of all basis states for exact simulation (self.shots = False)
            prob_list: list of the possibility of corresbonding eigen values in val_list
            poss: float, probability of the optimal solution
        """
        
        backend = Aer.get_backend('aer_simulator')
        q = QuantumRegister(self.n_qubits, name='q')
        circ = QuantumCircuit(q)
        circ.clear()

        circ.h(q[:])
        circ.barrier(q[:])
        ## add quantum circuit to circ
        if (self.ansatz_type == 'linear_cnot') or (self.ansatz_type == 'circular_cnot') or (self.ansatz_type == 'parallel_cz'):
            self.Efficient_su2(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo':
            self.Ansatz_structure_like_qubo(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YZ_Y':
            self.Ansatz_structure_like_qubo_YZ_Y(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YZ_Y_2':
            self.Ansatz_structure_like_qubo_YZ_Y_2(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YZ':
            self.Ansatz_structure_like_qubo_YZ(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YZ_2':
            self.Ansatz_structure_like_qubo_YZ_2(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YX_2':
            self.Ansatz_structure_like_qubo_YX_2(q, circ, params)
            
        elif self.ansatz_type == 'structure_like_qubo_YZ_YX_2':
            self.Ansatz_structure_like_qubo_YZ_YX_2(q, circ, params)

        elif self.ansatz_type == 'single_qubit_rotation':
            self.Ansatz_single_qubit_rotation(q, circ, params)
        elif self.ansatz_type == 'qaoa':
            self.QAOA(q, circ, params)
        elif self.ansatz_type == 'ma_qaoa':
            self.Ma_QAOA(q, circ, params)
        else:
            sys.stderr.write('\n!!! Wrong about the quantum ansatz_type !!!')
            sys.exit()

        if self.circuit_show == True:
            print('\n Show circuit!')
            circ.draw(output='mpl', filename = 'ansatz_{}_circuit_N_{}_layer_{}'.format(self.ansatz_type, self.n_qubits, self.layer))
            
        if self.shots:
            ### with finite shots
            circ.measure_all()
            job = backend.run(circ, shots=self.shots)
            counts_dict = job.result().get_counts(0)  # dict with key being bitstring : {q_n ....q_0:count}

            prob_list = []
            val_list = []
            poss = 0 ### fidelity, possibility of the optimal solution(s) in the current quantum state
            for bitstr, count in counts_dict.items():
                index = int(bitstr, 2)  # convert binary number to decimal
                val_list.append(self.eigen_list[index])
                prob_list.append(count / self.shots)
                if index in self.ground_id_list:
                    poss += count / self.shots
        else:
            ### exact simulation with infinate shots
            circ.save_statevector()
            job = backend.run(circ)
            result = job.result()
            outputstate = np.array(result.get_statevector(circ))   ### amplitude
            prob_list = [abs(outputstate[i]) ** 2 for i in range(len(outputstate))]
            val_list = self.eigen_list
            poss = 0
            for i in self.ground_id_list:
                poss += prob_list[i]
        
        return val_list, prob_list, poss

    

    
########### CVaR expectation
    def CVaR_expectation(self, params):
        """Cost function in VQE
        Args:
            params: list of parameters in ansatz
        Returns:
            cvar: float
        """
        
        val_list, prob_list, poss = self.Quantum_circuit(params)
            
        if self.params_print:
            print('\nfidelity, i.e, poss: ', poss)
            print('params: ', params)
        if self.params_record:
            self.params_eval.append(params)
             
      
        self.poss_eval.append(poss)  ### record the probability of optimal solution in each iteration

        cvar, std = self.compute_cvar(prob_list, val_list, self.alpha)
        self.cvar_eval.append(cvar.real)  ### record the cvar of each iteration during VQE
        self.std_eval.append(std.real)    ### record the standard deviation of the best alpha eigenvalue distribution
        r = (cvar.real) / (self.exp_min)  ### approximation ratio
        self.r_eval.append(r)
        
        return cvar.real
    
    def compute_cvar(self, probabilities, values, alpha):
        """ 
        Auxilliary method to computes CVaR for given probabilities, values, and confidence level.

        Attributes:
        - probabilities: list/array of probabilities
        - values: list/array of corresponding values
        - alpha: confidence level

        Returns:
        - CVaR
        - std: standard deviation of the best alpha eigenvalue distribution
        
        """

        sorted_indices = np.argsort(values)
        probs = np.array(probabilities)[sorted_indices]
        vals = np.array(values)[sorted_indices]

        probs_1 = []  ### pribality of the eigenvalues counted in CVaR
        vals_1 = []   ### eigenvalues counted in CVaR

        cvar = 0
        total_prob = 0
        for i, (p, v) in enumerate(zip(probs, vals)):
            if p >= alpha - total_prob:
                p = alpha - total_prob
            total_prob += p
            cvar += p * v
            probs_1.append(p)
            vals_1.append(v)
            if abs(total_prob - alpha) < 1e-8:
                break
#         print('total_prob: ', total_prob)

        cvar /= total_prob
        probs_1 = np.array(probs_1)/total_prob
        vals_1 = np.array(vals_1)
        # Calculate the weighted variance
        variance = np.average((vals_1- cvar)**2, weights=probs_1)
        if self.shots:
            variance *= total_prob*self.shots/(total_prob*self.shots-1)  ## sampled variance
        std = pow(variance, 0.5)

        return cvar, std
      
    
    def call_back(self, params):
        self.n_iter += 1
        print('\nn_iter: ', self.n_iter)
        print('cvar: ', self.cvar_eval[-1], '  poss: ', self.poss_eval[-1])
        
        if self.ansatz_type == 'classical_ansatz':
            prob_list = self.Classical_circuit(params)
        else:
            prob_list = self.Quantum_circuit(params)
        
        print('n_eval: ', len(self.cvar_eval))
        
        exp_poss_dict = {}
        for i in range(len(self.eigen_list)):
            exp = self.eigen_list[i]
            if exp in exp_poss_dict.keys():
                exp_poss_dict[exp] += prob_list[i]
            else:
                exp_poss_dict[exp] = prob_list[i]
                
        sorted_exp_poss_dict = dict(sorted(exp_poss_dict.items(), key=lambda x:x[0]))
        self.distribution[self.ansatz_type].append(sorted_exp_poss_dict)
                
        
        return False
    
    def call_back_histogram(self, params):
    
        if self.ansatz_type == 'classical_ansatz':
            prob_list = self.Classical_circuit(params)
        
        else:
            prob_list = self.Quantum_circuit(params)
            
        print('n_eval: ', len(self.cvar_eval))
        
        exp_poss_dict = {}
        for i in range(len(self.eigen_list)):
            exp = self.eigen_list[i]
            if exp in exp_poss_dict.keys():
                exp_poss_dict[exp] += prob_list[i]
            else:
                exp_poss_dict[exp] = prob_list[i]
                
        key_list = sorted(exp_poss_dict.keys())
        poss_list = [exp_poss_dict[key] for key in key_list]
        
        exp = self.CVaR_expectation(params)
        
        print('\nexp: ', exp)
        print('\nposs_ground_state: ', poss_list[0])
        plt.figure()
        plt.bar(key_list[:1], poss_list[:1], width=0.05, fc = 'r')
        plt.bar(key_list[1::], poss_list[1::], width=0.05, fc = 'b', alpha=0.3, label = 'E = ' + str(exp))
        plt.title(self.ansatz_type + '_iter_' + str(self.n_iter))
        plt.xlabel('q(x)', fontsize = 20)
        plt.ylabel('Prob(x)', fontsize = 20)
        plt.legend()
        
        plt.savefig(self.figdir + '/Ansatz_' + self.ansatz_type + '_iter_' + str(self.n_iter)+ '.png')
        
        
        self.n_iter += 1
        
        
        return 0
        

