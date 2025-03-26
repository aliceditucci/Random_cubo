import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [40]
N_r = 50
alpha_value = 0.01
num_shots = 10000
tau_list = [0.05, 0.1, 0.2, 0.4]
num_layer = 1
graph_type_list = ['3regular']
adaptive_list = [0,1]
analytic = 1
bond_dimension = 100 
backend = 'matrix_product_state'

job = htcondor.Submit({
    "executable": "job_parallel.sh",
    "arguments": "$(N) $(r) $(alpha) $(shots) $(tau) $(layer) $(graph_type) $(if_adsorting) $(if_analytic)",
    "requirements": 'OpSysAndVer == "AlmaLinux9"',
    "should_transfer_files" : "IF_NEEDED",

    "error" : "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/QUBO_N$(N)_shots$(shots).error",

    "output": "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/QUBO_N$(N)_shots$(shots).out",

    "log" : "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/QUBO_N$(N)_shots$(shots).log",

    "+RequestRuntime": "172800",
    "request_memory": "10GB",

    "PREEMPTION_REQUIREMENTS": "True",
})

itemdata = []
for N in N_list: 
        for r in range(N_r):
            for tau_value in tau_list:
                for graph in graph_type_list:
                    for adaptive in adaptive_list:
                        itemdata.append({"N": str(N), "r": str(r), "alpha": str(alpha_value), "shots": str(num_shots), "layer": str(num_layer), "tau": str(tau_value), "graph_type": str(graph), "if_adsorting": str(adaptive), "if_analytic": str(analytic), "bond": str(bond_dimension), "backend_method": str(backend)})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
#print(itemdata)
#print(f"Job submission data: {itemdata[-1]}")  # Print the last job added