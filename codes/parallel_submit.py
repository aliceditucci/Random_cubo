import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [22]
N_r = 400
alpha_value = 0.01
num_shots = 0
tau_list = [0.3]
num_layer = 1
graph_type_list = ['3regular', '050', '070', '080', '090', '095', '100', 'complete']
adaptive = 1

job = htcondor.Submit({
    "executable": "job_parallel_entropy.sh",
    "arguments": "$(N) $(r) $(alpha) $(shots) $(tau) $(layer) $(graph_type) $(if_adsorting)",
    "requirements": 'OpSysAndVer == "AlmaLinux9"',
    "should_transfer_files" : "IF_NEEDED",

    "error" : "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).error",

    "output": "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).out",

    "log" : "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).log",

    "+RequestRuntime": "172800",
    "request_memory": "10GB",

    "PREEMPTION_REQUIREMENTS": "True",
})

itemdata = []
for N in N_list: 
        for r in range(N_r):
            for tau_value in tau_list:
                for graph in graph_type_list:
                    itemdata.append({"N": str(N), "r": str(r), "alpha": str(alpha_value), "shots": str(num_shots), "layer": str(num_layer), "tau": str(tau_value), "graph_type": str(graph), "if_adsorting": str(adaptive)})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
#print(itemdata)
