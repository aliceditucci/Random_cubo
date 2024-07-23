import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [12]

N_r = 100
num_layer = 1
tau_value = 0.3
alpha_value = 0.01
init = 'warm_start_measure_lightcone'
type_of_ansatz = 'structure_like_qubo_YZ_2'
num_shots = 0

job = htcondor.Submit({
    "executable": "job_parallel.sh",
    "arguments": "$(N) $(r) $(alpha) $(shots) $(ansatz_type) $(layer) $(tau) $(initialization)",
    "requirements": 'OpSysAndVer == "AlmaLinux9"',
    "should_transfer_files" : "IF_NEEDED",

    "error" : "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).error",

    "output": "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).out",

    "log" : "/lustre/fs24/group/cqta/atucci/Random_cubo/Random_cubo_codes/Logs/QUBO_N$(N)_shots$(shots).log",

    "+RequestRuntime": "432000",
    "request_memory": "128GB",

    "PREEMPTION_REQUIREMENTS": "True",
})

itemdata = []
for N in N_list: 
        for r in range(N_r):
            itemdata.append({"N": str(N), "r": str(r), "alpha": str(alpha_value), "shots": str(num_shots), "ansatz_type": str(type_of_ansatz), "layer": str(num_layer), "tau": str(tau_value), "initialization":str(init)})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))

# "n_ins": str(n_ins), "r": str(r), "alpha": str(alpha), "shots": str(shots), "ansatz_type": str(ansatz_type), "layer": str(layer), "tau": str(tau), "initialization": str(initialization)

