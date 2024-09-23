import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [6]

N_r = 2
num_layer = 1
tau_value = 1
alpha_value = 0.01
init = 'warm_start_measure'
type_of_ansatz = 'structure_like_qubo_YZ_2'
num_shots = 0

sort_list = [True, False]
absolute_list = [True, False] 
invert_list = [True, False]

job = htcondor.Submit({
    "executable": "job_parallel.sh",
    "arguments": "$(N) $(r) $(alpha) $(shots) $(ansatz_type) $(layer) $(tau) $(initialization) $(sorting) $(absolute) $(invert)",
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
            for sorting in sort_list:
                    
                    if not sorting:
                        absolute_list = [False] 
                        invert_list = [False]

                    for absolute in absolute_list:
                            for invert in invert_list: 
                                itemdata.append({"N": str(N), "r": str(r), "alpha": str(alpha_value), "shots": str(num_shots), "ansatz_type": str(type_of_ansatz), "layer": str(num_layer), "tau": str(tau_value), "initialization":str(init), "sorting": str(sorting), "absolute" : str(absolute), "invert" : str(invert)})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))

# "n_ins": str(n_ins), "r": str(r), "alpha": str(alpha), "shots": str(shots), "ansatz_type": str(ansatz_type), "layer": str(layer), "tau": str(tau), "initialization": str(initialization)

