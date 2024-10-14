import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [20]

N_r = 100
num_layer = 1
tau_value = 0.3
alpha_value = 0.01
init = 'warm_start_measure'
type_of_ansatz = 'structure_like_qubo_YZ_2'
num_shots = 0
sort_value = [1]
abs_value = [1,0]
inv_value = [1,0]

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
            for sort in sort_value:
                for i in abs_value:
                    for j in inv_value:
                        itemdata.append({"N": str(N), "r": str(r), "alpha": str(alpha_value), "shots": str(num_shots), "ansatz_type": str(type_of_ansatz), "layer": str(num_layer), "tau": str(tau_value), "initialization": str(init), "sorting": str(sort), "absolute": str(i), "invert": str(j)})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
print(itemdata[0]['invert'])

print(type(itemdata[0]['invert']))

# "n_ins": str(n_ins), "r": str(r), "alpha": str(alpha), "shots": str(shots), "ansatz_type": str(ansatz_type), "layer": str(layer), "tau": str(tau), "initialization": str(initialization)



                    #if not sorting:
                        #absolute_list = [False] 
                        #invert_list = [False]
