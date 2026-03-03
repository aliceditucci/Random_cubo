import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [120, 150]
r_list = range(100)
alpha_list = [0.01]
ansatz_type_list = ['structure_like_qubo_YZ_2']
initialization_list = ['zeros']


job = htcondor.Submit({
    "executable": "job_parallel.sh",
    "arguments": "$(N) $(r) $(alpha) $(shots) $(ansatz_type) $(initialization)",
    "requirements": 'OpSysAndVer == "AlmaLinux9"',
    "output": "/lustre/fs24/group/cqta/atucci/Random_cubo/VQE_largesize/VQE_QAOA/Logs/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots)_ansatz$(ansatz_type)_init$(initialization).out",
    "error": "/lustre/fs24/group/cqta/atucci/Random_cubo/VQE_largesize/VQE_QAOA/Logs/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots)_ansatz$(ansatz_type)_init$(initialization).err",
    "log": "/lustre/fs24/group/cqta/atucci/Random_cubo/VQE_largesize/VQE_QAOA/Logs/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots)_ansatz$(ansatz_type)_init$(initialization).log",
    "request_cpus": "1",
    "request_memory": "5GB",
    "+RequestRuntime": "432000",
    "+JobBatchName": "\"qubo_ansalikequbo_zeros\"",
    "PREEMPTION_REQUIREMENTS": "True",
})
itemdata = []
for N in N_list:
    for r in r_list:
        for alpha in alpha_list:
            for initialization in initialization_list:
                for ansatz_type in ansatz_type_list:
                    shots = round(100 / alpha)

                    itemdata.append({
                        "N": str(N),
                        "r": str(r),
                        "shots": str(shots),
                        "alpha": str(alpha),
                        "ansatz_type": str(ansatz_type),
                        "initialization": str(initialization)
                    })              

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
