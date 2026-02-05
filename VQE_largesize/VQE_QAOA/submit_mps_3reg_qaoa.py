import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


N_list = [30, 60, 90, 120, 150]
r_list = range(100)
alpha_list = [1.0, 0.01, 0.001]

job = htcondor.Submit({
    "executable": "/lustre/fs23/group/nic/yahuichai/package/miniconda3/envs/qiskit_v1/bin/python",
    "arguments": "run_qaoa.py --N $(N) --r $(r) --alpha $(alpha) --shots $(shots)",
    "output": "/lustre/fs24/group/cqta/yhchai/qubo/VQE_and_QAOA/output/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots).out",
    "error": "/lustre/fs24/group/cqta/yhchai/qubo/VQE_and_QAOA/output/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots).err",
    "log": "/lustre/fs24/group/cqta/yhchai/qubo/VQE_and_QAOA/output/N$(N)_000_r$(r)_alpha$(alpha)_shots$(shots).log",
    "request_cpus": "8",
    "request_memory": "16GB",
    "+JobBatchName": "\"qubo_mps_000_qaoa\"",
})
itemdata = []
for N in N_list:
    for r in r_list:
        for alpha in alpha_list:
            shots = round(100 / alpha)

            itemdata.append({
                "N": str(N),
                "r": str(r),
                "shots": str(shots),
                "alpha": str(alpha)
            })              

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
