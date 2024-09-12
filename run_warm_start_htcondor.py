import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os

N_list = [16, 20]
r_list = range(100)
ws_method_list = ['single_term', 'lightcone', 'block']
numpara_list = [2, 4, 6]
tau_list = [0.2, 0.4, 0.6, 0.8]




job = htcondor.Submit({
    "executable": "/lustre/fs23/group/nic/yahuichai/package/miniconda3/envs/py37/bin/python3.7",
    "arguments": "run_warm_start.py --N $(N) --r $(r) --ws_method $(ws_method) --numpara $(numpara) --tau $(tau)",
    "output": "/lustre/fs24/group/cqta/yhchai/qubo/output/N$(N)_$(r)_$(ws_method)_$(numpara)_$(tau).out",
    "error": "/lustre/fs24/group/cqta/yhchai/qubo/output/N$(N)_$(r)_$(ws_method)_$(numpara)_$(tau).err",
    "log": "/lustre/fs24/group/cqta/yhchai/qubo/output/N$(N)_$(r)_$(ws_method)_$(numpara)_$(tau).log",
    "request_cpus": "8",
    "request_memory": "32GB",
})
itemdata = []
for N in N_list:
    for r in r_list:
        for ws_method in ws_method_list:
            for numpara in numpara_list:
                for tau in tau_list:
                    itemdata.append({
                        "N": str(N),
                        "r": str(r),
                        "ws_method": ws_method,
                        "numpara": str(numpara),
                        "tau": str(tau),
                    })

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
