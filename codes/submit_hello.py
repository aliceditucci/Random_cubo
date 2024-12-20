import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format
import os


job = htcondor.Submit({
    "executable": "job_hello.sh",
    "requirements": 'OpSysAndVer == "AlmaLinux9"',
    "should_transfer_files" : "IF_NEEDED",

    "error" : "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/hello.error",

    "output": "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/hello.out",

    "log" : "/lustre/fs24/group/cqta/atucci/Random_cubo/codes/Logs/hello.log",

    "+RequestRuntime": "172800",
    "request_memory": "10GB",

    "PREEMPTION_REQUIREMENTS": "True",
})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job)
#print(itemdata)
