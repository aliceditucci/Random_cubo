import subprocess

# Define the range of n_ins values
r_range = range(0,1)  # This will iterate n_ins from 0 to 10

# Loop over n_ins values


for r in r_range:
    # Command to run script.py with current n_ins value
    # command = ['py', 'run_qubo.py',  '--shots', str(0),  '--N', str(12), '--initialization', str(initialization), '--r', str(r)]
    command = ['py', 'run_qubo_shots.py',  '--shots', str(0),  '--N', str(20), '--graph_type', str('complete'), '--if_adsorting', str(1), '--if_analytic', str(0), '--r', str(r),'--tau', str(0.2) ]
    # print(command)


    # command = ['py', 'hello.py']
    # Execute the command
    subprocess.run(command)
