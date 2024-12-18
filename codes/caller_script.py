import subprocess

# Define the range of n_ins values
r_range = range(100,101)  # This will iterate n_ins from 0 to 10

# Loop over n_ins values

for i in range(1):
    for r in r_range:
        # Command to run script.py with current n_ins value
        # command = ['py', 'run_qubo.py',  '--shots', str(0),  '--N', str(12), '--initialization', str(initialization), '--r', str(r)]
        command = ['py', 'run_qubo_shots.py',  '--shots', str(1000),  '--N', str(6), '--graph_type', str('3regular'), '--if_adsorting', str(1), '--r', str(100)]
        # print(command)


        # command = ['py', 'hello.py']
        # Execute the command
        subprocess.run(command)
