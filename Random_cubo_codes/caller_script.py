import subprocess

# Define the range of n_ins values
r_range = range(100)  # This will iterate n_ins from 0 to 10

# Loop over n_ins values

for initialization in ['warm_start_measure_lightcone']:
    for r in r_range:
        # Command to run script.py with current n_ins value
        command = ['py', 'run_qubo.py',  '--shots', str(0),  '--N', str(6), '--initialization', str(initialization), '--r', str(r)]
        
        # Execute the command
        subprocess.run(command)