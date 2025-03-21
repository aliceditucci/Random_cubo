import subprocess

# Define the range of n_ins values
r_range = range(100,101)  # This will iterate n_ins from 0 to 10

# Loop over n_ins values

for initialization in ['warm_start_measure_lightcone']:
    for r in r_range:
        # Command to run script.py with current n_ins value
        # command = ['py', 'run_qubo.py',  '--shots', str(0),  '--N', str(12), '--initialization', str(initialization), '--r', str(r)]
        command = ['py', 'run_qubo_mps.py',  '--shots', str(10000),  '--N', str(40), '--graph_type', str('3regular'), '--if_adsorting', str(0), '--backend_method', str('matrix_product_state'), '--bond', str(100), '--if_analytic', str(1), '--alpha', str(0.001)]
        
        # Execute the command
        subprocess.run(command)
