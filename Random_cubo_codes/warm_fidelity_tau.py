import numpy as np 
import itertools
import matplotlib.pyplot as plt
import pickle
import sys

#FUNCTION
def partition_N(n):
    '''do the partition of a complete graph'''
    indexs = range(n)
    pairs_all = []

    swap_even = [i + pow(-1, i) for i in range(n)]

    swap_odd = [0]
    swap_odd.extend([i + pow(-1, i+1) for i in range(1,n-1)])
    swap_odd.append(n-1)

    pairs_even = [(i, i+1) for i in range(0, n, 2)]
    indexs = np.array(indexs)[swap_even]   ### indexs after swap even
    #     print('\nindexs after swap {}: {}'.format(0, indexs))
    pairs_all.append(pairs_even)
    for i in range(1, n):
        if (i%2)==1:
            pair_odd = [(indexs[i], indexs[i+1]) for i in range(1, n-1, 2)]
            pairs_all.append(pair_odd)
            indexs = np.array(indexs)[swap_odd]   ### indexs after swap even
    #             print('\nindexs after swap {}: {}'.format(i, indexs))

        elif (i%2)==0:
            pair_even = [(indexs[i], indexs[i+1]) for i in range(0, n-1, 2)]
            pairs_all.append(pair_even)
            indexs = np.array(indexs)[swap_even]   ### indexs after swap even
    #             print('\nindexs after swap {}: {}'.format(i, indexs))

    return pairs_all



num_variables_list = [str(num).zfill(3) for num in range(6,14, 2)]

N_ins = 100 ###100 50 number of random instances

# initialization =  'warm_start_measure' #'random' or 'zeros' or 'warm_start_measure'

ansatz_type = 'structure_like_qubo_YZ_2' #'R_y' or 'structure_like_qubo_YZ_2'

layer = 1

shots = None

alpha = 0.01

dir_0 = './data' + '/ansatz_type_{}/shots_{}'\
                        .format(ansatz_type,  shots)


y_qubits = []

fig4, ax4 = plt.subplots()

tau_list = [0.1, 0.2]

for tau in tau_list:

    print('tau', tau)

    for num_variables in num_variables_list:
        
        n_qubits = int(num_variables)
        print('num qubits', n_qubits)

        pairs_all = list(itertools.chain.from_iterable(partition_N(n_qubits)))
        num_pairs = len(pairs_all)
        num_params = (n_qubits + 2*num_pairs) * layer

        # Create separate figures outside the inner loop
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        y_qubits = []

        for initialization in ['warm_start_measure_lightcone']:#['warm_start_measure', 'warm_start_measure_lightcone' ]:

            y = []

            dir_name =  dir_0 + '/num_variables_{}/params_{}_layer_{}/alpha_{}/initial_{}'\
                                .format(num_variables, num_params, layer, alpha, initialization)
            
            for r in range(N_ins):
                    
                file_dir_name = dir_name + '/r_{}'\
                                .format(r)
                
                if initialization ==  'warm_start_measure':
                    file_path = file_dir_name + '/tau_{}.pkl'\
                                                .format(tau)
                    # Open the .pkl file in read mode
                    with open(file_path, 'rb') as f:
                        # Load the content of the file using pickle.load()
                        data = pickle.load(f)            
                    fidelity = list(data['layers_exp_poss_dict']['l_1'].items())[0][1]

                if initialization == 'warm_start_measure_lightcone':
                    file_path_lightcone = file_dir_name + '/tau_{}.pkl'\
                                                        .format(tau)
                    
                    # Open the .pkl file in read mode
                    with open(file_path_lightcone, 'rb') as f:
                        # Load the content of the file using pickle.load()
                        data_lightcone = pickle.load(f)            
                    fidelity = list(data_lightcone['exp_poss_dict'].items())[0][1]

                else:
                    sys.stderr.write('something is wrong with the lightcones')
                    sys.exit()
                
                # print('fidelity', fidelity)

                # x.append(n_qubits)
                y.append(fidelity)

            x = np.arange(N_ins)
            # print('x', x)

            y = np.array(y)

            # ax1.scatter(x, y, marker='o', label = initialization)

            # # plt.plot( range(6,26,2), y2, '--', label = '1/2^N' , color = 'blue')
            # # plt.yscale('log')
            # ax1.set_title(f'Warm start fidelity, {num_variables} qubits')
            # ax1.set_xlabel('instance')
            # ax1.set_ylabel('fidelity')

            # # Display the legend
            # # plt.legend()
            # ax1.legend()
            
            # specific_dir_name =  dir_0 + '/num_variables_{}'\
            #                     .format(num_variables, num_params, layer, alpha, initialization)
            # # Save the plot
            # fig1.savefig(specific_dir_name + '/warm_fidelity_tau{}.png'.format(tau))
            # # plt.close()  # Close the figure

        
            y2 = np.sort(y)
            print(y2)

            # plt.figure()
            # plt.scatter(x, y, marker='o', label = initialization)

            # # plt.plot( range(6,26,2), y2, '--', label = '1/2^N' , color = 'blue')
            # # plt.yscale('log')
            # plt.title(f'Warm start fidelity, {num_variables} qubits')
            # plt.xlabel('instance')
            # plt.ylabel('fidelity')

            # # Display the legend
            # plt.legend()

            # ax2.scatter(x, y2, marker='o', label = initialization)

            # # plt.plot( range(6,26,2), y2, '--', label = '1/2^N' , color = 'blue')
            # # plt.yscale('log')
            # ax2.set_title(f'Warm start fidelity, {num_variables} qubits')
            # ax2.set_xlabel('instance')
            # ax2.set_ylabel('fidelity')

            # # Display the legend
            # # plt.legend()
            # ax2.legend()
            

            # specific_dir_name =  dir_0 + '/num_variables_{}'\
            #                     .format(num_variables)
            # # Save the plot
            # fig2.savefig(specific_dir_name + '/warm_fidelity_sorted_tau{}.png'.format(tau))
            # plt.close()  # Close the figure

            # if initialization == 'warm_start_measure' :
            #     sorted_indices = np.argsort(y)                   #SORTED_2  

            # y3 = y[sorted_indices]

            # ax3.scatter(x, y3, marker='o', label = initialization)

            # # plt.plot( range(6,26,2), y2, '--', label = '1/2^N' , color = 'blue')
            # # plt.yscale('log')
            # ax3.set_title(f'Warm start fidelity, {num_variables} qubits')
            # ax3.set_xlabel('instance')
            # ax3.set_ylabel('fidelity')

            # # Display the legend
            # # plt.legend()
            # ax3.legend()
            
            
            # specific_dir_name =  dir_0 + '/num_variables_{}'\
            #                     .format(num_variables, num_params, layer, alpha, initialization)
            # # Save the plot
            # fig3.savefig(specific_dir_name + '/warm_fidelity_sorted_2_tau{}.png'.format(tau))
            # plt.close()  # Close the figure

            
            y4 = [1/(2**float(num_variables)) for X in x]
            # print('y4', y4)
            
            # print('x axis', x + 50*(float(num_variables)-6))

            ax4.scatter(x + 50*(float(num_variables)-6), y2, marker='o', label = tau)

            ax4.plot( x + 50*(float(num_variables)-6), y4, '--' , color = 'blue')
            ax4.set_yscale('log')
            ax4.set_title(f'Warm start fidelity tau vs tau, all qubits')
            ax4.set_xlabel('N qubits')
            # ax4.set_ylabel('fidelity')
            ax4.set_ylabel('fidelity log')

            # Display the legend
            # plt.legend()
            #ax4.legend()
            
            # Save the plot
            fig4.savefig(dir_0 + '/warm_fidelity_sorted_2_qubits_log_taus.png')
            #fig4.savefig(dir_0 + '/warm_fidelity_sorted_2_qubits_taus.png')
            #fig4.savefig(dir_0 + '/warm_fidelity_sorted_2_qubits_taus_oldwarm.png')
            # plt.close()  # Close the figure
        
        # Close the figures after the loop
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

plt.close(fig4)
        
