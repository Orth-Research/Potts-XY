#matplotlib inline
from __future__ import division, print_function
import numpy as np
from numpy.random import rand
import time
import sys
import os
#import zipfile
#from memory_profiler import profile
from joblib import Parallel, delayed
import math
from numpy import pi as pi
from numpy import cos as cos
from numpy import sin as sin
from numpy import exp as exp
from numpy import mod as mod
from numpy.random import randint as randint
from numpy import absolute as absolute
from functions_mcstep3 import ModifiedWolffLayeredClusterSize as ModifiedWolffLayeredClusterSize
from functions_mcstep3 import ModifiedWolffLayered as ModifiedWolffLayered
from functions_mcstep3 import EnergyCalc as EnergyCalc
from functions_mcstep3 import MeasureConfigNumba as MeasureConfigNumba
import gc

def ModifiedWolffLayeredFunc(config_init, temp, N, j2, j6, lambda3, neighbors_list, niter):

    #the config
    config = config_init.copy()

    #mc step
    avg_size_clust = ModifiedWolffLayeredClusterSize(config, temp, N, j2, j6, lambda3, neighbors_list, niter)

    return [config, avg_size_clust]

#for the serial calculation
#a simple parallelization: 
#throw this function to the core and run temps in parallel - don't talk to each other 
def PTTstepMeasure(config_init, temp, N, j2, j6, lambda3, neighbors_list, niter):

    #the config
    config = config_init.copy()

    ModifiedWolffLayered(config, temp, N, j2, j6, lambda3, neighbors_list, niter)
    data_thermo = MeasureConfigNumba(config, N, j2, j6, lambda3)

    return [config, data_thermo[0], data_thermo]

def PTTstepTherm(config_init, temp, N, j2, j6, lambda3, neighbors_list, niter):

    #the config
    config = config_init.copy()

    ModifiedWolffLayered(config, temp, N, j2, j6, lambda3, neighbors_list, niter)
    energy = EnergyCalc(config, N, j2, j6, lambda3)

    return [config, energy]

##############-------------------------------------------------------------
#Main function that runs the MC code
##############-------------------------------------------------------------

#@profile
def main():

    N = int(sys.argv[1])  #note that N is in fact L the linear size
    j2 = float(sys.argv[2])
    Kc = 0.0
    lambda3 = float(sys.argv[3])
    factor_print = 10000

    j6 = (2.0 - j2) # Ferromagnetic interaction on the 2nd layer

    L_size = N

    nt = int(sys.argv[4])
    num_cores = int(sys.argv[5])
    Tmin = float(sys.argv[6])
    Tmax = float(sys.argv[7])

    #number of steps
    length_box = 20 # number of MC steps in each bin (both during measurement and during thermalization period)    
    number_box = 5*10**3 # // number of MC bins during which measurement is applied 
    therm = 5*10**3 # number of MC bins during thermalization time

    #beta range
    #then figure out T range
    #but the adaptative step will be on beta
    Beta_min = 1/Tmax
    Beta_max = 1/Tmin

    #the list of temperatures and the list of energies, initialized
    ratio_T = (Tmax/Tmin)**(1/(nt-1))
    #range_temp = np.zeros(nt)
    #for i in range(nt):
    #    range_temp[i]=Tmin*((ratio_T)**(i))
    #list_temps = range_temp
    list_temps = np.linspace(Tmin, Tmax, nt)
    list_energies = np.zeros(nt)    

    ######
    #initialize list of neighbors
    ######

    neighbors_list = np.zeros(4*2*(N**2))
    sizeN = N**2
    for i in range(2*(N**2)):
        if i < sizeN:
            layer_add = sizeN
            layer = 0
        else:
            layer_add = -sizeN
            layer = sizeN
        vec_nn_x = [-1, 1, 0, 0, 0]
        vec_nn_y = [0,  0, -1,1, 0]
        vec_nn_z = [0,  0 ,0, 0, layer_add]

        site_studied_1 = i//N
        site_studied_2 = i%N
        for p in range(4):
            neighbors_list[4*i + p] = (layer + N*mod((site_studied_1 + vec_nn_x[p]),N) + mod((site_studied_2 + vec_nn_y[p]),N))
        #neighbors_list[5*i + 4] = ((i + sizeN)%(2*sizeN))

    neighbors_list = np.array(list(map(int, neighbors_list)))
    #print neighbors_list

    #########
    #initialize folder
    ##########

    #see if the folder exists. if it does not, create one
    name_dir = 'testJ2={:.2f}'.format(j2) + 'J6={:.2f}'.format(j6) \
    +'Lambda={:.2f}'.format(lambda3) + 'L='+str(int(N)) + 'Kc={:.2f}'.format(Kc)

    #z = zipfile.ZipFile(name_dir + ".zip", "w")
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)

    #print the initial parameters
    print() 
    print('Linear size of the system L=' + str(N))
    print('Interaction strength:')
    print('J2 = ' + str(j2))
    print('J6 = ' + str(j6))
    print('lambda = ' + str(lambda3))
    print('Kc = ' + str(Kc))
    print()
    print('From temperature Tmax='+ str(Tmax)+' to Tmin='+str(Tmin))
    print('In '+str(nt)+' steps')
    print()
    print('Size of bins:' + str(length_box))
    #print 'N_inter' + str(N_inter)
    print('Number of thermalization bins:' + str(therm))
    print('Number of measurement bins:' + str(number_box))
    print('number of Cores' + str(num_cores))
    print()

    #initializing the configurations 
    config_start = []
    for q in range(nt):
        config_start.append(2*pi*rand(2*N**2))
    config_start = np.array(config_start)

    #important definitions for parallel tempering

    indices_temp = [i for i in range(nt)] #pt_TtoE
    pt_TtoE = indices_temp
    indices_ensemble = [i for i in range(nt)] #pt_EtoT
    pt_EtoT = indices_ensemble


    print('starting the initialization step')
    print()

    start = time.time()

    print('list of temperatures')
    print(list_temps)

    #first run in order to get some estimate of the energy
    #use pt in this run
    print('start with pre therm')
    config_at_T = config_start #the initial config of config_at_T is defined

    #######-------
    #Monte Carlo step 
    #######------------
    #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
    niters = (2*N*N)*np.ones(nt)
    niters = np.array(list(map(int, niters)))
    list_avg_clus_size = np.zeros(nt)

    with Parallel(n_jobs=num_cores, max_nbytes = '10M') as parallel:
        #######-------
        #Monte Carlo step 
        #######------------

        #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
        resultsPreTherm = parallel(delayed(ModifiedWolffLayeredFunc)(config_init = config_at_T[m], \
            temp = list_temps[m], N= N, j2 = j2, j6 = j6, lambda3 = lambda3, \
            neighbors_list = neighbors_list, niter = niters[m]) for m in range(nt))

        #energy_start = []
        for q in range(nt):
            list_avg_clus_size[q] = resultsPreTherm[q][1]
            config_at_T[q] = resultsPreTherm[q][0]

        gc.collect()


    niters = (2*N**2)*np.ones(nt)/list_avg_clus_size
    niters = np.array(list(map(int, niters)))
    print('new number of iteration per temp')
    print(niters)    

    end = time.time()

    print()
    print('done with initialization in '+ str((end - start) ) + ' secs')
    print()



    #saving the variables of the computation
    saving_variables_pre = np.array([j2,j6,lambda3, Kc, length_box, number_box, therm])
    saving_variables = np.append(saving_variables_pre, list_temps)
    np.savetxt('./'+ name_dir +'/variables.data', saving_variables)
    #np.savetxt('./'+ name_dir +'variables.data', saving_variables)

    #-----------------
    #Prep for the Therm + Measure using Parallel Tempering
    #------------


    #list of tuples for the parallel tempering
    tuples_1 = [indices_temp[i:i + 2] for i in range(0, len(indices_temp), 2)] #odd switch #len of nt/2
    tuples_2 = [indices_temp[i:i + 2] for i in range(1, len(indices_temp) - 1, 2)] #even switch #len of nt/2 -1 
    tuples_tot = [tuples_1, tuples_2]  
    half_length = int(nt/2)
    len_tuples_tot = [half_length, half_length - 1]

    ###-------------------------------------------------------------------
    #main program
    #does the MC steps (Metropolis and Wolff) + parallel tempering
    #then measures the energy/mag and spin stiffness
    ###-------------------------------------------------------------------
    #we already have an initial config as config_start
    #and we have a list of initial energies as energy_start

    #we want to keep the measured quantities for mc_data_len steps
    #number_of_parallel_it = length_box*number_box
    mc_data_len = length_box*number_box
    #we want to thermalize the system for therm steps
    length = therm*(length_box)

    ###------------------------------------------------------------------
    #start the thermalization
    ###------------------------------------------------------------------


    print()
    print('Starting the thermalization')
    print()
    start = time.time()

    #swap even pairs or not: initiate at 0
    swap_even_pairs = 0
    #the therm procedure
    #opening single threads and not destroying them
    #note that length box

    with Parallel(n_jobs=num_cores, max_nbytes = '5M') as parallel:
        for il in range(therm):
            for jl in range(length_box):
                #the - period in the range of length_box is to account for the 'period' steps in optimization part
                #print('gone to  ' + str(int(il)) + ' ' + str(int(jl)))

                #######-------
                #Monte Carlo step 
                #######------------

                #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
                resultsTherm = parallel(delayed(PTTstepTherm)(config_init = config_at_T[m], \
                    temp = list_temps[pt_EtoT[m]], N= N, j2 = j2, j6 = j6, lambda3 = lambda3, \
                    neighbors_list = neighbors_list, niter = niters[m]) for m in range(nt))
                #resultsTherm = parallel(delayed(ModifiedWolffLayeredFunc2)(config_init = config_at_T[m], \
                #    temp_init = list_temps[pt_EtoT[m]], N= N, j2 = j2, j6 = j6, lambda3 = lambda3, niter = niters[m]) for m in range(nt))
                for q in range(nt):
                    list_energies[q] = resultsTherm[q][1]
                    config_at_T[q] = resultsTherm[q][0]

                #####---------
                #The Parallel Tempering Step
                #####----------

                #tuples to use
                tuples_used = tuples_tot[swap_even_pairs]
                for sw in range(len_tuples_tot[swap_even_pairs]):
                    index_i = tuples_used[sw][0]
                    index_j = tuples_used[sw][1]
                    initial_i_temp = list_temps[index_i]
                    initial_j_temp = list_temps[index_j]
                    index_energy_i = pt_TtoE[index_i]
                    index_energy_j = pt_TtoE[index_j]

                    Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
                    if Delta_ij > 0:
                        pt_TtoE[index_i] = index_energy_j
                        pt_TtoE[index_j] = index_energy_i  
                        pt_EtoT[index_energy_i] = index_j
                        pt_EtoT[index_energy_j] = index_i
                    else:                  
                        if rand() < exp(Delta_ij):
                            pt_TtoE[index_i] = index_energy_j
                            pt_TtoE[index_j] = index_energy_i  
                            pt_EtoT[index_energy_i] = index_j
                            pt_EtoT[index_energy_j] = index_i

                #change the pair swapper for next run
                swap_even_pairs = (1 - swap_even_pairs)
                gc.collect()

    end = time.time()
    #done with thermalization
    print()
    print('Done with thermalization')
    print('in '+str(end - start)+' seconds')
    print('number of steps ' + str(length))
    print('time per step (full PT + Metro)' + str((end - start)/(length) ))
    print('time per step per temp (full PT + Metro)' + str((end - start)/(length*nt) ))
    print()

       

    start = time.time() 

    ###------------------------------------------------------------------
    #start the measurements
    ###------------------------------------------------------------------

    

    #the data sets:
    all_data_thermo = np.zeros((nt,number_box*length_box, 17))
    all_data_stiff = np.zeros((nt,number_box*length_box, 6))
    all_data_vort = np.zeros((nt,number_box*length_box, 2))

    #swap even pairs or not: initiate at 0
    swap_even_pairs = 0
    #the therm procedure
    #opening single threads and not destroying them

    with Parallel(n_jobs=num_cores, max_nbytes = '5M') as parallel:
        for il in range(number_box):
            for jl in range(length_box):

                                #the - period in the range of length_box is to account for the 'period' steps in optimization part

                #######-------
                #Monte Carlo step 
                #######------------

                #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
                resultsMeasure = parallel(delayed(PTTstepMeasure)(config_init = config_at_T[m], \
                    temp = list_temps[pt_EtoT[m]], N= N, j2 = j2, j6 = j6, lambda3 = lambda3, \
                    neighbors_list = neighbors_list, niter = niters[m]) for m in range(nt))
                for q in range(nt):
                    list_energies[q] = resultsMeasure[q][1]
                    config_at_T[q] = resultsMeasure[q][0]

                #####---------
                #The Parallel Tempering Step
                #####----------

                #tuples to use
                tuples_used = tuples_tot[swap_even_pairs]
                for sw in range(len_tuples_tot[swap_even_pairs]):
                    index_i = tuples_used[sw][0]
                    index_j = tuples_used[sw][1]
                    initial_i_temp = list_temps[index_i]
                    initial_j_temp = list_temps[index_j]
                    index_energy_i = pt_TtoE[index_i]
                    index_energy_j = pt_TtoE[index_j]

                    Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
                    if Delta_ij > 0:
                        pt_TtoE[index_i] = index_energy_j
                        pt_TtoE[index_j] = index_energy_i  
                        pt_EtoT[index_energy_i] = index_j
                        pt_EtoT[index_energy_j] = index_i
                    else:                  
                        if rand() < exp(Delta_ij):
                            pt_TtoE[index_i] = index_energy_j
                            pt_TtoE[index_j] = index_energy_i  
                            pt_EtoT[index_energy_i] = index_j
                            pt_EtoT[index_energy_j] = index_i

                #change the pair swapper for next run
                swap_even_pairs = (1 - swap_even_pairs)

                #reading the data and saving it
                ind = int(length_box*il + jl)
                #note that I want a given column of these data set to a distinct temperature, so I use pt_TtoE in there.
                
                for q in range(nt):                
                    all_data_thermo[q][ind] = resultsMeasure[pt_TtoE[q]][2][0:17]
                    all_data_vort[q][ind] = resultsMeasure[pt_TtoE[q]][2][23:25]
                    all_data_stiff[q][ind] = resultsMeasure[pt_TtoE[q]][2][17:23]


    end = time.time()
    #done with measurements
    print()  
    print('Done with measurements')
    print('in '+str(end - start)+' seconds')
    print('number of PT steps ' + str(mc_data_len))
    print('time per step (full PT + Metro)' + str((end - start)/(mc_data_len) ))
    print('time per step per temp (full PT + Metro)' +str((end - start)/(mc_data_len*nt) )  )
    print ()

    ############
    #-------------------------
    ############
    #Exporting the data
    ############
    #------------------------
    ############

    #export data
    #define folder

    for q in range(nt):
        temp_init = list_temps[q]
        np.savetxt('./'+ name_dir +'/configatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',config_at_T[pt_TtoE[q]])
        np.savetxt('./'+ name_dir +'/outputatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',all_data_thermo[q])
        np.savetxt('./'+ name_dir +'/stiffnessPreDataatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',all_data_stiff[q])
        np.savetxt('./'+ name_dir +'/VorticityDataatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',all_data_vort[q])

       
    print() 
    print('Done with exporting data')
    

#------------------------------------------------
#the executable part
#------------------------------------------------

if __name__ == '__main__':

    main()
