#matplotlib inline
from __future__ import division
import numpy as np
from numpy.random import rand
from numpy import linalg as LA
import time
import sys
from itertools import chain
import os
from numba import jit
#from scipy.optimize import curve_fit

#####
#function for curve fit
#####
def func(x, c, a):
    return (1-a)*np.exp(-c*x) + a

def func2(x, c):
    return np.exp(-c*x)


#####
#jackknife function
#####
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def jackBlocks(original_list, num_of_blocks, length_of_blocks):
    block_list = np.zeros(num_of_blocks)
    length_of_blocks = int(length_of_blocks)
    for i in range(num_of_blocks):
        block_list[i] = (1/length_of_blocks)*np.sum(original_list[i*(length_of_blocks) : (i + 1)*(length_of_blocks)])
    return block_list

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeError(blocks, length_of_blocks):
    #blocks is already O_(B,n)
    blocks = np.array(blocks)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeErrorFromFullList(original_list, num_of_blocks, length_of_blocks):
    blocks = jackBlocks(original_list, num_of_blocks, length_of_blocks)

    #blocks is already O_(B,n)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

########-------------------------------
#Autocorrelation functions
########-------------------------------

def BinningLevel(vals):
    new_vals = np.zeros(len(vals)/2)
    for i in range(len(new_vals)):
        new_vals[i] = (vals[2*i]+vals[2*i+1])/2
    return new_vals


def autocorrelation(data_box, K_max):
    avg_length = len(data_box)
    stack = data_box
    Q_val = np.zeros(K_max)
    for l in range(K_max):
        temp_val = 0.
        num_val = 0
        boolean = True

        while boolean == True:
            l_top = num_val + l
            temp_val += stack[num_val]*stack[l_top]
            
            if l_top == avg_length - 1:
                boolean = False

            num_val += 1

        Q_val[l] = temp_val/num_val

    val_avg = np.sum(stack)/len(stack) 

    #denominator = Q_val[0] - val_avg**2
    #nominator = Q_val - np.ones(len(Q_val))*(val_avg**2)

    #corrData = (1/denominator)*nominator
    return Q_val, val_avg

def autocorrelationBis(data_box, K_max):
    avg_length = len(data_box)
    stack = data_box
    Q_val = np.zeros(K_max)
    Q_err = np.zeros(K_max)
    for l in range(K_max):
        temp_val = []
        num_val = 0
        boolean = True

        while boolean == True:
            l_top = num_val + l
            temp_val.append(stack[num_val]*stack[l_top])
            
            if l_top == avg_length - 1:
                boolean = False

            num_val += 1

        temp_val_avg = np.sum(temp_val)/len(temp_val)
        temp_val_err = np.std(temp_val)

        Q_val[l] = temp_val_avg
        Q_err[l] = temp_val_err


    return Q_val, Q_err

#K : length of the stack array for the values
#avg_length : the number of evaluations taken for the averages of the stack
#error _length : number of times it is computed to get error bars
def mainPartAutocorrelation(data, K, avg_length, error_length):
    
    full_data = []
    average_file = []
    num_of_elements = np.array([(avg_length - i) for i in range(K)])

    for i in range(error_length):
        Q_val, val_avg  = autocorrelation(data[avg_length*i:avg_length*(i+1)], K)
        full_data.append(Q_val) #has all the list of averages Q_t Q_t+tau
        average_file.append(val_avg) #has the average of Q_t
    full_data = np.array(full_data)
    average_file = np.array(average_file)
    
    #Jackknife error analysis
    #see function
    #error on <o> 
    avg_o, error_o = JackknifeError(average_file, error_length)
    #error on <o_i o_i+T>
    avg_ooT = np.zeros(K)
    error_ooT = np.zeros(K)
    for m in range(K):
        avg_ooT[m], error_ooT[m] = JackknifeError(full_data[:,m], num_of_elements[m])

    A_corr_denom = avg_ooT[0] - avg_o**2
    A_corr_nom = avg_ooT - np.ones(K)*(avg_o**2)

    A_corr = (1/(A_corr_denom))*(A_corr_nom) 
    error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K)*(2*np.fabs(avg_o)*error_o)**2)
    error_A_corr = np.fabs(A_corr)*np.sqrt((np.divide(error_A_corr_nom,A_corr_nom))**2 + (np.divide(error_A_corr_denom,A_corr_denom))**2)

    return A_corr, error_A_corr

def mainPartAutocorrelationBis(data, K_max):
    #avg and error on <o_i o_i+T>
    avg_ooT, error_ooT  = autocorrelationBis(data, K_max)
    #print np.absolute(np.divide(error_ooT, avg_ooT))[:6]
    #avg and error on the average of O, <o>
    avg_o = np.sum(data)/len(data)
    error_o = np.std(data)

    #print np.absolute(np.divide(error_o, avg_o))

    A_corr_denom = avg_ooT[0] - avg_o**2
    A_corr_nom = avg_ooT - np.ones(K_max)*(avg_o**2)

    A_corr = (1/(A_corr_denom))*(A_corr_nom) 
    error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    #print A_corr_denom, error_A_corr_denom
    error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K_max)*((2*np.fabs(avg_o)*error_o)**2))
    #error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    #error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K_max)*(2*np.fabs(avg_o)*error_o)**2)
    #print np.absolute(np.divide(error_A_corr_denom, A_corr_denom))
    #print np.absolute(np.divide(error_A_corr_nom, A_corr_nom))[:6]
    error_A_corr = np.fabs(A_corr)*np.sqrt((np.divide(error_A_corr_nom,A_corr_nom))**2 + (np.divide(error_A_corr_denom,A_corr_denom))**2)

    #print A_corr[:6]
    #print error_A_corr[:6]

    return A_corr, error_A_corr


#Need to first extract the zip file
#the format the string files
def main():

    ######
    #-----------------------------------------------------------------------------------------------------------------------
    #######
    #parameters of the code
    ######
    #-----------------------------------------------------------------------------------------------------------------------
    ######

    N=int(sys.argv[1])
    j2 = float(sys.argv[2])
    #j6 = float(sys.argv[3])
    Kc = 0.0
    lambda3 = float(sys.argv[3])  #take lambda from sys input
    j6 = (2.0 - j2)
    #lambda3 = float(import_param[1])
    factor_print = 10000

    ##########
    #when you do it right after computation 
    ##########

    #the directory to take the data from 
    name_pre_dir = 'testJ2={:.2f}'.format(j2) + 'J6={:.2f}'.format(j6) \
    +'Lambda={:.2f}'.format(lambda3) + 'L='+str(int(N)) + 'Kc={:.2f}'.format(Kc)

    name_dir = name_pre_dir

    #folder to put it in
    folder_data_final = name_pre_dir+'finalData'
    if not os.path.exists(folder_data_final):
        os.mkdir(folder_data_final)

    #saving the variables in the other folder
    #the form of variables.data
    #saving_variables_pre = np.array([j2,j6,lambda3, Kc, length_box, number_box, therm])
    #saving_variables = np.append(saving_variables_pre, range_temp)
    
    saving_variables = np.loadtxt('./'+name_dir+'/variables.data')
    number_box = int(saving_variables[5])
    #number_box = 1600
    length_box = int(saving_variables[4])
    #length_box = 10
    range_temp = saving_variables[7:]
    np.savetxt('./'+folder_data_final+'/variables.data', saving_variables)
    #number of temperature steps
    nt = len(range_temp)


    ########
    #Processing the data in order to be plotted
    ########

    """
    #originally came as
    all_dat = np.array([energy, total.real, total.imag, ord6.real, ord6.imag,\
    ord2.real, ord2.imag, tot_sector.real, tot_sector.imag,\
    cm_order.real, cm_order.imag, locking, bond_avg,\
    ord6_xi_m, ord2_xi_m, ordP_xi_m, ordU_xi_m,\
                       H6_tot, I6_tot, H2_tot, I2_tot, Hx_tot, Ix_tot,\
                       vort6, vort2])

    #order of data
    #thermo
    output_thermo = [energy, np.real(total), np.imag(total), np.real(ord6), np.imag(ord6),\
    np.real(ord2), np.imag(ord2), np.real(tot_sector), np.imag(tot_sector),\
    sector_avg, sector_std, np.real(cm_order), np.imag(cm_order), locking, bond_avg,\
    np.real(ord6_xi), np.imag(ord6_xi)]
    output_thermo = [energy, total.real, total.imag, ord6.real, ord6.imag,\
    ord2.real, ord2.imag, tot_sector.real, tot_sector.imag,\
    cm_order.real, cm_order.imag, locking, bond_avg,\
    ord6_xi_m, ord2_xi_m, ordP_xi_m, ordU_xi_m]
    #len of 17
      

    #stiffness
    output_stiff = [H6_tot, I6_tot, H2_tot, I2_tot, Hx_tot, Ix_tot]
    #len of 6

    #vortex
    output_vortex = [vort1, vort2, deviation1/4, deviation2/4, diff_vortices_1, \
    diff_vortices_2, total_on_top, vortFracPhi, overlap, vort_num_theta,\
      min_dist_vort_to_vort, scnd_min_dist_vort_to_vort]
    #len of 12

    """

    print
    print 'Starting Analysis'
    print

    

    #########-----------------------------------------------
    #Measurements for the energy and ~magnetization
    ########------------------------------------------------

    Energy = np.zeros(2*nt)
    SpecificHeat = np.zeros(2*nt)
    OrderCumulant = np.zeros(2*nt)

    number_bins_ene = int(number_box*length_box/20)
    Energy_histo = np.zeros((nt, number_bins_ene))
    Energy_histo_edges = np.zeros((nt, number_bins_ene + 1))

    OrderParam = np.zeros(2*nt)
    OrderParam_BIS = np.zeros(2*nt)
    Susceptibility1 = np.zeros(2*nt)
    Susceptibility2 = np.zeros(2*nt)
    BinderCumulant = np.zeros(2*nt)

    OrderTheta = np.zeros(2*nt)
    OrderTheta_BIS = np.zeros(2*nt)
    BinderTheta = np.zeros(2*nt)
    SusceptibilityTheta1 = np.zeros(2*nt)
    SusceptibilityTheta2 = np.zeros(2*nt)

    OrderPhi = np.zeros(2*nt)
    OrderPhi_BIS = np.zeros(2*nt)
    BinderPhi = np.zeros(2*nt)
    SusceptibilityPhi1 = np.zeros(2*nt)
    SusceptibilityPhi2 = np.zeros(2*nt)

    OrderSigma = np.zeros(2*nt)
    OrderSigma_BIS = np.zeros(2*nt)
    BinderSigma = np.zeros(2*nt)
    SusceptibilitySigma1 = np.zeros(2*nt)
    SusceptibilitySigma2 = np.zeros(2*nt)

    avg_sigma = np.zeros(2*nt)
    std_sigma = np.zeros(2*nt)

    OrderTot = np.zeros(2*nt)
    OrderTot_BIS = np.zeros(2*nt)
    BinderTot = np.zeros(2*nt)
    SusceptibilityTot1 = np.zeros(2*nt)
    SusceptibilityTot2 = np.zeros(2*nt)

    OrderLocking = np.zeros(2*nt)
    OrderLocking_BIS = np.zeros(2*nt)
    BinderLocking = np.zeros(2*nt)
    Susceptibility1Locking = np.zeros(2*nt)
    Susceptibility2Locking = np.zeros(2*nt)

    OrderBondSig = np.zeros(2*nt)
    OrderBondSig_BIS = np.zeros(2*nt)
    BinderBondSig = np.zeros(2*nt)
    Susceptibility1BondSig = np.zeros(2*nt)
    Susceptibility2BondSig = np.zeros(2*nt)

    CorrLength6 = np.zeros(2*nt)
    CorrLength2 = np.zeros(2*nt)
    CorrLengthP = np.zeros(2*nt)
    CorrLengthU = np.zeros(2*nt)

    ## This part runs through the data and creates the errors
    for m in range(nt):
        data = np.loadtxt('./' + name_dir +'/outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        #data = np.loadtxt('outputL='+str(int(N))+'atT='+str(int(range_temp[m]*1000)).zfill(5)+'.data')
        """
        output_thermo = [energy, total.real, total.imag, ord6.real, ord6.imag,\
        ord2.real, ord2.imag, tot_sector.real, tot_sector.imag,\
        cm_order.real, cm_order.imag, locking, bond_avg,\
        ord6_xi_m, ord2_xi_m, ordP_xi_m, ordU_xi_m]
        """
        
        E1_init = np.divide(np.array(data[:,0]), N**2)
        M1_init = np.divide(np.array(data[:,1]) + 1j*np.array(data[:,2]), N**2)
        Mtheta1_init = np.divide(np.array(data[:,3]) + 1j*np.array(data[:,4]), N**2)
        Mphi1_init = np.divide(np.array(data[:,5]) + 1j*np.array(data[:,6]), N**2)

        #this is the tot sector
        sigma_init = np.divide(np.array(data[:,7]) + 1j*np.array(data[:,8]), N**2)

        #sigmaAvg_init = np.divide(np.array(data[:,9]), N**2)
        #sigmaStd_init = np.divide(np.array(data[:,10]), N**2)
        #domain_init = np.divide(np.array(data[:,11]), N**2)  #does not exist anymore, phantom data
        #domain_init = np.zeros()

        #this is the cm order
        tot_init = np.divide(np.array(data[:,9]) + 1j*np.array(data[:,10]), N**2)

        lock_dat = np.divide(np.array(data[:,11]), N**2)
        bond_dat = np.divide(np.array(data[:,12]), N**2)
        #corr_length prep
        ord6_xi = np.divide(np.array(data[:,13]), N**4) 
        ord2_xi = np.divide(np.array(data[:,14]), N**4) 
        ordP_xi = np.divide(np.array(data[:,15]), N**4) 
        ordU_xi = np.divide(np.array(data[:,16]), N**4) 

        ord6_xi_0 = np.divide(np.array(data[:,3])**2 + np.array(data[:,4])**2, N**4) 
        ord2_xi_0 = np.divide(np.array(data[:,5])**2 + np.array(data[:,6])**2, N**4) 
        ordP_xi_0 = np.divide(np.array(data[:,7])**2 + np.array(data[:,8])**2, N**4) 
        ordU_xi_0 = np.divide(np.array(data[:,9])**2 + np.array(data[:,10])**2, N**4) 
        
        #energy
        E1 = E1_init
        E2 = E1*E1
        E4 = E2*E2
        
        #locking variable
        M1_real = np.real(M1_init) # only avg of cos
        M1_imag = np.imag(M1_init) # only avg of cos
        M1_tot = np.absolute(M1_init)
        M2 = np.absolute(M1_init)**2
        M4 = M2*M2

        #theta variable
        Mtheta1_real = np.real(Mtheta1_init) # only avg of cos
        Mtheta1_imag = np.imag(Mtheta1_init) # only avg of sin
        Mtheta_tot = np.absolute(Mtheta1_init)
        Mtheta2 = np.absolute(Mtheta1_init)**2
        Mtheta4 = Mtheta2*Mtheta2

        #phi variable
        Mphi1_real = np.real(Mphi1_init) # only avg of cos
        Mphi1_imag = np.imag(Mphi1_init) # only avg of sin
        Mphi_tot = np.absolute(Mphi1_init)
        Mphi2 = np.absolute(Mphi1_init)**2
        Mphi4 = Mphi2*Mphi2

        #sigma variable
        Sigma_real = np.real(sigma_init)
        Sigma_imag = np.imag(sigma_init)
        Sigma_tot = np.absolute(sigma_init)
        Sigma_2 = np.absolute(sigma_init)**2
        Sigma_4 = Sigma_2*Sigma_2

        #tot order variable
        tot_real = np.real(tot_init)
        tot_imag = np.imag(tot_init)
        tot_tot = np.absolute(tot_init)
        tot_2 = np.absolute(tot_init)**2
        tot_4 = tot_2*tot_2

        #locking order variable
        lock_init = lock_dat
        lock_tot = np.absolute(lock_init)
        lock_2 = np.absolute(lock_init)**2
        lock_4 = lock_2*lock_2

        #bond order variable
        bond_init = bond_dat
        bond_tot = np.absolute(bond_init)
        bond_2 = np.absolute(bond_init)**2
        bond_4 = bond_2*bond_2


        #correlation length are all fine


        #we use a version of the jackknife function that creates the boxes in the function
        E1_avg , E1_error = JackknifeErrorFromFullList(E1, number_box, length_box)
        E2_avg , E2_error = JackknifeErrorFromFullList(E2, number_box, length_box)
        E4_avg , E4_error = JackknifeErrorFromFullList(E4, number_box, length_box)
        #all order
        M1_real_avg , M1_real_error = JackknifeErrorFromFullList(M1_real, number_box, length_box)
        M1_imag_avg , M1_imag_error = JackknifeErrorFromFullList(M1_imag, number_box, length_box)
        M1_avg, M1_error = JackknifeErrorFromFullList(M1_tot, number_box, length_box)
        M2_avg , M2_error = JackknifeErrorFromFullList(M2, number_box, length_box)
        M4_avg , M4_error = JackknifeErrorFromFullList(M4, number_box, length_box)
        #theta order
        Mtheta1_real_avg , Mtheta1_real_error = JackknifeErrorFromFullList(Mtheta1_real, number_box, length_box)
        Mtheta1_imag_avg , Mtheta1_imag_error = JackknifeErrorFromFullList(Mtheta1_imag, number_box, length_box)
        Mtheta_avg, Mtheta_error = JackknifeErrorFromFullList(Mtheta_tot, number_box, length_box)
        Mtheta2_avg , Mtheta2_error = JackknifeErrorFromFullList(Mtheta2, number_box, length_box)
        Mtheta4_avg , Mtheta4_error = JackknifeErrorFromFullList(Mtheta4, number_box, length_box)

        #phi order
        Mphi1_real_avg , Mphi1_real_error = JackknifeErrorFromFullList(Mphi1_real, number_box, length_box)
        Mphi1_imag_avg , Mphi1_imag_error = JackknifeErrorFromFullList(Mphi1_imag, number_box, length_box)
        Mphi_avg, Mphi_error = JackknifeErrorFromFullList(Mphi_tot, number_box, length_box)
        Mphi2_avg , Mphi2_error = JackknifeErrorFromFullList(Mphi2, number_box, length_box)
        Mphi4_avg , Mphi4_error = JackknifeErrorFromFullList(Mphi4, number_box, length_box)
        #sigma
        sigma1_real_avg , sigma1_real_error = JackknifeErrorFromFullList(Sigma_real, number_box, length_box)
        sigma1_imag_avg , sigma1_imag_error = JackknifeErrorFromFullList(Sigma_imag, number_box, length_box)
        sigma_avg, sigma_error = JackknifeErrorFromFullList(Sigma_tot, number_box, length_box)
        sigma2_avg , sigma2_error = JackknifeErrorFromFullList(Sigma_2, number_box, length_box)
        sigma4_avg , sigma4_error = JackknifeErrorFromFullList(Sigma_4, number_box, length_box)
        
        #tot order
        tot1_real_avg , tot1_real_error = JackknifeErrorFromFullList(tot_real, number_box, length_box)
        tot1_imag_avg , tot1_imag_error = JackknifeErrorFromFullList(tot_imag, number_box, length_box)
        tot_avg, tot_error = JackknifeErrorFromFullList(tot_tot, number_box, length_box)
        tot2_avg , tot2_error = JackknifeErrorFromFullList(tot_2, number_box, length_box)
        tot4_avg , tot4_error = JackknifeErrorFromFullList(tot_4, number_box, length_box)

        #lock order
        lock1_real_avg , lock1_real_error = JackknifeErrorFromFullList(lock_init, number_box, length_box)
        lock_avg, lock_error = JackknifeErrorFromFullList(lock_tot, number_box, length_box)
        lock2_avg , lock2_error = JackknifeErrorFromFullList(lock_2, number_box, length_box)
        lock4_avg , lock4_error = JackknifeErrorFromFullList(lock_4, number_box, length_box)

        #bond order
        bond1_real_avg , bond1_real_error = JackknifeErrorFromFullList(bond_init, number_box, length_box)
        bond_avg, bond_error = JackknifeErrorFromFullList(bond_tot, number_box, length_box)
        bond2_avg , bond2_error = JackknifeErrorFromFullList(bond_2, number_box, length_box)
        bond4_avg , bond4_error = JackknifeErrorFromFullList(bond_4, number_box, length_box)

        #correlation length
        ord6_xi_avg, ord6_xi_err = JackknifeErrorFromFullList(ord6_xi, number_box, length_box)
        ord2_xi_avg, ord2_xi_err = JackknifeErrorFromFullList(ord2_xi, number_box, length_box)
        ordP_xi_avg, ordP_xi_err = JackknifeErrorFromFullList(ordP_xi, number_box, length_box)
        ordU_xi_avg, ordU_xi_err = JackknifeErrorFromFullList(ordU_xi, number_box, length_box)
        ord6_xi_0_avg, ord6_xi_0_err = JackknifeErrorFromFullList(ord6_xi_0, number_box, length_box)
        ord2_xi_0_avg, ord2_xi_0_err = JackknifeErrorFromFullList(ord2_xi_0, number_box, length_box)
        ordP_xi_0_avg, ordP_xi_0_err = JackknifeErrorFromFullList(ordP_xi_0, number_box, length_box)
        ordU_xi_0_avg, ordU_xi_0_err = JackknifeErrorFromFullList(ordU_xi_0, number_box, length_box)

        #energy related observables
        #E, Cv
        Energy[m]         = E1_avg
        Energy[nt + m]         = E1_error
        div_sp = (range_temp[m]**2)
        SpecificHeat[m]   = ( E2_avg - E1_avg**2)/div_sp 
        SpecificHeat[nt + m]   = (((E2_error)**2 + (2*E1_error*E1_avg)**2)**(0.5))/div_sp
        ord_cum = (E4_avg)/(E2_avg**2) - 1
        OrderCumulant[m] = ord_cum
        OrderCumulant[nt + m] = np.fabs(ord_cum)*np.sqrt((E4_error/E4_avg)**2 + (2*E2_error/E2_avg)**2)

        #locking related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder
        u_op = M1_real_avg**2 + M1_imag_avg**2
        u_op_err = np.sqrt((2*M1_real_error*M1_real_avg)**2 + (2*M1_imag_error*M1_imag_avg)**2)
        OrderParam[m]  = np.sqrt(u_op)
        OrderParam[nt + m] = 0.5*u_op_err/np.sqrt(u_op)
        OrderParam_BIS[m] = M1_avg
        OrderParam_BIS[nt + m] = M1_error
        Susceptibility1[m] = ( M2_avg - M1_avg**2)/(range_temp[m]);
        Susceptibility1[nt + m] = np.sqrt((M2_error)**2 + (2*M1_avg*M1_error)**2)/(range_temp[m]);
        Susceptibility2[m] = ( M2_avg)/(range_temp[m]);
        Susceptibility2[nt + m] = ( M2_error)/(range_temp[m]);  
        bind_cum = M4_avg/(M2_avg**2)
        BinderCumulant[m] = 1 - bind_cum/3
        BinderCumulant[nt + m] = (1/3)*np.fabs(bind_cum)*np.sqrt((M4_error/M4_avg)**2 + (2*M2_error/M2_avg)**2) 

        #theta related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder
        u_op_theta = Mtheta1_real_avg**2 + Mtheta1_imag_avg**2
        u_op_theta_err = np.sqrt((2*Mtheta1_real_error*Mtheta1_real_avg)**2 + (2*Mtheta1_imag_error*Mtheta1_imag_avg)**2)
        OrderTheta[m]  = np.sqrt(u_op_theta)
        OrderTheta[nt + m] = 0.5*u_op_theta_err/np.sqrt(u_op_theta)
        OrderTheta_BIS[m] = Mtheta_avg
        OrderTheta_BIS[nt + m] = Mtheta_error  
        SusceptibilityTheta1[m] = (Mtheta2_avg - Mtheta_avg**2)/range_temp[m]
        SusceptibilityTheta1[nt + m] = np.sqrt(Mtheta2_error**2 + (2*Mtheta_avg*Mtheta_error)**2)/range_temp[m]
        SusceptibilityTheta2[m] = ( Mtheta2_avg)/(range_temp[m]);
        SusceptibilityTheta2[nt + m] = ( Mtheta2_error)/(range_temp[m]);
        bind_cum_theta = Mtheta4_avg/(Mtheta2_avg**2) 
        BinderTheta[m] = 1 - bind_cum_theta/3
        BinderTheta[nt + m] = (1/3)*np.fabs(bind_cum_theta)*np.sqrt((Mtheta4_error/Mtheta4_avg)**2 + (2*Mtheta2_error/Mtheta2_avg)**2) 

        #phi related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder
        u_op_phi = Mphi1_real_avg**2 + Mphi1_imag_avg**2
        u_op_phi_err = np.sqrt((2*Mphi1_real_error*Mphi1_real_avg)**2 + (2*Mphi1_imag_error*Mphi1_imag_avg)**2)
        OrderPhi[m]  = np.sqrt(u_op_phi)
        OrderPhi[nt + m] = 0.5*u_op_phi_err/np.sqrt(u_op_phi)
        OrderPhi_BIS[m] = Mphi_avg
        OrderPhi_BIS[nt + m] = Mphi_error
        SusceptibilityPhi1[m] = (Mphi2_avg - Mphi_avg**2)/range_temp[m]
        SusceptibilityPhi1[nt + m] = np.sqrt(Mphi2_error**2 + (2*Mphi_avg*Mphi_error)**2)/range_temp[m]
        SusceptibilityPhi2[m] = ( Mphi2_avg)/(range_temp[m]);
        SusceptibilityPhi2[nt + m] = ( Mphi2_error)/(range_temp[m])
        bind_cum_phi = Mphi4_avg/(Mphi2_avg**2) 
        BinderPhi[m] = 1 - bind_cum_phi/3
        BinderPhi[nt + m] = (1/3)*np.fabs(bind_cum_phi)*np.sqrt((Mphi4_error/Mphi4_avg)**2 + (2*Mphi2_error/Mphi2_avg)**2) 

        #sigma related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder       
        sigma_op = sigma1_real_avg**2 + sigma1_imag_avg**2
        sigma_op_err = np.sqrt((2*sigma1_real_error*sigma1_real_avg)**2 \
            + (2*sigma1_imag_avg*sigma1_imag_error)**2)
        OrderSigma[m]  = np.sqrt(sigma_op)
        OrderSigma[nt + m] = 0.5*sigma_op_err/np.sqrt(sigma_op)
        OrderSigma_BIS[m] = sigma_avg
        OrderSigma_BIS[nt + m] = sigma_error
        SusceptibilitySigma1[m] = (sigma2_avg - sigma_avg**2)/range_temp[m];
        SusceptibilitySigma1[nt + m] = np.sqrt((sigma2_error)**2 + (2*sigma_error*sigma_avg)**2)/(range_temp[m]);
        SusceptibilitySigma2[m] = ( sigma2_avg)/(range_temp[m]);
        SusceptibilitySigma2[nt + m] = ( sigma2_error)/(range_temp[m]);
        bind_cum_sig = sigma4_avg/(sigma2_avg**2)
        BinderSigma[m] = 1 - bind_cum_sig/3
        BinderSigma[nt + m] = (1/3)*np.fabs(bind_cum_sig)*np.sqrt((sigma4_error/sigma4_avg)**2 \
            + (2*sigma2_error/sigma2_avg)**2) 

        #extra sigma measurements
        #domain wall  
        #avg_sigma[m] = sigmaAvg_avg
        #avg_sigma[nt + m] = sigmaAvg_error
        #std_sigma[m] = sigmaStd_avg
        #std_sigma[nt + m] = sigmaStd_error

        #total_order (theta + 3 phi) related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder
        tot_op = tot1_real_avg**2 + tot1_imag_avg**2
        tot_op_err = np.sqrt((2*tot1_real_error*tot1_real_avg)**2 \
            + (2*tot1_imag_avg*tot1_imag_error)**2)
        OrderTot[m]  = np.sqrt(tot_op)
        OrderTot[nt + m] = 0.5*tot_op_err/np.sqrt(tot_op)
        OrderTot_BIS[m] = tot_avg
        OrderTot_BIS[nt + m] = tot_error
        SusceptibilityTot1[m] = ( tot2_avg - tot_avg**2)/(range_temp[m]);
        SusceptibilityTot1[nt + m] = np.sqrt((tot2_error)**2 + (2*tot_error*tot_avg)**2)/(range_temp[m]);
        SusceptibilityTot2[m] = ( tot2_avg)/(range_temp[m]);
        SusceptibilityTot2[nt + m] = ( tot2_error)/(range_temp[m]);
        bind_cum_totop = tot4_avg/(tot2_avg**2)
        BinderTot[m] = 1 - bind_cum_totop/3
        BinderTot[nt + m] = (1/3)*np.fabs(bind_cum_totop)*np.sqrt((tot4_error/tot4_avg)**2 \
            + (2*tot2_error/tot2_avg)**2) 

        #the locking observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder       
        lock_op = lock1_real_avg**2 
        lock_op_err = np.sqrt((2*lock1_real_error*lock1_real_avg)**2)
        OrderLocking[m]  = np.sqrt(lock_op)
        OrderLocking[nt + m] = 0.5*lock_op_err/np.sqrt(lock_op)
        OrderLocking_BIS[m] = lock_avg
        OrderLocking_BIS[nt + m] = lock_error
        Susceptibility1Locking[m] = (lock2_avg - lock_avg**2)/range_temp[m];
        Susceptibility1Locking[nt + m] = np.sqrt((lock2_error)**2 + (2*lock_error*lock_avg)**2)/(range_temp[m]);
        Susceptibility2Locking[m] = ( lock2_avg)/(range_temp[m]);
        Susceptibility2Locking[nt + m] = ( lock2_error)/(range_temp[m]);
        bind_cum_lock = lock4_avg/(lock2_avg**2)
        BinderLocking[m] = 1 - bind_cum_lock/3
        BinderLocking[nt + m] = (1/3)*np.fabs(bind_cum_lock)*np.sqrt((lock4_error/lock4_avg)**2 \
            + (2*lock2_error/lock2_avg)**2) 


        #the bond order observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder       
        bond_op = bond1_real_avg**2 
        bond_op_err = np.sqrt((2*bond1_real_error*bond1_real_avg)**2)
        OrderBondSig[m]  = np.sqrt(bond_op)
        OrderBondSig[nt + m] = 0.5*bond_op_err/np.sqrt(bond_op)
        OrderBondSig_BIS[m] = bond_avg
        OrderBondSig_BIS[nt + m] = bond_error
        Susceptibility1BondSig[m] = (bond2_avg - bond_avg**2)/range_temp[m];
        Susceptibility1BondSig[nt + m] = np.sqrt((bond2_error)**2 + (2*bond_error*bond_avg)**2)/(range_temp[m]);
        Susceptibility2BondSig[m] = ( bond2_avg)/(range_temp[m]);
        Susceptibility2BondSig[nt + m] = ( bond2_error)/(range_temp[m]);
        bind_cum_bond = bond4_avg/(bond2_avg**2)
        BinderBondSig[m] = 1 - bind_cum_bond/3
        BinderBondSig[nt + m] = (1/3)*np.fabs(bind_cum_bond)*np.sqrt((bond4_error/bond4_avg)**2 \
            + (2*bond2_error/bond2_avg)**2) 

        #######
        #put the correlation length
        #######
        #corr length
        fact_sin = (1/(2*np.sin(np.pi/N))**2)
        val_6_c = ((ord6_xi_0_avg/ord6_xi_avg) -1)
        CorrLength6[m] = fact_sin*val_6_c
        CorrLength6[nt + m] = fact_sin*(np.sqrt((ord6_xi_0_err/ord6_xi_0_avg)**2 \
            + (ord6_xi_err/ord6_xi_avg)**2))
        val_2_c = ((ord2_xi_0_avg/ord2_xi_avg) -1)
        CorrLength2[m] = fact_sin*val_2_c
        CorrLength2[nt + m] = fact_sin*(np.sqrt((ord2_xi_0_err/ord2_xi_0_avg)**2 \
            + (ord2_xi_err/ord2_xi_avg)**2))
        #cm order
        val_U_c = ((ordU_xi_0_avg/ordU_xi_avg) -1)
        CorrLengthU[m] = fact_sin*val_U_c
        CorrLengthU[nt + m] = fact_sin*(np.sqrt((ordU_xi_0_err/ordU_xi_0_avg)**2 \
            + (ordU_xi_err/ordU_xi_avg)**2))
        #sigma tot
        val_P_c = ((ordP_xi_0_avg/ordP_xi_avg) -1)
        CorrLengthP[m] = fact_sin*val_P_c
        CorrLengthP[nt + m] = fact_sin*(np.sqrt((ordP_xi_0_err/ordP_xi_0_avg)**2 \
            + (ordP_xi_err/ordP_xi_avg)**2))        


        ########
        #do a temperature histogram
        #######
        min_energy = np.min(E1)
        max_energy = np.max(E1)
        bound_low = min_energy + 0.01*(max_energy - min_energy)
        bound_high = max_energy - 0.01*(max_energy - min_energy)
        ener_histo = np.histogram(E1, number_bins_ene, range = (bound_low, bound_high))
        Energy_histo[m] = ener_histo[0]
        Energy_histo_edges[m] = ener_histo[1]

    np.savetxt('./'+ folder_data_final +'/histo_output.data',
        np.c_[Energy_histo])
    np.savetxt('./'+ folder_data_final +'/histo_edges_output.data',
        np.c_[Energy_histo_edges])



    #np.savetxt('./testF=1L='+str(int(N))+'finalData'+'/thermo_outputL='+str(int(N))+'.data',
    #    np.c_[Energy, SpecificHeat, OrderCumulant, OrderParam, Susceptibility, BinderCumulant, \
    #    Order6, Susc6,Binder6, Order2,  Susc2, Binder2, MomentumNullHexatic, MomentumNullNematic])

    np.savetxt('./'+ folder_data_final +'/thermo_output.data',
        np.c_[Energy, SpecificHeat, OrderCumulant,\
        OrderParam, OrderParam_BIS, Susceptibility1, Susceptibility2, BinderCumulant,\
        OrderTheta, OrderTheta_BIS, SusceptibilityTheta1, SusceptibilityTheta2, BinderTheta,\
        OrderPhi, OrderPhi_BIS, SusceptibilityPhi1, SusceptibilityPhi2, BinderPhi,\
        OrderSigma, OrderSigma_BIS, SusceptibilitySigma1, SusceptibilitySigma2, BinderSigma,\
        OrderTot, OrderTot_BIS, SusceptibilityTot1, SusceptibilityTot2, BinderTot,\
        avg_sigma, std_sigma,\
        OrderLocking, OrderLocking_BIS, Susceptibility1Locking, Susceptibility2Locking, BinderLocking,\
        OrderBondSig, OrderBondSig_BIS, Susceptibility1BondSig, Susceptibility2BondSig, BinderBondSig,\
        CorrLength6, CorrLength2, CorrLengthU, CorrLengthP])

    print
    print 'Done with Order/Thermo Analysis'
    print



    #########-----------------------------------------------
    #Measurements for the Stiffness
    ########------------------------------------------------

    RhoTheta = np.zeros(2*nt)
    RhoPhi = np.zeros(2*nt)
    RhoTot = np.zeros(2*nt)

    fourthOrderTheta = np.zeros(2*nt)
    fourthOrderPhi = np.zeros(2*nt)
    fourthOrderTot = np.zeros(2*nt)
    fourthOrderTotScaled = np.zeros(2*nt)


    ## This part runs through the data and creates the errors
    for m in range(nt):
        data = np.loadtxt('./'+ name_dir +'/stiffnessPreDataatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        #data = np.loadtxt('stiffnessPreDataL='+str(int(N))+'atT='+str(int(range_temp[m]*1000)).zfill(5)+'.data')
        Stiff_6_H = data[:,0]/N**2
        Stiff_6_I = data[:,1]/N**2
        Stiff_6_I2 = Stiff_6_I*Stiff_6_I
        Stiff_6_I4 = Stiff_6_I2*Stiff_6_I2
        Stiff_2_H = data[:,2]/N**2
        Stiff_2_I = data[:,3]/N**2
        Stiff_2_I2 = Stiff_2_I*Stiff_2_I
        Stiff_2_I4 = Stiff_2_I2*Stiff_2_I2
        Stiff_tot_H = data[:,4]/N**2
        Stiff_tot_I = data[:,5]/N**2
        Stiff_tot_I2 = Stiff_tot_I*Stiff_tot_I
        Stiff_tot_I4 = Stiff_tot_I2*Stiff_tot_I2

        #we use a version of the jackknife function that creates the boxes in the function
        Stiff_6_H_avg , Stiff_6_H_error = JackknifeErrorFromFullList(Stiff_6_H, number_box, length_box)
        Stiff_6_I_avg , Stiff_6_I_error = JackknifeErrorFromFullList(Stiff_6_I, number_box, length_box)
        Stiff_6_I2_avg , Stiff_6_I2_error = JackknifeErrorFromFullList(Stiff_6_I2, number_box, length_box)
        Stiff_6_I4_avg , Stiff_6_I4_error = JackknifeErrorFromFullList(Stiff_6_I4, number_box, length_box)
        
        Stiff_2_H_avg , Stiff_2_H_error = JackknifeErrorFromFullList(Stiff_2_H, number_box, length_box)
        Stiff_2_I_avg , Stiff_2_I_error = JackknifeErrorFromFullList(Stiff_2_I, number_box, length_box)
        Stiff_2_I2_avg , Stiff_2_I2_error = JackknifeErrorFromFullList(Stiff_2_I2, number_box, length_box)
        Stiff_2_I4_avg , Stiff_2_I4_error = JackknifeErrorFromFullList(Stiff_2_I4, number_box, length_box)
        
        Stiff_tot_H_avg , Stiff_tot_H_error = JackknifeErrorFromFullList(Stiff_tot_H, number_box, length_box)
        Stiff_tot_I_avg , Stiff_tot_I_error = JackknifeErrorFromFullList(Stiff_tot_I, number_box, length_box)
        Stiff_tot_I2_avg , Stiff_tot_I2_error = JackknifeErrorFromFullList(Stiff_tot_I2, number_box, length_box)
        Stiff_tot_I4_avg , Stiff_tot_I4_error = JackknifeErrorFromFullList(Stiff_tot_I4, number_box, length_box)

        T = range_temp[m]    
        RhoTheta[m] = Stiff_6_H_avg - (N**2/T)*(Stiff_6_I2_avg - Stiff_6_I_avg**2)
        RhoTheta[nt + m] = np.sqrt(Stiff_6_H_error**2 + ((N**2)*Stiff_6_I2_error/T)**2 + ((N**2)*2*Stiff_6_I_error*Stiff_6_I_avg/T)**2 )
        RhoPhi[m] = Stiff_2_H_avg - (N**2/T)*(Stiff_2_I2_avg - Stiff_2_I_avg**2)
        RhoPhi[nt + m] = np.sqrt(Stiff_2_H_error**2 + ((N**2)*Stiff_2_I2_error/T)**2 + ((N**2)*2*Stiff_2_I_error*Stiff_2_I_avg/T)**2 )
        RhoTot[m] = Stiff_tot_H_avg - (N**2/T)*(Stiff_tot_I2_avg - Stiff_tot_I_avg**2)
        RhoTot[nt + m] = np.sqrt(Stiff_tot_H_error**2 + ((N**2)*Stiff_tot_I2_error/T)**2 + ((N**2)*2*Stiff_tot_I_error*Stiff_tot_I_avg/T)**2 )

        #avg <(Y-<Y>)^2> = <Y^2> - <Y>^2
        list_Ysq_6 = (Stiff_6_H -  (N**2/T)*(Stiff_6_I2 - Stiff_6_I**2))**2
        list_Ysq_6_avg, list_Ysq_6_error = JackknifeErrorFromFullList(list_Ysq_6, number_box, length_box)
        list_Ysq_2 = (Stiff_2_H -  (N**2/T)*(Stiff_2_I2 - Stiff_2_I**2))**2
        list_Ysq_2_avg, list_Ysq_2_error = JackknifeErrorFromFullList(list_Ysq_2, number_box, length_box)
        list_Ysq_tot = (Stiff_tot_H - (N**2/T)*(Stiff_tot_I2 - Stiff_tot_I**2))**2
        list_Ysq_tot_avg, list_Ysq_tot_error = JackknifeErrorFromFullList(list_Ysq_tot, number_box, length_box)

        #here we compute <L**2 Y_4>
        fourthOrderTheta[m] = (-1)*4*RhoTheta[m] + 3*(Stiff_6_H_avg - (N**2/T)*(list_Ysq_6_avg - RhoTheta[m]**2)) \
        + 2*(N**6/(T**3))*Stiff_6_I4_avg
        fourthOrderPhi[m] = (-1)*4*RhoPhi[m] + 3*(Stiff_2_H_avg - (N**2/T)*(list_Ysq_2_avg - RhoPhi[m]**2)) \
        + 2*(N**6/(T**3))*Stiff_2_I4_avg
        fourthOrderTot[m] = (-1)*4*RhoTot[m] + 3*(Stiff_tot_H_avg\
            - (N**2/T)*(list_Ysq_tot_avg - RhoTot[m]**2)) + 2*(N**6/(T**3))*Stiff_tot_I4_avg
        fourthOrderTotScaled[m] = (-1)*4*RhoTot[m]/(9*j6 + j2) + 3*(Stiff_tot_H_avg/(9*j6 + j2)\
            - (N**2/T)*(list_Ysq_tot_avg/(9*j6 + j2) - (RhoTot[m]/(9*j6 + j2))**2)) + 2*(N**6/(T**3))*Stiff_tot_I4_avg/((3*j6 + j2)**4)
        #print(fourthOrderTot[m])


    np.savetxt('./'+ folder_data_final +'/STIFF_thermo_output.data',
        np.c_[RhoTheta, RhoPhi, RhoTot, fourthOrderTheta, fourthOrderPhi, fourthOrderTot, fourthOrderTotScaled])

    print
    print 'Done with Stiffness Analysis'
    print


    #########-----------------------------------------------
    #Measurements for the Vorticity
    ########------------------------------------------------

    VorticityTheta = np.zeros(2*nt)
    VorticityPhi = np.zeros(2*nt)
    DeviationTheta = np.zeros(2*nt)
    DeviationPhi = np.zeros(2*nt)
    diffTheta = np.zeros(2*nt)
    diffPhi = np.zeros(2*nt)
    diff_of_vort_Theta = np.zeros(2*nt)
    diff_of_vort_Phi = np.zeros(2*nt)
    vort_on_top = np.zeros(2*nt)
    
    fracVortexPhi = np.zeros(2*nt)
    overlap_fct = np.zeros(2*nt)
    number_plus_theta_vort = np.zeros(2*nt)
    dist_to_nn_theta_vort = np.zeros(2*nt)
    dist_to_next_nn_theta_vort = np.zeros(2*nt)
    avg_radius_vort = np.zeros(2*nt)
    instance_mult_vort = np.zeros(2*nt)

    #vorticity measurements
    for m in range(nt):
        data = np.loadtxt('./'+ name_dir +'/VorticityDataatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        Vort6 = data[:,0]
        Vort2 = data[:,1]
        #we use a version of the jackknife function that creates the boxes in the function
        Vort6_avg , Vort6_error = JackknifeErrorFromFullList(Vort6, number_box, length_box)
        Vort2_avg , Vort2_error = JackknifeErrorFromFullList(Vort2, number_box, length_box)
        VorticityTheta[m] = Vort6_avg/N**2
        VorticityTheta[nt + m] = Vort6_error/N**2
        VorticityPhi[m] = Vort2_avg/N**2
        VorticityPhi[nt + m] = Vort2_error/N**2

        """

        Deviation6 = data[:,2]
        Deviation2 = data[:,3]
        diff6 = data[:,4]
        diff2 = data[:,5]
        vot = data[:,6]
        ff2 = data[:,7]

        overlaps = data[:,8]
        num_theta_v = data[:,9]
        dd_nn_theta = data[:,10]
        dd_next_nn_theta = data[:,11]
        length_side = data[:,12]
        inst_vort_in = data[:,13]

        #we use a version of the jackknife function that creates the boxes in the function
        Vort6_avg , Vort6_error = JackknifeErrorFromFullList(Vort6, number_box, length_box)
        Vort2_avg , Vort2_error = JackknifeErrorFromFullList(Vort2, number_box, length_box)
        Deviation6_avg , Deviation6_error = JackknifeErrorFromFullList(Deviation6, number_box, length_box)
        Deviation2_avg , Deviation2_error = JackknifeErrorFromFullList(Deviation2, number_box, length_box)
        diff6_avg, diff6_error = JackknifeErrorFromFullList(diff6, number_box, length_box)
        diff2_avg, diff2_error = JackknifeErrorFromFullList(diff2, number_box, length_box)
        vot_avg, vot_error = JackknifeErrorFromFullList(vot, number_box, length_box)
        ff2_avg, ff2_error = JackknifeErrorFromFullList(ff2, number_box, length_box)
        ovlap_avg, ovlap_error = JackknifeErrorFromFullList(overlaps, number_box, length_box)
        num_theta_v_avg, num_theta_v_error = JackknifeErrorFromFullList(num_theta_v, number_box, length_box)
        dd_nn_theta_avg, dd_nn_theta_error = JackknifeErrorFromFullList(dd_nn_theta, number_box, length_box)
        dd_next_nn_theta_avg, dd_next_nn_theta_error = JackknifeErrorFromFullList(dd_next_nn_theta, number_box, length_box)
        length_side_avg, length_side_error = JackknifeErrorFromFullList(length_side, number_box, length_box)
        instance_mult_vort_avg, instance_mult_vort_error = JackknifeErrorFromFullList(inst_vort_in, number_box, length_box)
    
        VorticityTheta[m] = Vort6_avg/N**2
        VorticityTheta[nt + m] = Vort6_error/N**2
        VorticityPhi[m] = Vort2_avg/N**2
        VorticityPhi[nt + m] = Vort2_error/N**2

        DeviationTheta[m] = Deviation6_avg/N**2
        DeviationTheta[nt + m] = Deviation6_error/N**2
        DeviationPhi[m] = Deviation2_avg/N**2
        DeviationPhi[nt + m] = Deviation2_error/N**2

        diffTheta[m] = Deviation6_avg/N**2 - Vort6_avg/N**2 
        diffTheta[nt + m] = np.sqrt(Deviation6_error**2 + Vort6_error**2)/N**2
        diffPhi[m] = Deviation2_avg/N**2 - Vort2_avg/N**2 
        diffPhi[nt + m] = np.sqrt(Deviation2_error**2 + Vort2_error**2)/N**2

        diff_of_vort_Theta[m] = diff6_avg/N**2
        diff_of_vort_Theta[nt + m] = diff6_error/N**2
        diff_of_vort_Phi[m] = diff2_avg/N**2
        diff_of_vort_Phi[nt + m] = diff2_error/N**2

        vort_on_top[m] = vot_avg/N**2
        vort_on_top[m + nt] = vot_error/N**2

        fracVortexPhi[m] = ff2_avg/N**2
        fracVortexPhi[nt + m] = ff2_error/N**2

        overlap_fct[m] = ovlap_avg
        overlap_fct[nt + m] = ovlap_error

        number_plus_theta_vort[m] = num_theta_v_avg/N**2
        number_plus_theta_vort[nt + m] =  num_theta_v_error/N**2

        #the next ones don't have a N**2 factor because they are just lengths
        dist_to_nn_theta_vort[m] = dd_nn_theta_avg
        dist_to_nn_theta_vort[nt + m] = dd_nn_theta_error
        dist_to_next_nn_theta_vort[m] = dd_next_nn_theta_avg
        dist_to_next_nn_theta_vort[nt + m] = dd_next_nn_theta_error

        avg_radius_vort[m] = length_side_avg
        avg_radius_vort[nt + m] = length_side_error

        instance_mult_vort[m] = instance_mult_vort_avg
        instance_mult_vort[nt + m] = instance_mult_vort_error
        """

    #np.savetxt('./'+ folder_data_final +'/Vorticity_thermo_output.data',
    #    np.c_[VorticityTheta, VorticityPhi, DeviationTheta, DeviationPhi, diffTheta, diffPhi, fracVortexPhi])
    np.savetxt('./'+ folder_data_final +'/Vorticity_thermo_output.data',
        np.c_[VorticityTheta, VorticityPhi, DeviationTheta, DeviationPhi,\
         diffTheta, diffPhi, diff_of_vort_Theta, diff_of_vort_Phi, vort_on_top,\
         fracVortexPhi, overlap_fct, number_plus_theta_vort, dist_to_nn_theta_vort,\
         dist_to_next_nn_theta_vort, instance_mult_vort])

    print
    print 'Done with Vorticity Analysis'
    print

    

    ###########
    #Measurement of the autocorrelation
    ###########

    ####
    #length of autocor analysis
    ####
    K_max = length_box
    K_val = np.arange(K_max)

    for m in range(nt):
        data_1 = np.loadtxt('./' + name_dir +'/outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        AutoCorrEnergy = np.zeros(2*K_max)
        AutoCorrM2 = np.zeros(2*K_max)
        AutoCorrM6 = np.zeros(2*K_max)
        AutoCorrMrel = np.zeros(2*K_max)

        #energy is 0
        #m6 is 3,4
        #m2 is 5,6
        #mrel is 7,8

        energy_to_autocorr = np.array(data_1[:,0])
        auto_ene, auto_ene_err = mainPartAutocorrelationBis(np.divide(energy_to_autocorr, N**2), K_max)
        AutoCorrEnergy = np.concatenate((auto_ene, auto_ene_err))
  
        m2_to_autocorr = np.absolute(np.array(data_1[:,5]) + 1j*np.array(data_1[:,6]))
        auto_m2, auto_m2_err = mainPartAutocorrelationBis(np.divide(m2_to_autocorr, N**2), K_max)
        AutoCorrM2 = np.concatenate((auto_m2, auto_m2_err))

        m6_to_autocorr = np.absolute(np.array(data_1[:,3]) + 1j*np.array(data_1[:,4]))
        auto_m6, auto_m6_err = mainPartAutocorrelationBis(np.divide(m6_to_autocorr, N**2), K_max)
        AutoCorrM6 = np.concatenate((auto_m6, auto_m6_err))

        mrel_to_autocorr = np.absolute(np.array(data_1[:,7]) + 1j*np.array(data_1[:,8]))
        auto_mrel, auto_mrel_err = mainPartAutocorrelationBis(np.divide(mrel_to_autocorr, N**2), K_max)
        AutoCorrMrel = np.concatenate((auto_mrel, auto_mrel_err))

        ##save it
        np.savetxt('./'+ folder_data_final +'/Autocorr_outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data',
            np.c_[AutoCorrEnergy, AutoCorrM2, AutoCorrM6, AutoCorrMrel])

    print
    print 'Done with Autocorrelation Analysis'
    print


    #####
    #move config to here
    #####
    for m in range(nt):
        data_1 = np.loadtxt('./' + name_dir +'/configatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        np.savetxt('./'+ folder_data_final +'/configatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data',data_1)        


    
#------------------------------------------------
#the executable part
#------------------------------------------------

if __name__ == '__main__':

    main()
