#matplotlib inline
from __future__ import division
import numpy as np
from numpy.random import rand
from numpy import pi as pi
from numpy import cos as cos
from numpy import sin as sin
from numpy import exp as exp
from numpy import mod as mod
from numpy.random import randint as randint
from numpy import absolute as absolute
from numba import jit


######
#-----------------------------------------------------------------------------------------------------------------------
#######
#functions for the Metropolis and Wolff algorithm
######
#-----------------------------------------------------------------------------------------------------------------------
######


#one step of the Wolff algorithm
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def ModifiedWolffLayeredClusterSize(config, temp, N, j2, j6, lambda3, neighbors_list, niter):
    beta=1./temp
    
    #do wolff steps as long as the total cluster size flipped
    # is smaller than the size of the system
    size = N*N
    numItTot = 2*(size)
    #initialize outputs
    avg_size_clust = 0.

    cluster = np.zeros(numItTot, dtype = np.int8)
    listQ = np.zeros(numItTot + 1, dtype = np.int64)

    half_niter = int(np.ceil(niter/2))

    for nn in range(niter):
        #cluster = np.zeros(numItTot, dtype = np.int8)
        #listQ = np.zeros(numItTot + 1, dtype = np.int64)
        init = randint(numItTot)
        listQ[0] = init + 1
        theta_rand = (np.pi)*rand()   #angle of p*pi, here p is 3
        random_angle = [3*theta_rand, theta_rand]
        
        cluster[init] = 1 #this site is in the cluster now
        sc_in = 0
        sc_to_add = 1

        #accept or not clust in avg
        cond_clust = nn//half_niter
        
        while listQ[sc_in] != 0:
            site_studied = listQ[sc_in] + (-1)
            sc_in += 1
            avg_size_clust += 1*(cond_clust)
            if site_studied < N*N:
                l = 0
                l3 = 1
                intJ = j6

            else:
                l = 1
                l3 = 0
                intJ = j2
            int1 = (4*l3 - 3)
            int2 = (1 - 4*l3)
                
            prime_layer_rand = random_angle[l]
            site_angle=config[site_studied]  #find the current angle of the studied site
            config[site_studied] = (2*prime_layer_rand - site_angle) #site selected has been flipped
 
            for kk in range(4):
                site_nn = neighbors_list[4*site_studied + kk]
                near_angle = config[site_nn]
                if cluster[site_nn] == 0:
                    energy_difference = (-1)*intJ*(cos(site_angle - near_angle) - cos(site_angle - (2*prime_layer_rand - near_angle)))
                    freezProb_next = 1. - exp(beta*energy_difference)
                    if (rand() < freezProb_next):
                        #listQ.append(site_nn)
                        listQ[sc_to_add] = site_nn + (1)                    
                        cluster[site_nn] = 1
                        sc_to_add += 1

            #altSite = neighbors_list[5*site_studied + 4]
            altSite = (site_studied + size)%(2*size)
            layerang=config[altSite]
            if cluster[altSite] == 0: 
                energy_difference = (-1)*lambda3*(cos(int1*site_angle + int2*layerang) - cos(int1*site_angle + int2*(2*random_angle[l3] - layerang)))
                freezProb_next_A = 1. - exp(beta*energy_difference)
                if (rand() < freezProb_next_A):
                    listQ[sc_to_add] = altSite + (1)
                    cluster[altSite] = 1
                    sc_to_add += 1

        listQ[:] = 0
        cluster[:] = 0

    #average size cluster
    avg_size_clust = avg_size_clust/half_niter    

    return avg_size_clust



###################
#----------------------------------
###################
#Function for the measure part

#one step of the Wolff algorithm
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def ModifiedWolffLayered(config, temp, N, j2, j6, lambda3, neighbors_list, niter):
    beta=1./temp
    
    #do wolff steps as long as the total cluster size flipped
    # is smaller than the size of the system
    size = N*N
    numItTot = 2*(size)
    #initialize outputs
    #avg_size_clust = 0.

    cluster = np.zeros(numItTot, dtype = np.int8)
    listQ = np.zeros(numItTot + 1, dtype = np.int64)

    #half_niter = int(np.ceil(niter/2))

    for nn in range(niter):
        #cluster = np.zeros(numItTot, dtype = np.int8)
        #listQ = np.zeros(numItTot + 1, dtype = np.int64)
        init = randint(numItTot)
        listQ[0] = init + 1
        theta_rand = (np.pi)*rand()   #angle of p*pi, here p is 3
        random_angle = [3*theta_rand, theta_rand]
        
        cluster[init] = 1 #this site is in the cluster now
        sc_in = 0
        sc_to_add = 1

        #accept or not clust in avg
        #cond_clust = nn//half_niter
        
        while listQ[sc_in] != 0:
            site_studied = listQ[sc_in] + (-1)
            sc_in += 1
            #avg_size_clust += 1*(cond_clust)
            if site_studied < N*N:
                l = 0
                l3 = 1
                intJ = j6

            else:
                l = 1
                l3 = 0
                intJ = j2
            int1 = (4*l3 - 3)
            int2 = (1 - 4*l3)
                
            prime_layer_rand = random_angle[l]
            site_angle=config[site_studied]  #find the current angle of the studied site
            config[site_studied] = (2*prime_layer_rand - site_angle) #site selected has been flipped
 
            for kk in range(4):
                site_nn = neighbors_list[4*site_studied + kk]
                near_angle = config[site_nn]
                if cluster[site_nn] == 0:
                    energy_difference = (-1)*intJ*(cos(site_angle - near_angle) - cos(site_angle - (2*prime_layer_rand - near_angle)))
                    freezProb_next = 1. - exp(beta*energy_difference)
                    if (rand() < freezProb_next):
                        #listQ.append(site_nn)
                        listQ[sc_to_add] = site_nn + (1)                    
                        cluster[site_nn] = 1
                        sc_to_add += 1

            #altSite = neighbors_list[5*site_studied + 4]
            altSite = (site_studied + size)%(numItTot)
            layerang=config[altSite]
            if cluster[altSite] == 0: 
                energy_difference = (-1)*lambda3*(cos(int1*site_angle + int2*layerang) - cos(int1*site_angle + int2*(2*random_angle[l3] - layerang)))
                freezProb_next_A = 1. - exp(beta*energy_difference)
                if (rand() < freezProb_next_A):
                    listQ[sc_to_add] = altSite + (1)
                    cluster[altSite] = 1
                    sc_to_add += 1

        listQ[:] = 0
        cluster[:] = 0

#one step of the Wolff algorithm
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def EnergyCalc(config, N, j2, j6, lambda3):
    energy = 0.
    #calculate the energy
    ####
    for i in range(N):
        for j in range(N):
            latt1 = config[0 + N*i + j]
            latt1shiftX = config[0 + N*(i-1) + j]
            latt1shiftY = config[0 + N*i + j-1]
            latt2 = config[N*N + N*i + j]
            latt2shiftX = config[N*N + N*(i-1) + j]
            latt2shiftY = config[N*N + N*i + j-1]
            energy += (-1.0)*(j6*(cos(latt1+(-1)*latt1shiftX) + \
                                  cos(latt1+(-1)*latt1shiftY))+ \
                              j2*(cos(latt2+(-1)*latt2shiftX) +\
                                  cos(latt2+(-1)*latt2shiftY) )+ \
                              lambda3*cos(latt1 - 3*latt2))
    return energy


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def MeasureConfigNumba(config, N, j2, j6, lambda3):
    #######
    #all calculations ----------
    #######  
    tpi = 2*pi

    config_re = config
    config_re = config_re.reshape((2,N,N))
    latt1 = config_re[0]
    latt2 = config_re[1]
    mod_latt1 = mod(latt1, tpi)
    mod_latt2_f3 = 3*mod(latt2, tpi)
    rel_angles = np.divide(mod_latt1 - mod_latt2_f3, 3)
    center_of_mass = np.divide(mod_latt1 + mod_latt2_f3,2) 
 

    energy = 0.
    H6_tot = 0.
    I6_tot = 0.
    H2_tot = 0.
    I2_tot = 0.
    #total
    total = 0.
    #bond average
    bond_avg = 0.
    #relative order term
    tot_sector = 0.
    #ord per layer
    ord6 = 0.
    ord2 = 0.
    #center of mass term
    cm_order = 0.
    #average of locking
    locking = 0.
    
    #hexatic correlation length
    ord6_xi_x = 0.
    ord6_xi_y = 0.
    #hexatic correlation length
    ord2_xi_x = 0.
    ord2_xi_y = 0.
    #potts correlation length
    ordP_xi_x = 0.
    ordP_xi_y = 0.
    #U(1) correlation length
    ordU_xi_x = 0.
    ordU_xi_y = 0.
    
    #some vortex in there
    vort6 = 0.
    vort2 = 0.
    
    for i in range(N):
        for j in range(N):
            platt1 = config_re[0,i,j]
            platt1shiftX = config_re[0,i-1,j]
            platt1shiftY = config_re[0,i,j-1]
            platt1shiftXshiftY = config_re[0,i-1,j-1]
            platt2 = config_re[1,i,j]
            platt2shiftX = config_re[1,i-1,j]
            platt2shiftY = config_re[1,i,j-1]
            platt2shiftXshiftY = config_re[1,i-1,j-1]
            vcos6 = cos(platt1+(-1)*platt1shiftX)
            vcos2 = cos(platt2+(-1)*platt2shiftX)
            energy += (-1.0)*(j6*(vcos6 + \
                                  cos(platt1+(-1)*platt1shiftY))+ \
                              j2*(vcos2 +\
                                  cos(platt2+(-1)*platt2shiftY) )+ \
                              lambda3*cos(platt1 - 3*platt2))
            H6_tot += vcos6
            I6_tot += sin(platt1+(-1)*platt1shiftX)
            H2_tot += vcos2
            I2_tot += sin(platt2+(-1)*platt2shiftX)
            
            total += exp(1j*(3*rel_angles[i,j]))
            bond_avg += np.cos(rel_angles[i,j] - rel_angles[i-1,j]) + np.cos(rel_angles[i,j] - rel_angles[i,j-1])   
            tot_sector += exp(1j*(rel_angles[i,j]))
            ord6 += exp(1j*platt1)
            ord2 += exp(1j*platt2)
            cm_order += exp(1j*center_of_mass[i,j])
            locking += cos(3*rel_angles[i,j])
            
            ord6_xi_x += cos(platt1)*exp(1j*i*tpi/N)
            ord6_xi_y += sin(platt1)*exp(1j*i*tpi/N)
            #hexatic correlation length
            ord2_xi_x += cos(platt2)*exp(1j*i*tpi/N)
            ord2_xi_y += sin(platt2)*exp(1j*i*tpi/N)
            #potts correlation length
            ordP_xi_x += cos(rel_angles[i,j])*exp(1j*i*tpi/N)
            ordP_xi_y += sin(rel_angles[i,j])*exp(1j*i*tpi/N)
            #U(1) correlation length
            ordU_xi_x += cos(center_of_mass[i,j])*exp(1j*i*tpi/N)
            ordU_xi_y += sin(center_of_mass[i,j])*exp(1j*i*tpi/N)
            
            #vortex calc
            platt1v = mod(platt1, tpi)
            platt1shiftXv = mod(platt1shiftX, tpi)
            platt1shiftXshiftYv = mod(platt1shiftXshiftY, tpi)
            platt1shiftYv = mod(platt1shiftY,tpi)
            diff_list1 = np.array([platt1v - platt1shiftXv, platt1shiftXv - platt1shiftXshiftYv,\
                                   platt1shiftXshiftYv - platt1shiftYv, platt1shiftYv - platt1v])
            platt2v = mod(platt2, tpi)
            platt2shiftXv = mod(platt2shiftX, tpi)
            platt2shiftXshiftYv = mod(platt2shiftXshiftY, tpi)
            platt2shiftYv = mod(platt2shiftY,tpi)
            diff_list2 = np.array([platt2v - platt2shiftXv, platt2shiftXv - platt2shiftXshiftYv,\
                                   platt2shiftXshiftYv - platt2shiftYv, platt2shiftYv - platt2v])
            
            vort2_here = 0.
            vort6_here = 0.
            for ll_1 in diff_list1:
                if ll_1 > np.pi:
                    ll_1 = ll_1 - tpi
                if ll_1 < -np.pi:
                    ll_1 = ll_1 + tpi   
                ll_1 = ll_1/tpi
                vort6_here += ll_1
            
            for ll_2 in diff_list2:     
                if ll_2 > np.pi:
                    ll_2 = ll_2 - tpi
                if ll_2 < -np.pi:
                    ll_2 = ll_2 + tpi   
                ll_2 = ll_2/tpi
                vort2_here += ll_2
                    
            vort6 += absolute(vort6_here)
            vort2 += absolute(vort2_here)

    vort6 = vort6/tpi
    vort2 = vort2/tpi
    
    #stiffness definitions        
    Hx_tot = 9*j6*H6_tot + j2*H2_tot 
    Ix_tot = 3*j6*I6_tot + j2*I2_tot 
    
    #get the norm of these things
    #hexatic correlation length
    ord6_xi_m = (ord6_xi_x.real)**2 + (ord6_xi_x.imag)**2 + (ord6_xi_y.real)**2 + (ord6_xi_y.imag)**2
    #nematic correlation length
    ord2_xi_m = (ord2_xi_x.real)**2 + (ord2_xi_x.imag)**2 + (ord2_xi_y.real)**2 + (ord2_xi_y.imag)**2
    #potts correlation length
    ordP_xi_m = (ordP_xi_x.real)**2 + (ordP_xi_x.imag)**2 + (ordP_xi_y.real)**2 + (ordP_xi_y.imag)**2
    #U(1) correlation length
    ordU_xi_m = (ordU_xi_x.real)**2 + (ordU_xi_x.imag)**2 + (ordU_xi_y.real)**2 + (ordU_xi_y.imag)**2
    

    #now, pack as one list

    #total output
    #to_return = [output_thermo, output_stiff, output_vortex]
    all_dat = np.array([energy, total.real, total.imag, ord6.real, ord6.imag,\
    ord2.real, ord2.imag, tot_sector.real, tot_sector.imag,\
    cm_order.real, cm_order.imag, locking, bond_avg,\
    ord6_xi_m, ord2_xi_m, ordP_xi_m, ordU_xi_m,\
                       H6_tot, I6_tot, H2_tot, I2_tot, Hx_tot, Ix_tot,\
                       vort6, vort2])
    #len of 25

    return all_dat