#order of data thermo:

#the code starts with  the following routine
"""
#extract the data to plot
data_thermo = []
error_thermo = []
range_x = []
for i in range(len(N_list)):
    name_dir = 'testJ2={:.2f}'.format(j2) + 'J6={:.2f}'.format(j6) +'Lambda={:.2f}'.format(lambda3) + 'L='+str(int(N_list[i]))+ 'Kc={:.2f}'.format(Kc)
    preambule = './'+name_dir+'finalData/'
    data = np.loadtxt(preambule+'thermo_output.data')
    nt = int(len(data[:,0])/2)
    data_thermo.append(data[0:(nt),:])
    error_thermo.append(data[nt:(2*nt),:])
    param = np.loadtxt(preambule + 'variables.data')
    #temperature range
    range_temp = param[7:]
    range_x.append(range_temp)

data_thermo = np.array(data_thermo)
error_thermo = np.array(error_thermo)
range_x = np.array(range_x)

"""
#all the data is in data_thermo. 
#To access observable 'i' for a given N whose index is 'n', call data_thermo[n][:,i]
#similar for its error with error_thermo
#the temperature range is range_x[n]



"""
#note on obs: 
Order = |<m>|
Order _BIS = <|m|>
Susceptibility 1  = (<m^2> - <|m|>^2)/T
Susceptibility 2 = (<m^2>)/T
binder = 1 - (<m^4>/3<m^2>^2)
#important note: one needs to multiply Cv by N (L**2 or N_list[n]) and error on Cv by sqrt(N) = L
#to get Cv/N
#because I have a wrong factor of N in an earlier step (due to confused programmer)

#energy stuff
Energy = 0
SpecificHeat = 1
OrderCumulant = 2

###
e^(i (theta - 3 phi))
###
OrderParam = 3
OrderParam_BIS = 4
Susceptibility1 = 5 
Susceptibility2 = 6 
BinderCumulant = 7

###
e^(i theta)
###
OrderTheta = 8
OrderTheta_BIS = 9
SusceptibilityTheta1 = 10
SusceptibilityTheta2 = 11
BinderTheta = 12

###
e^(i phi)
###
OrderPhi = 13
OrderPhi_BIS = 14
SusceptibilityPhi1 = 15     
SusceptibilityPhi2 = 16    
BinderPhi = 17

###
e^(i (theta - 3 phi)/3)
###
OrderSigma = 18
OrderSigma_BIS = 19
SusceptibilitySigma1 = 20    
SusceptibilitySigma2 = 21    
BinderSigma = 22

###
e^(i(theta + 3 phi))
###
OrderTot = 23
OrderTot_BIS = 24
SusceptibilityTot1 = 25
SusceptibilityTot2 = 26
BinderTot = 27

avg_sigma = 28 (defunct)
std_sigma = 29 (defunct)

####
#cos(theta - 3 phi)
####
OrderLocking = 30
OrderLocking_BIS = 31
SusceptibilityLocking1 = 32
SusceptibilityLocking2 = 33
BinderLocking = 34


###
# cos(sigma_i - sigma_(i + x)) + cos(sigma_i - sigma_(i + y))
####
OrderBondSig = 35
OrderBondSig_BIS = 36
SusceptibilityBondSig1 = 37
SusceptibilityBondSig2 = 38
BinderBondSig = 39

###
#correlation lengths (see Nui (2019))
###
for theta = 40
for phi = 41
for (theta - 3 phi)/3 = 42
for theta + 3 phi = 43


"""

#order of data vorticity
"""

VorticityTheta = 0
VorticityPhi = 1
(rest is defunct)
DeviationTheta = 2 
DeviationPhi = 3 
diffTheta = 4
diffPhi = 5
diff_of_vort_Theta = 6
diff_of_vort_Phi = 7
vort_on_top = 8

"""

#order of data stiffness

"""

RhoTheta = 0
RhoPhi = 1
RhoTot = 2
fourthOrderTheta = 3
fourthOrderPhi = 4
"""