#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import numpy as np
from numpy.random import rand
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import math
import sys
import os
from random import shuffle
from random import gauss
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fmin
from scipy.optimize import fsolve
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from pylab import polyfit
import matplotlib.ticker as ticker
from matplotlib import gridspec
from scipy.optimize import differential_evolution
import warnings
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from matplotlib import rc
rc('font',**{'family':'sans-serif', 'size' : 10}) #, 'sans-serif':['Arial']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

color_red = (0.73, 0.13869999999999993, 0.)
color_orange = (1., 0.6699999999999999, 0.)
color_green = (0.14959999999999996, 0.43999999999999995, 0.12759999999999994)
color_blue = (0.06673600000000002, 0.164512, 0.776)
color_purple = (0.25091600000000003, 0.137378, 0.29800000000000004)
color_ocker = (0.6631400000000001, 0.71, 0.1491)
color_pink = (0.71, 0.1491, 0.44730000000000003)
color_brown = (0.651, 0.33331200000000005, 0.054683999999999955)

color_all = [color_red, color_orange, color_green, color_blue, color_purple, color_ocker,color_pink, color_brown]

def fit_func_cv(x, b, c, d):
        return  b*np.absolute(x - d)**(-c)


######
#-----------------------------------------------------------------------------------------------------------------------
#######
#parameters of the code
######
#-----------------------------------------------------------------------------------------------------------------------
######

j2 = 1.0
j6 = 2.0 - j2
lambda3 = 2.1
Kc = 0.0
N_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 380]

#data process for N_list

all_data_N = np.load('interpol_data.npy',allow_pickle=True)

########
#find temp of max of spheat
########
q_Q = 1

cv_max = []
cv_max_temp = []

cv_result_temp = []
cv_result_max = []
cv_result_temp_err = []
cv_result_max_err = []
cross_1 = []
cross_1_err = []
for i in range(len(N_list)):

    data_that_N = all_data_N[i]
    #print(data_that_N)
    range_x =  data_that_N[0]


    cv_max_1 = []
    cv_max_temp_1 = []

    orig_temp_r = range_x

    numb_of_try = 10*len(orig_temp_r)

    for u in range(numb_of_try):
        alt_data = np.array([gauss((N_list[i]**2)*data_that_N[1][h], (N_list[i])*data_that_N[2][h]) for h in range(len(orig_temp_r))])
        cv_max_1.append(np.max(alt_data))
        cv_max_temp_1.append(orig_temp_r[np.argmax(alt_data)])

    #T and Cv analysis : get means and std
    #then plot errorbar

    cv_result_temp.append(np.mean(cv_max_temp_1)) #temp at which max
    cv_result_max.append(np.mean(cv_max_1)) #value of max
    cv_result_temp_err.append(np.std(cv_max_temp_1))
    cv_result_max_err.append(np.std(cv_max_1))

    cv_max.append(cv_max_1)
    cv_max_temp.append(cv_max_temp_1)


    stop = 0
    for u in range(len(range_x)):
        if range_x[u] < 1.09:
            stop = u


    #find crossing value
    orig_temp_r = range_x[stop:]
    nt = len(orig_temp_r)
    data_stiff_fit = data_that_N[3][stop:]
    err_stiff_fit = data_that_N[4][stop:]

    #
    #print(orig_temp_r, data_stiff_fit)
    spl = InterpolatedUnivariateSpline(orig_temp_r, data_stiff_fit, k=1)
    func1 = lambda x: spl(x) - (q_Q**2)*2*x/np.pi


    range_temp_try = np.linspace(np.min(orig_temp_r), np.max(orig_temp_r), nt*100)
    idx = np.argwhere(np.diff(np.sign((q_Q**2)*2*range_temp_try/np.pi - spl(range_temp_try)))).flatten()

    #this is a check if any intersection exist
    if idx.size == 0:
        idx = [-1]

    list_of_Tbkt = [range_temp_try[idx][0]]

    numb_of_try = 30*len(orig_temp_r)

    for u in range(numb_of_try):
        alt_data = np.array([gauss(data_stiff_fit[h], math.sqrt(err_stiff_fit[h])) for h in range(len(orig_temp_r))])
        spl_alt = InterpolatedUnivariateSpline(orig_temp_r, alt_data, k=5)
        idx_alt = np.argwhere(np.diff(np.sign((q_Q**2)*2*range_temp_try/np.pi - spl_alt(range_temp_try)))).flatten()
        if idx_alt.size == 0:
            idx_alt = [-1]
        list_of_Tbkt.append(range_temp_try[idx_alt][0])

    #list_of_Tbkt = [range_temp_try[idx][0], range_temp_try[idx_alt_1][0], range_temp_try[idx_alt_2][0], range_temp_try[idx_alt_3][0], range_temp_try[idx_alt_4][0]]
    avg_Tbkt = np.mean(list_of_Tbkt)
    err_Tbkt = np.std(list_of_Tbkt)
    cross_1.append(avg_Tbkt)
    cross_1_err.append(err_Tbkt)

#fit 1 Cv
#need to screen the errors so that there are no zeros
threshold = 1e-7
for u in range(len(cv_result_temp_err)):
    if cv_result_temp_err[u] < threshold:
        cv_result_temp_err[u] = threshold

for u in range(len(cross_1_err)):
    if cross_1_err[u] < threshold:
        cross_1_err[u] = threshold
print('done')


# In[4]:


######
#fit picewise the BKT extrapolation

######
#- initialize plot
######

fig, ax2 = plt.subplots( figsize = (2.23,2.4) )
fin = 4


########
#have an inset
#######
#list of minimum stiffness crossings
#list of T at Cv max

# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
#mark_inset(ax0, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
N_list1 = N_list[fin-1:]
print('L used in first fit')
print(N_list1)
cv_result_temp_a = cv_result_temp[fin-1:]
cv_result_temp_a_err = cv_result_temp_err[fin-1:]
cross_1_a = cross_1[fin-1:]
cross_1_a_err = cross_1_err[fin-1:]

inset_xrange = 1/(np.log((N_list1))**2)
print(f'inset_range = {inset_xrange}.')
#ax2.plot(inset_xrange, tcvmax, Blocks_size[0], color = color_red)
ax2.errorbar(inset_xrange, cv_result_temp_a, yerr = cv_result_temp_a_err, fmt = 'o', linewidth = 0.5, color = color_red, markersize = '2')
#ax2.plot(inset_xrange, tcross, Blocks_size[1], color = 'blue')
ax2.errorbar(inset_xrange, cross_1_a, yerr = cross_1_a_err, fmt = 'o', linewidth = 0.5,  color = 'teal', markersize = '2')


#fits
#func 1
def fit_func1(Nrangex, a, b, c):
    return (a + b*Nrangex**(-(1/c)))

def fit_func1_bis(logrange, a, b,c):
    return a + b*np.exp(-1/(c*np.sqrt(logrange)))

def fit_func1_alt(Nrangex, a, c):
    return a*(1 + Nrangex**(-(1/c)))

def fit_func1_bis_alt(logrange, a,c):
    return a*(1 + np.exp(-1/(c*np.sqrt(logrange))))

#func 2
def fit_func2(logrange, a, b):
    return a*logrange + b

#func 2- divide
def fit_func2_bis(x, a, b, a2, lm):
    return np.piecewise(x, [x < lm], [lambda x:a*x + b-a*lm, lambda x:a2*x + b-a2*lm])

line_plot_range = np.linspace(1e-7, np.max(inset_xrange) + 0.01, 100)

poptC, pcovC = curve_fit(fit_func1, N_list1, cv_result_temp_a, sigma = cv_result_temp_a_err, absolute_sigma = True, p0 = [1.2, 1.2, 0.8], bounds = ([1., 1., 0.6],[1.5, 1.5, 1.0]), maxfev = 9000)
#poptC, pcovC = curve_fit(fit_func1_alt, N_list_extra, cv_result_temp, sigma = cv_result_temp_err,\
# absolute_sigma = True, p0 = [1.2, 0.8], bounds = ([1.,  0.6],[1.5,  1.0]), maxfev = 9000)
valTCV = poptC[0]
errTCV = np.sqrt(np.diag(pcovC))[0]
print('fit of T for CV')
print(valTCV)
print(errTCV)

print('nu')
print(poptC[2])
print(np.sqrt(np.diag(pcovC))[2])
ax2.plot(line_plot_range, fit_func1_bis(line_plot_range, *poptC), '--', linewidth = 0.5, color = color_red, label = r'Potts $T_3$')
#ax2.plot(line_plot_range, fit_func1_bis_alt(line_plot_range, *poptC), '--', color = color_red)

#fit 2 BKT
#poptB, pcovB = curve_fit(fit_func2, inset_xrange, cross_1, sigma = cross_1_err, absolute_sigma = True)
poptB, pcovB = curve_fit(fit_func2, inset_xrange, cross_1_a, sigma = cross_1_a_err, absolute_sigma = True )
valTBKT = poptB[1]
errTBKT = np.sqrt(np.diag(pcovB))[1]
print('fit of T for KT')
print(valTBKT)
print(errTBKT)
#print('length of change')
#print(np.exp(np.sqrt(1/poptB[3])))
#ax2.plot(line_plot_range, fit_func2(line_plot_range, *poptB), '--', color = 'blue')
ax2.plot(line_plot_range, fit_func2(line_plot_range, *poptB), '--', linewidth = 0.5, label = r'KT nematic $T_{2}$', color = 'teal')

####
#compute chances of overlap
mu1=valTBKT
sigma1=errTBKT
mu2=valTCV
sigma2=errTCV
c = (mu2*sigma1**2 - sigma2*(mu1*sigma2 + sigma1*np.sqrt((mu1 - mu2)**2 + 2*(sigma1**2 - sigma2**2)*np.log(sigma1/sigma2))))/(sigma1**2 - sigma2**2)

prob = 1 - 0.5*math.erf((c - mu1)/(np.sqrt(2)*sigma1)) + 0.5*math.erf((c - mu2)/(np.sqrt(2)*sigma2))
print('probability of overlap of Tcs')
print(prob*100)

ax2.set_xlim([0, 0.065])
ax2.set_xlabel(r'$1/(\ln{L})^2$', fontsize = 10)
ax2.set_ylabel(r'Transition temperature $T/J$', fontsize = 10)
#ax2.yaxis.set_label_coords(0.01, 1.08)


# Create patch collection with specified colour/alpha
pcCV = patches.Rectangle((0.0, valTCV - errTCV), 0.003, 2*errTCV, facecolor=color_red, alpha=0.3, edgecolor='none')
# Add collection to axes
ax2.add_patch(pcCV)

# Create patch collection with specified colour/alpha
pcBKT = patches.Rectangle((0.0, valTBKT - errTBKT), 0.003, 2*errTBKT, facecolor='teal', alpha=0.3, edgecolor='none')
# Add collection to axes
ax2.add_patch(pcBKT)

ax2.set_xlim([0, 0.06])
ax2.set_ylim([1.19, 1.218])

major_ticks_x = np.arange(0, 0.07, 0.02)
minor_ticks_x = np.arange(0, 0.07, 0.01)

major_ticks_y = np.arange(1.19, 1.215, 0.01)
minor_ticks_y = np.arange(1.19, 1.215, 0.005)

ax2.set_xticks(major_ticks_x)
ax2.set_yticks(major_ticks_y)
xtick_print = []
ytick_print = []
for elem in major_ticks_x:
    xtick_print.append('${:.2f}$'.format(elem))

for elem in major_ticks_y:
        ytick_print.append('${:.2f}$'.format(elem))

ax2.set_xticklabels(xtick_print, fontsize = 10)
ax2.set_yticklabels(ytick_print, fontsize = 10)

ax2.set_xticks(minor_ticks_x, minor=True)
ax2.set_yticks(minor_ticks_y, minor=True)


# ax2.set_xticks(minor_ticks, minor=True)
# ax2.set_yticks(minor_ticks_y_2, minor = True)


ax2.grid(which='major', axis='both', linestyle='-', alpha = 0.4)
ax2.grid(which='minor', axis='both', linestyle='-', alpha = 0.2)
#ax2.tick_params(axis='both', which='minor', labelsize = 10)



ax2.legend(loc = 'best', fontsize = '9')
#, borderpad = 0.3, labelspacing = 0.4)

plt.tight_layout()

plt.savefig('./fig-interpolate-2.png', format='png', dpi = 600, bbox_inches='tight')
#plt.show()
#plt.close()
