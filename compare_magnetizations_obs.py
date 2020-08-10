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
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fmin
from scipy.optimize import fsolve
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.signal import savgol_filter
from random import gauss
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

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


#simple plots
######
#- plot
######
#fig = plt.figure(figsize = (3.375, 4.17))

fig, axs = plt.subplots(nrows=3, ncols = 3, sharex=True, sharey = True, figsize = (2*3.375 - 0.83, 4.17) ) # frameon=False removes frames

#plt.subplots_adjust(hspace=.02, wspace=.02)

for i in range(3):
    for j in range(3):
        #axs[i,j].grid()

        major_ticks_x = np.arange(0, 2.5 + 0.01, 1.)
        minor_ticks_x = np.arange(0., 2.5 + 0.01, 0.25)

        major_ticks_y = np.arange(0.4, 0.61, 0.2)
        minor_ticks_y = np.arange(0.3, 0.7, 0.1)

        axs[i,j].set_xticks(major_ticks_x)
        axs[i,j].set_xticks(minor_ticks_x, minor = True)
        axs[i,j].set_yticks(major_ticks_y)
        axs[i,j].set_yticks(minor_ticks_y, minor = True)

        axs[i,j].plot([0.0, 2.5], [0.33, 0.33], color = 'black', linewidth = 0.75, linestyle = '--')
        axs[i,j].plot([0.0, 2.5], [0.66, 0.66], color = 'black', linewidth = 0.75, linestyle = '--')
        axs[i,j].set_ylim([0.28, 0.7])

        axs[i,j].grid(which='major', axis='both', linestyle='-', alpha = 0.4)
        axs[i,j].grid(which='minor', axis='both', linestyle='-', alpha = 0.2)
        axs[i,j].tick_params(axis='both', which='major', labelsize=10)
        axs[i,j].tick_params(axis='both', which='minor', labelsize=10)

        if (j == 0):
            axs[i,j].plot([0.7017847457627119, 0.7017847457627119], axs[i,j].get_ylim(), color = color_red, linewidth = 0.75, linestyle = '--')
            axs[i,j].plot([1.34, 1.34], axs[i,j].get_ylim(), color = 'mediumpurple', linewidth = 0.75, linestyle = '--')
        elif (j == 1):
            axs[i,j].plot([1.2038, 1.2038], axs[i,j].get_ylim(), color = color_red, linewidth = 0.75, linestyle = '--')
            axs[i,j].plot([1.194, 1.194], axs[i,j].get_ylim(), color = 'teal', linewidth = 0.75, linestyle = '--')
        elif(j == 2):
            axs[i,j].plot([1.43, 1.43], axs[i,j].get_ylim(), color = 'teal', linewidth = 0.75, linestyle = '--')
        else:
            pass


deltas = [0.5, 1.0, 1.5]
lambda3 = 2.1
Kc = 0.0
# \Delta = [0.5, 1.0, 1.5]
N_list = [[20, 40, 60, 80], [20, 40, 60, 80, 100, 140, 180], [20, 40, 60, 80, 100, 120]]



#inds = [12, 17, 22] becomes
#ind = 22 is sigma
#ind = 12 is theta
#ind = 17 is phi
inds = [2,5,8] #of obs

#if you want mag
#inds = [0,3,6] #of obs
#if you want susc
#inds = [1,4,7] #of obs


cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='viridis_r')

all_data = np.load('data_mag_compare.npy',allow_pickle=True)


for i in range(len(deltas)):
    #get data
    j2 = deltas[i]
    j6 = 2.0-j2

    #data_thermo = []
    #error_thermo = []
    #range_x = []
    N_d = N_list[1]

    colors_size = [scalarMap.to_rgba(j/(len(N_d)-1)) for j in range(len(N_d))]

    for n in range(len(N_list[i])):
        for l in range(3):

            ind = inds[l]
            temp = all_data[i][n][0]
            data = all_data[i][n][2*ind + 1]
            error = all_data[i][n][2*ind + 2]
            if (l == 1 and i == 1):
                pass
            else:
                axs[l,i].errorbar(temp, data, yerr = error, fmt ='-o', markersize = 2, linestyle = '-', linewidth = 0.5, color=colors_size[n])

# do the i=l=1 case separately to ensure the legend can be printed.
#get data
j2 = deltas[1]
j6 = 2.0-j2

#data_thermo = []
#error_thermo = []
#range_x = []
N_d = N_list[1]

colors_size = [scalarMap.to_rgba(j/(len(N_d)-1)) for j in range(len(N_d))]

for n in range(len(N_list[1])):
    ind = inds[1]
    temp = all_data[1][n][0]
    data = all_data[1][n][2*ind + 1]
    error = all_data[1][n][2*ind + 2]
    print(f'N_list = {N_list[1][n]}.')
    axs[1,1].errorbar(temp, data, yerr = error, fmt ='-o', markersize = 2, linestyle = '-', linewidth = 0.5, color=colors_size[n], label = r'$' + str(int(N_list[1][n])) + '$' )


fig.legend(title = '$L = $', loc = 'upper left', bbox_to_anchor=(1.0, 0.94), title_fontsize = '10', fontsize = '10')

#do the legend and append at the end

axs[0,0].set_xlabel(r'$\Delta = 0.5$ ', fontsize=10);
axs[0,0].xaxis.set_label_position('top')
axs[0,1].set_xlabel(r'$\Delta = 1.0$ ', fontsize=10);
axs[0,1].xaxis.set_label_position('top')
axs[0,2].set_xlabel(r'$\Delta = 1.5$ ', fontsize=10);
axs[0,2].xaxis.set_label_position('top')

axs[0,0].set_ylabel(r'$B_{\theta}$ ', fontsize=10);
axs[1,0].set_ylabel(r'$B_{\phi}$ ', fontsize=10);
axs[2,0].set_ylabel(r'$B_{\sigma}$ ', fontsize=10);


axs[2,0].set_xlabel('$T/J$', fontsize = 10)
axs[2,1].set_xlabel('$T/J$', fontsize = 10)
axs[2,2].set_xlabel('$T/J$', fontsize = 10)

plt.tight_layout()

plt.savefig('./Compare_binders.png', format='png', dpi = 600, bbox_inches='tight')
#plt.show()


# In[ ]:
