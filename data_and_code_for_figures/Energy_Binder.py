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
from matplotlib.patches import Rectangle, Circle
#from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker

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

def fit_func(xrange_s, a, b):
    return a*xrange_s + b


######
#-----------------------------------------------------------------------------------------------------------------------
#######
#parameters of the code
######
#-----------------------------------------------------------------------------------------------------------------------
######

deltas = [0.5, 1.0, 1.5]
N = 80;

######
#- initialize plot
######
fig = plt.figure(figsize = (3.375,2.09))
ax1 = plt.subplot(1,1,1)

cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='brg')
colors_size = [scalarMap.to_rgba(i/(len(deltas)-1)) for i in range(len(deltas))]


ax1.set_xlabel(r'$T/J$', fontsize = 10)
ax1.set_ylabel(r'$\ln \; B_E$', fontsize = 10)

all_data_d = np.load('data_energy_binder_compare.npy',allow_pickle=True)


for n in range(len(deltas)):
    #ax1.plot(range_x[n], data_thermo[n][:,2], color = colors_size[n], marker = '*', linestyle = '', markersize = 8.0, label = r'$\Delta =$'+str(deltas[n]))
    ax1.errorbar(all_data_d[n][0], all_data_d[n][1], yerr = (N)*all_data_d[n][2],  color = color_all[n+1], marker = 'o', linestyle = '-', linewidth = 0.5, markersize = 1.0, label = r'$\Delta ='+str(deltas[n])+ '$')

# \Delta = 0.5
ax1.plot([0.701784, 0.701784], ax1.get_ylim(), color = color_red, linewidth = 0.75, linestyle = '--')
#ax1.plot([1.34, 1.34], ax1.get_ylim(), color = 'mediumpurple', linewidth = 0.75, linestyle = '--')
# \Delta = 1.0
ax1.plot([1.2038, 1.2038], ax1.get_ylim(), color = color_red, linewidth = 0.75, linestyle = '--')
#ax1.plot([1.194, 1.194], ax1.get_ylim(), color = 'teal', linewidth = 0.75, linestyle = '--')

#\Delta = 1.5
#ax1.plot([1.43, 1.43], ax1.get_ylim(), color = 'teal', linewidth = 0.75, linestyle = '--')

ax1.set_yscale('log')
#ax1.ticklabel_format(axis = 'x', style = 'plain')
#ax1.tick_params(axis='both', which='major', labelsize=16)
#ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
#ax1.ticklabel_format(axis = 'x', style = 'plain')
plt.legend(fontsize = 10)

#major_ticks = np.linspace(Tmin2, Tmax2, 5)
#minor_ticks = np.linspace(Tmin2, Tmax2, 9)

Tmin2 = 0.6
Tmax2 = 2.2
major_ticks = np.arange(Tmin2, Tmax2+0.0001, 0.4)
minor_ticks = np.arange(Tmin2, Tmax2+0.0001, 0.1)
ax1.set_xticks(major_ticks)
tick_print = []
for elem in major_ticks:
    tick_print.append('${:.1f}$'.format(elem))
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_xticklabels(tick_print, fontsize = 10)

#y ticks
yticksvalpre = np.arange(-5, -3-0.01, 0.4)
yticksval = 10**(yticksvalpre)

ytick_print = []
for elem in yticksvalpre:
    ytick_print.append(r'${:.1f}$'.format(elem))

ax1.set_yticks(yticksval)
ax1.set_yticklabels(ytick_print, fontsize = 10)

#ax1.set_yticks(yticksvalmin, minor=True)
#ax1.set_yticklabels(ytick_print, fontsize = 10)

#ax1.set_yticklabels(ax1.get_yticks(), fontsize = 10)
#ax3.xaxis.set_label_coords(1.08, 0.01)
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.4)


plt.savefig('./CompareBinderEnergy.png', format='png', dpi = 600, bbox_inches='tight')
#plt.show()
