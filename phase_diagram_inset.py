#!/usr/bin/env python
# coding: utf-8

# In[24]:


#matplotlib inline
from __future__ import division
import numpy as np
from numpy.random import rand
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.patches import Arrow, Circle, Rectangle
from matplotlib.patches import ConnectionPatch, Polygon

from matplotlib import rc
rc('font',**{'family':'sans-serif', 'size' : 19}) #, 'sans-serif':['Arial']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



#info on phase diagram
#black dot -> Q=1/3 vortices unbind
#red dot -> Q=1 vortices unbind
#green triangles -> cv max

#list of tcs at L=40
list_of_everything = np.loadtxt('tcs.data')


lambda3=2.1
#fraction=j2/j6

#temperature range
Tmax = 1.6
Tmax_plot = 1.6
Tmin = 0.6


fig, ax = plt.subplots(figsize = (10, 5))
#lambda = 0 KT points
tkt = 0.89
#plotting the two bare KT transitions
# """
# plt.plot([0,2],[2*tkt,0], '--', color="Blue");
# plt.plot([0,2],[0,2*tkt], '--', color="Blue");
# """

#all_cross = [[stiff_cross_j2, '*', 'black'], [sp_heat_cross_j2, '*', 'blue'], [binder_potts_j2, 'o', 'blue']]

#plot the black dotted box of the inside part
#plt.plot([0.5, 1.5], [Tmin, Tmin], color = 'black', linestyle = '--')
#plt.plot([0.5, 1.5], [Tmax, Tmax], color = 'black', linestyle = '--')

patches_stiff = []
patches_cv = []
patches_stiff2 = []
patches_cv2 = []
range_J2 = []

ixB = []
iyB = []
ixC = []
iyC = []

fP = []
fP_x = []

fKT1 = []
fKT1_x = []

fKT2 = []
fKT2_x = []


for i in range(len(list_of_everything)):
    vals = list_of_everything[i]
    if vals[3] == 0:
        col = 'mediumpurple'
    else:
        col = 'teal'

    patches_stiff.append(Circle((vals[0], vals[2]), radius=0.01, facecolor=col, edgecolor = 'black'))
    #patches_cv.append(Circle((vals[0], vals[1]), radius=0.01, facecolor='red', edgecolor = 'black'))
    patches_stiff2.append(Circle((vals[0], vals[2]), radius=0.01, facecolor=col, edgecolor = 'black'))
    #patches_cv2.append(Circle((vals[0], vals[1]), radius=0.01, facecolor='red', edgecolor = 'black'))
    range_J2.append(vals[0])

    if 0.85 <= vals[0] <= 1.1:
        ixB.append(vals[0])
        ixC.append(vals[0])
        iyB.append(vals[2])

    if vals[0] <= 1.1:
        fP_x.append(vals[0])

    if vals[0] <= 0.85:
        fKT1.append(vals[2])
        fKT1_x.append(vals[0])

    if 0.85 <= vals[0]:
        fKT2.append(vals[2])
        fKT2_x.append(vals[0])

range_J2 = np.array(range_J2)

N_cp = 40
Kc = 0.0


range_T = np.linspace(Tmin + 0.0001, Tmax, 60)
#print(range_T)


initial_cv_val = np.loadtxt('CV_data_pd.txt')
gridplot_cv = np.zeros((len(range_T), len(range_J2)))
for j in range(len(range_J2)):

    #cv
    #gridplot_cv[:,j] = (final_cv_val)
    #log of cv
    gridplot_cv[:,j] = np.log(initial_cv_val[:,j])

    #get cv_max for that size
    initial_cv_val_here = initial_cv_val[:,j]
    maxcv = range_T[np.where(initial_cv_val_here == np.max(initial_cv_val_here))[0][0]]
    print(maxcv)
    if range_J2[j] > 1.2:
        maxcv = list_of_everything[j][1]

    if range_J2[j] <= 1.1:
        patches_cv.append(Circle((range_J2[j], maxcv), radius=0.01, facecolor='red', edgecolor = 'black'))
        patches_cv2.append(Circle((range_J2[j], maxcv), radius=0.01, facecolor='red', edgecolor = 'black'))
    else:
        patches_cv.append(Rectangle((range_J2[j]- 0.01, maxcv - 0.01), 0.01, 0.01, facecolor='red', edgecolor = 'black'))
        patches_cv2.append(Rectangle((range_J2[j] - 0.01, maxcv - 0.01), 0.01, 0.01, facecolor='red', edgecolor = 'black'))

    if 0.85 <= range_J2[j] <= 1.1:
        iyC.append(maxcv)

    if range_J2[j] <= 1.1:
        fP.append(maxcv)


ixB = np.array(ixB)[::-1]
ixC = np.array(ixC)
iyB = np.array(iyB)[::-1]
iyC = np.array(iyC)

im = ax.imshow(gridplot_cv, interpolation='spline16', cmap='YlGn',origin='lower',  aspect='auto',  extent = [0.5 - 0.025, 1.5 + 0.025, 0.6 - 1/(2*59), 1.6 + 1/(2*59)])

#clb = plt.colorbar(im, shrink=0.5)
#clb.ax.tick_params(labelsize=12)
#clb.ax.set_title(r'$C_v/N$', fontsize = 12)
#clb.ax.set_title(r'$\log \; C_v$', fontsize = 12)


x1, x2, y1, y2 = 0.8, 1.15, 1.05, 1.3
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)

for p in patches_stiff2:
    ax.add_patch(p)

for ps in patches_cv2:
    ax.add_patch(ps)


plt.xlabel(r"$\Delta$", fontsize=26);
plt.ylabel(r"Temperature $T$", fontsize=26)

#ticks
major_ticks_x = np.arange(0.8, 1.15 + 0.01, 0.1)
minor_ticks_x = np.arange(0.8, 1.15 + 0.01, 0.025)
major_ticks_y = np.arange(1.05, 1.3 + 0.01, 0.1)
minor_ticks_y = np.arange(1.05, 1.3 + 0.01, 0.025)

tick_print_x = []
for elem in major_ticks_x:
    tick_print_x.append('{:.1f}'.format(elem))

tick_print_y = []
for elem in major_ticks_y:
    tick_print_y.append('{:.2f}'.format(elem))

ax.set_xticks(major_ticks_x)
ax.set_yticks(major_ticks_y)
ax.set_xticklabels(tick_print_x, fontsize = 20)
ax.set_yticklabels(tick_print_y, fontsize = 20)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(minor_ticks_y, minor=True)
#ax.set_xticklabels(tick_print, rotation=315)
ax.grid(which='minor', alpha=0.3)
ax.grid(which='major', alpha=0.6)

#ax.set_xlim([0,2])
#ax.set_ylim([0,Tmax_plot])
#ax.xaxis.set_label_coords(1.08, -0.03)

textstr = r'nematic'
ax.text(0.6, 0.2, textstr, transform=ax.transAxes, fontsize=26,
    verticalalignment='top')
textstr = r'disordered'
ax.text(0.1, 0.8, textstr, transform=ax.transAxes, fontsize=26,
    verticalalignment='top')
# textstr = '$Z_3$'
# ax.text(0.35, 0.52, textstr, transform=ax.transAxes, fontsize=35,
#     verticalalignment='top', color = 'white')

#insert a shaded region
verts = [*zip(ixC, iyC), *zip(ixB, iyB)]
poly = Polygon(verts, facecolor='crimson', edgecolor='none', alpha = 0.6)
ax.add_patch(poly)

ax.plot(fP_x, fP, color = 'red')
ax.plot(fKT1_x, fKT1, color = 'mediumpurple')
ax.plot(fKT2_x, fKT2, color = 'teal')



###########################
#####inset
###########################


#ax.set_ylim([0.6, 1.6])
#ax.set_ylim([0,Tmax_plot])

#ax.indicate_inset_zoom(axins)

#plt.tight_layout()
plt.savefig('./fig-phasediagram-inset-cv.png', format='png',dpi = 100, bbox_inches='tight')

plt.show()


# In[ ]:
