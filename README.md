[![Paper](https://img.shields.io/badge/paper-arXiv%3A2102.11288-B31B1B.svg)](https://arxiv.org/abs/2102.11288)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4553595.svg)](https://doi.org/) -->


# Emergent Potts Order in a Coupled Hexatic-Nematic XY model
[Victor Drouin-Touchette](https://github.com/vdrouint), [Peter P. Orth](https://faculty.sites.iastate.edu/porth/), Piers Coleman, Premala Chandra, Tom Lubensky

### Abstract
Addressing the nature of an unexpected smectic-A’ phase in liquid crystal 54COOBC films, we
perform large scale Monte Carlo simulations of a coupled hexatic-nematic XY model. The resulting
finite-temperature phase diagram reveals a small region with composite Potts Z<sub>3</sub> order above the
vortex binding transition; this phase is characterized by relative hexatic-nematic ordering though
both variables are disordered. The system develops algebraic hexatic and nematic order only at a
lower temperature. This multi-step melting scenario agrees well with the experimental observations
of a sharp specific heat anomaly that emerges above the onset of hexatic positional order. We
therefore propose that the smectic-A’ phase is characterized by composite Potts order and bound-
states of fractional vortices

### Description
This repository includes the Monte-Carlo code that was used to obtain the data, the jackknife error analysis, as well as information, scripts, and data to generate the figures in the paper.


### Monte Carlo routine

Details on the specific Monte Carlo routine are presented in the paper. 

### Figures

#### Zero Coupling Phase Diagram
This is Fig. 3 in the paper. 
<!-- <img src="https://github.com/Orth-Research/Potts-XY/blob/master/plots/blablabla.png" width="400px"> -->


### Support
This work was supported by Grant
No. DE-SC0020353 (P. Chandra) and Grant No. DE-FG02-
99ER45790 (P. Coleman and V.D.T.), all funded by the U.S. Depart-
ment of Energy (DOE), Office of Science, Basic Energy
Sciences, Division of Materials Sciences and Engineer-
ing. V.D.T. is thankful for the support of the Fonds de
Recherche Quebecois en Nature et Technologie. Part of
the research (P.P.O.) was performed at the Ames Labo-
ratory, which is operated for the U.S. DOE by Iowa State
University under Contract DE-AC02-07CH11358. Computational resources were provided by the Rutgers University Beowulf cluster.

<!-- [<img width="100px" src="https://www.nasa.gov/sites/all/themes/custom/nasatwo/images/nasa-logo.svg">](https://www.nasa.gov/offices/education/programs/national/epscor/home/index.html) -->


# Potts-XY

Plotting iPython notebooks for the Potts-XY project.

## Data

The data files are on .npy and .txt format, and have been extracted from the raw data.

## Prerequisites

You need to download the files

```
funcfssa.py
funcfssa2.py
funcfssa3.py
```

Which provide the function that does the finite size scaling fit (the number represents the added number of finite size corrections to scaling).

## Plotting codes

The following notebooks are present:

```
Energy_Binder_different_delta.ipynb
```

This creates a plot comparing the Energy's Binder cumulant for \Delta = 0.5 , 1.0 and 1.5. 

```
Interpolation_plot.ipynb
```

This creates a plot extrapolating the BKT critical temperature and Potts critical temperature to infinite system size, for Delta = 1.0.

```
Plot_low_Delta.ipynb
```

This creates a plot of the specific heat and the spin stiffness for Delta = 0.5, showing the 3 regimes (same result as Jiang et al.)


```
compare_magnetization_obs.ipynb
```

This creates a plot comparing the observables for \Delta = 0.5 , 1.0 and 1.5. Currently set for the Binder of the magnetization of theta, phi and sigma.

```
cv_rho_delta1.ipynb
```

This creates a plot with the specific heat and spin stiffness of L=300 for Delta = 1.0. Also creates what used to be the inset, which is the stiffness close to the transition for different system sizes. 

```
Scaling_Potts.ipynb
```

This does the finite size scaling for the specific heat, the Potts magnetization and the Potts susceptibility for Delta = 1.0.

```
Phase Diagram-v2full.ipynb
Phase Diagram-only inset.ipynb
```

These create the phase diagram plots (including what used to be the inset).

This creates a plot comparing the Energy's Binder cumulant for \Delta = 0.5 , 1.0 and 1.5. 

## Order of data

The data in the ```Delta1_data.npy``` file is the most important, and has data for L=[10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 380]. To extract thermodynamical data, one has the following scheme


```
data = np.load('Delta1_data.npy',allow_pickle = True)
x_data = data[n][0]
y_data = data[n][2*ind + 1]
y_err = data[n][2*ind + 2]
```
where n is the index for the size studied, and ind is an index corresponding to which observable to look at. We have ind = 0 ) energy 1) Specific Heat 2) Binder of the energy 3) m_theta 4) chi_theta 5) Binder theta 6) m_phi 7) chi_phi 8) Binder phi 9) m_sigma 10) chi_sigma 11) Binder sigma 12) total spin stiffness.



