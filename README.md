[![Paper](https://img.shields.io/badge/paper-arXiv%3A2102.11288-B31B1B.svg)](https://arxiv.org/abs/2102.11288)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4553595.svg)](https://doi.org/) -->


# Emergent Potts Order in a Coupled Hexatic-Nematic XY model
[Victor Drouin-Touchette](https://vdrouint.github.io/), [Peter P. Orth](https://faculty.sites.iastate.edu/porth/), Piers Coleman, Premala Chandra, Tom Lubensky

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

Details on the specific Monte Carlo routine are presented in the paper. The code can be found in the **MC_routine** folder. It was written on Python. We use a specifically adapted Wolff algorithm, tailored for coupled XY models. We also use a parallel tempering routine where <img src="https://render.githubusercontent.com/render/math?math= N_t"> temperatures between <img src="https://render.githubusercontent.com/render/math?math= T_{max}"> and <img src="https://render.githubusercontent.com/render/math?math= T_{min}">, on <img src="https://render.githubusercontent.com/render/math?math= N_c"> cores. This code was made to run on parallelize cluster environments.

```
python mcptdoublel.py 10 1.0 2.1 Nt Nc Tmax Tmin

```

where <img src="https://render.githubusercontent.com/render/math?math= L=10">, <img src="https://render.githubusercontent.com/render/math?math= \Delta = 1.0"> and <img src="https://render.githubusercontent.com/render/math?math= \lambda = 2.1"> are the parameters of the simulation one wants to do. This uses the file *functions_mcstep3.py* where the functions used for the Monte-Carlo sampling are found.

Then, using

```
python all_data_process.py 10 1.0 2.1

```

This processes the obtained raw data, using the Jackknife method to extract errobars, analyze the correlation between different configurations, and overall get the final usable data tables. 

### Data

Data used for the figures was re-packaged from the full obtained data in the sake of conserving memory space. These new compact files (of .npy format) are found in the **data_and_code_for_figures** folder. There is also a subfolder with an example of how one of these files was made, including the original data. For more information, contact Victor Drouin-Touchette.


### Figures

All the codes used to create the figures in the paper are found in the **data_and_code_for_figures** folder. They are all written in Python (version 3 compatible only, because of the data pickling), and extensively use the matplotlib library.

#### Zero Coupling Phase Diagram

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_03.png" width="400px"> 

This is Fig. 3 in the paper. This is obtained using

```
Phase-Diagram-lambda0.py
```

#### Finite Coupling Phase Diagram

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_04.png" width="400px"> 

This is Fig. 4 in the paper. This is obtained using

```
Phase-Diagram-full.py
Phase-Diagram-only-inset.py
```

#### Low-Delta Thermodynamics

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_05.png" width="400px"> 

This is Fig. 5 in the paper. This is obtained using

```
Plot_low_Delta.py
```

#### Delta = 1 Thermodynamics

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_06.png" width="400px"> 

This is Fig. 6 in the paper. Subfigures (a) and (b) are obtained with 

```
cv_rho_delta1.py
```

while (c) is obtained with

```
extrapolation_plot.py
```

#### Finite-Size Scaling

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_07.png" width="400px"> 

This is Fig. 7 in the paper. This is obtained using

```
Scaling_Potts.py
```

This uses previously obtained values for the unbiased scaling fits using the corrections to scaling. Those are done using 

```
pre_Scaling_Potts.ipynb
```

#### Energy Binder Cumulant Analysis

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_08.png" width="400px"> 

This is Fig. 8 in the paper. This is obtained using

```
Energy_Binder.py
```

#### Finite Coupling Phase Diagram

<img src="https://github.com/Orth-Research/Potts-XY/blob/master/Figures/Figure_App_2.png" width="400px"> 

This is Fig. 2 of the appendix in the paper. This is obtained using

```
compare_magnetizations_obs.py
```

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

[<img width="100px" src="logos/rutgers.png">]
[<img width="100px" src="logos/doe.jpg">]
[<img width="100px" src="logos/FRQNT_RGB.png">]




### Note: Order of data

The data in the ```Delta1_data.npy``` file is the most important, and has data for L=[10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 380]. To extract thermodynamical data, one has the following scheme


```
data = np.load('Delta1_data.npy',allow_pickle = True)
x_data = data[n][0]
y_data = data[n][2*ind + 1]
y_err = data[n][2*ind + 2]
```
where n is the index for the size studied, and ind is an index corresponding to which observable to look at. These are the following 

0. Energy 
1. Specific Heat
2. Binder of the energy 
3. m_theta 
4. chi_theta 
5. Binder theta 
6. m_phi 
7. chi_phi 
8. Binder phi 
9. m_sigma 
10. chi_sigma 
11. Binder sigma 
12. total spin stiffness



