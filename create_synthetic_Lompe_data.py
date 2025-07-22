#%% Import

import os
import numpy as np
import pandas as pd
import lompe
from secsy import get_SECS_B_G_matrices
from tqdm import tqdm
import h5py
from datetime import datetime
import matplotlib.pyplot as plt

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data')
path_out = os.path.join(base, 'figures')

#%% Import data

fn = os.path.join(path_in, 'ionospheric_data_from_Lompe.h5')

keys = ['S', 'Be', 'Bn', 'Bu']
dat = []
with h5py.File(fn, 'r') as hf:
    for step in hf.values():
        record = {key: step[key][()] for key in keys}
        record['time'] = datetime.fromisoformat(step.attrs['time'])
        dat.append(record)

#%% Import gmag location

fn = os.path.join(path_in, '20250717-14-57-supermag-stations.csv')

st_data = pd.read_csv(fn, usecols=range(6))

lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()[lat>0]
lat = lat[lat>0]
r = np.ones(len(lat)) * 6371.2e3

#%% Define grids

print('Defining Lompe CS grid')

position = (270, 79) # lon, lat for center of the grid
orientation = 0.
L = 45e6
Lres = 60e3#30e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)
s_limit = np.min([grid.Wres, grid.Lres])/2

#%% Func

def generate_data_point(lat, lon, r, latS, lonS, rS, S):
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, 
                                       latS, lonS, RI=rS, 
                                       singularity_limit=s_limit)
    
    return Ge.dot(S), Gn.dot(S), Gu.dot(S)

def generate_data_points(lats, lons, rs, latS, lonS, rS, S, pbar=False):
    
    Be = np.zeros(len(lats))
    Bn = np.zeros(len(lats))
    Bu = np.zeros(len(lats))
    
    if pbar:
        loop = tqdm(enumerate(zip(lats, lons, rs)), total=lats.size)
    else:
        loop = enumerate(zip(lats, lons, rs))
    for i, (lat, lon, r) in loop:
        Be_, Bn_, Bu_ = generate_data_point(lat, lon, r, latS, lonS, rS, S)
        Be[i], Bn[i], Bu[i] = Be_, Bn_, Bu_
    
    return Be, Bn, Bu

#%% Gamera comparison

Be, Bn, Bu = generate_data_points(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.xi_mesh.size)*6371.2e3, 
                                  grid.lat.flatten(), grid.lon.flatten(), grid.R, 
                                  dat[0]['S'].flatten(), pbar=True)

#%% Comparison plot

vmax1 = np.max(np.sqrt(Be**2 + Bn**2 + Bu**2))
vmax2 = np.max(np.sqrt(dat[0]['Be']**2 + dat[0]['Bn']**2 + dat[0]['Bu']**2))
print(vmax1, vmax2)
vmax = np.max([vmax1, vmax2])
clvls = np.linspace(-vmax, vmax, 40)
cmap = 'bwr'

plt.ioff()
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), Be, cmap=cmap, levels=clvls)
axs[0, 1].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), Bn, cmap=cmap, levels=clvls)
axs[0, 2].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), Bu, cmap=cmap, levels=clvls)

axs[1, 0].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), dat[0]['Be'].flatten(), cmap=cmap, levels=clvls)
axs[1, 1].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), dat[0]['Bn'].flatten(), cmap=cmap, levels=clvls)
axs[1, 2].tricontourf(grid.eta_mesh.flatten(), grid.xi_mesh.flatten(), dat[0]['Bu'].flatten(), cmap=cmap, levels=clvls)        

plt.savefig(os.path.join(path_out, 'synth_data_creation_with_SECS.png'), bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Generate data example

Be, Bn, Bu = [], [], []
for dat_ in tqdm(dat, total=len(dat), desc='Loop over time'):
    Be_, Bn_, Bu_ = generate_data_points(lat, lon, r, 
                                         grid.lat.flatten(), grid.lon.flatten(), grid.R, 
                                         dat_['S'].flatten())
    Be.append(Be_)
    Bn.append(Bn_)
    Bu.append(Bu_)
