#%% Import

import os
import numpy as np
import pandas as pd
import lompe
from secsy import get_SECS_B_G_matrices
from tqdm import tqdm
import h5py
from datetime import datetime

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data')
path_out = os.path.join(base, 'figures')

#%% Import data

fn = os.path.join(path_in, 'ionospheric_data_from_Lompe.h5')

keys = ['S']
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

def generate_data_points(lats, lons, rs, latS, lonS, rS, S):
    
    Be = np.zeros(len(lats))
    Bn = np.zeros(len(lats))
    Bu = np.zeros(len(lats))
    
    for i, (lat, lon, r) in enumerate(zip(lats, lons, rs)):
        Be_, Bn_, Bu_ = generate_data_point(lat, lon, r, latS, lonS, rS, S)
        Be[i], Bn[i], Bu[i] = Be_, Bn_, Bu_
    
    return Be, Bn, Bu

#%% Generate data

Be, Bn, Bu = [], [], []
for dat_ in tqdm(dat, total=len(dat), desc='Loop over time'):
    Be_, Bn_, Bu_ = generate_data_points(lat, lon, r, 
                                         grid.lat.flatten(), grid.lon.flatten(), grid.R, 
                                         dat_['S'].flatten())
    Be.append(Be_)
    Bn.append(Bn_)
    Bu.append(Bu_)
