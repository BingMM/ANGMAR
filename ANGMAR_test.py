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
import polplot

#%% Paths

cwd = os.getcwd()
base = cwd
path_in = os.path.join(base, 'data')
path_out = os.path.join(base, 'figures', 'ANGMAR_test')

#%% Import data

fn = os.path.join(path_in, 'ionospheric_data_from_Lompe.h5')

keys = ['Be', 'Bn', 'Bu', 'S']
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

#%% Define Lompe grids

print('Defining Lompe CS grid')

position = (270, 79) # lon, lat for center of the grid
orientation = 0.
L = 45e6
Lres = 60e3#30e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)
s_limit = np.min([grid.Wres, grid.Lres])/2

#%% Define SECS grids

print('Defining CS grid')

position = (25, 68) # lon, lat for center of the grid
orientation = 8
L = 1.1e6
Lres = 40e3#60e3
grid_s = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)

position = (210, 66) # lon, lat for center of the grid
orientation = -12
L = 1.4e6
Lres = 40e3#60e3
grid_a = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)

position = (255, 62) # lon, lat for center of the grid
orientation = -12
L = 3e6
W = 2.4e6
Lres = 40e3#60e3
grid_c = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Lres, R = 6371.2e3 + 110e3)

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

def add_noise(mu, std):    
    return np.random.normal(mu, std, mu.size)

#%% Solve inverse problems

Be_pred, Bn_pred, Bu_pred = [], [], []

for grid_ in [grid_a, grid_c, grid_s]:
    s_limit_ = np.min([grid_.Wres, grid_.Lres])/2
    
    f = grid_.ingrid(lon, lat)
    lon_, lat_, r_ = lon[f], lat[f], r[f]
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat_, lon_, r_, grid_.lat.flatten(), grid_.lon.flatten(), RI = grid_.R, singularity_limit=s_limit_)
    G = np.vstack((Ge, Gn, Gu))
    
    Qinv = np.diag(np.ones(G.shape[0]) / 5e-9**2)
    
    GTQG = G.T.dot(Qinv).dot(G)
    
    gmag = np.median(np.diag(GTQG))
    l1 = 1e0
    GTQGR = GTQG + l1*gmag*np.eye(GTQG.shape[0])    
    GTQGRinv = np.linalg.lstsq(GTQGR, np.eye(GTQGR.shape[0]))[0]
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(grid_.lat_mesh.flatten(), grid_.lon_mesh.flatten(), np.ones(grid_.xi_mesh.size)*6371.2e3, grid_.lat.flatten(), grid_.lon.flatten(), RI = grid_.R, singularity_limit=s_limit_)
    
    Be_pred_, Bn_pred_, Bu_pred_ = [], [], []
    for dat_ in tqdm(dat, total=len(dat)):
        time = dat_['time']
        S = dat_['S']
        
        Be_, Bn_, Bu_ = generate_data_points(lat_, lon_, r_, grid.lat.flatten(), grid.lon.flatten(), grid.R, S.flatten())
        Be_ = add_noise(Be_, 5e9)
        Bn_ = add_noise(Bn_, 5e9)
        Bu_ = add_noise(Bu_, 5e9)
        d = np.hstack((Be_, Bn_, Bu_))
            
        GTQd = G.T.dot(Qinv).dot(d)
        
        #m = np.linalg.lstsq(GTQGR, GTQd)[0]
        m = GTQGRinv.dot(GTQd)

        Be_pred_.append(Ge.dot(m))
        Bn_pred_.append(Gn.dot(m))
        Bu_pred_.append(Gu.dot(m))
        
    Be_pred.append(Be_pred_)
    Bn_pred.append(Bn_pred_)
    Bu_pred.append(Bu_pred_)

#%%

for i, grid_ in enumerate([grid_a, grid_c, grid_s]):
    folder = os.path.join(path_out, f'{i}')
    os.makedirs(folder, exist_ok=True)
    
    f = grid_.ingrid(lon, lat)
    lon_, lat_, r_ = lon[f], lat[f], r[f]
    xi_, eta_ = grid_.projection.geo2cube(lon_, lat_)
    
    for j, dat_ in enumerate(dat):
        
        fn = os.path.join(folder, f'{j}.png')
        
        Be, Bn, Bu = Be_pred[i][j], Bn_pred[i][j], Bu_pred[i][j]
        vmax = np.max(abs(np.sqrt(Be**2 + Bn**2 + Bu**2)))
        clvls = np.linspace(-vmax, vmax, 40)
        cmap = 'bwr'
        
        plt.ioff()
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        axs[0, 0].tricontourf(grid_.eta_mesh.flatten(), grid_.xi_mesh.flatten(), Be, cmap=cmap, levels=clvls)
        axs[0, 1].tricontourf(grid_.eta_mesh.flatten(), grid_.xi_mesh.flatten(), Bn, cmap=cmap, levels=clvls)
        axs[0, 2].tricontourf(grid_.eta_mesh.flatten(), grid_.xi_mesh.flatten(), Bu, cmap=cmap, levels=clvls)
        
        for ax in axs[0, :]:
            ax.plot(xi_, eta_, '*', markersize=10, color='tab:green')
        
        coastlines = np.load(os.path.join(os.path.dirname(polplot.__file__), 'data', 'coastlines_50m.npz'))
        for cl in coastlines:
            latc, lonc = coastlines[cl]
            xic, etac = grid_.projection.geo2cube(lonc, latc)
            for ax in axs.flatten():
                ax.plot(xic, etac, linewidth=.5, color='k')
                        
        f = grid_.ingrid(grid.lon_mesh.flatten(), grid.lat_mesh.flatten())        
        xi, eta = grid_.projection.geo2cube(grid.lon_mesh.flatten(), grid.lat_mesh.flatten())        
        axs[1, 0].tricontourf(eta[f], xi[f], dat_['Be'].flatten()[f], cmap=cmap, levels=clvls)
        axs[1, 1].tricontourf(eta[f], xi[f], dat_['Bn'].flatten()[f], cmap=cmap, levels=clvls)
        axs[1, 2].tricontourf(eta[f], xi[f], dat_['Bu'].flatten()[f], cmap=cmap, levels=clvls)        
        
        ax.set_xlim(grid_.eta_mesh.min(), grid_.eta_mesh.max())
        ax.set_ylim(grid_.xi_mesh.min(), grid_.xi_mesh.max())
        
        plt.savefig(fn, bbox_inches='tight')
        plt.close('all')
        plt.ion()












