#%% Import

import os
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot
import h5py
from datetime import datetime, timedelta
from dipole import Dipole
from tqdm import tqdm
from apexpy import Apex
import lompe

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', 'ionospheric_data_from_Lompe.h5')
path_out = os.path.join(base, 'figures')

#%% Import data

keys = ['SH', 'SP', 'FAC', 'Je', 'Jn', 'Be', 'Bn', 'Bu', 'S', 'FAC_int']
dat = []
with h5py.File(path_in, 'r') as hf:
    for step in hf.values():
        record = {key: step[key][()] for key in keys}
        record['time'] = datetime.fromisoformat(step.attrs['time'])
        dat.append(record)

#%% Define grid

print('Defining CS grid')

position = (270, 79) # lon, lat for center of the grid
orientation = 0.
L = 45e6
Lres = 200e3#60e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)

#%% Convert to magnetic

for i in tqdm(range(len(dat)), total=len(dat), desc='Converting to magnetic'):
    time = dat[i]['time']    
    dpl = Dipole(epoch=time.year)
    
    _, _, Je, Jn = dpl.geo2mag(grid.lat, grid.lon, dat[i]['Je'], dat[i]['Jn'])
    dat[i]['Je'], dat[i]['Jn'] = Je, Jn
    
    _, _, Be, Bn = dpl.geo2mag(grid.lat_mesh, grid.lon_mesh, dat[i]['Be'], dat[i]['Bn'])
    dat[i]['Be'], dat[i]['Bn'] = Be, Bn

#%% Plot data

for key in tqdm(['SH', 'SP', 'FAC', 'FAC_int', 'Bu'], total=10):
    folder = os.path.join(path_out, 'plot_Lompe', key)
    os.makedirs(folder, exist_ok=True)
    for i in range(len(dat)):
        fn = os.path.join(folder, f'{i}.png')
        
        time = dat[i]['time']
        
        dpl = Dipole(epoch=time.year)
        apx = Apex(time)
        
        var = dat[i][key]
        if var.shape[0] == grid.shape[0]:
            lat, lon = grid.lat, grid.lon
        else:
            lat, lon = grid.lat_mesh, grid.lon_mesh
        
        mlat, mlon = dpl.geo2mag(lat, lon)
        mlt = dpl.mlon2mlt(mlon, time)
        
        if key == 'SH' or key == 'SP':
            vmax = np.max(var)
            clvls = np.linspace(0, vmax, 40)
            cmap = 'magma'
        else:
            vmax = np.max(abs(var))
            clvls = np.linspace(-vmax, vmax, 40)
            cmap = 'bwr'
            
        plt.ioff()
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
        pax = Polarplot(ax)
        pax.coastlines(time, mag=apx)
        pax.tricontourf(mlat.flatten(), mlt.flatten(), var.flatten(), cmap=cmap, levels=clvls)
        
        if key == 'FAC' or key == 'Bu':
            if key == 'FAC':
                ve, vn = dat[i]['Je'], dat[i]['Jn']
            else:
                ve, vn = dat[i]['Be'], dat[i]['Bn']
            qmax = np.max(np.sqrt(ve**2 + vn**2))*10
            s=1
            pax.quiver(mlat[::s, ::s].flatten(), mlt[::s, ::s], vn[::s, ::s].flatten(), ve[::s, ::s].flatten(), scale=qmax, width=.001)
                
        plt.title(f"{dat[i]['time']} : {key}")
        plt.savefig(fn, bbox_inches='tight')
        plt.close('all')
        plt.ion()

#%% Rotate and test positions

folder = os.path.join(path_out, 'off_set_tests')
os.makedirs(folder, exist_ok=True)


for i in tqdm(range(len(dat)), total=len(dat), desc='Rotating to test off-set'):
    fn = os.path.join(folder, f'{i}.png')
    
    time = dat[i]['time']
        
    dpl = Dipole(epoch=time.year)    
        
    var = dat[i]['FAC']
    lat, lon = grid.lat, grid.lon
    
    mlat, mlon = dpl.geo2mag(lat, lon)
    mlt = dpl.mlon2mlt(mlon, time)
    
    vmax = np.max(abs(var))
    clvls = np.linspace(-vmax, vmax, 40)
    cmap = 'bwr'
    
    plt.ioff()
    fig, axs = plt.subplots(4, 6, figsize=(30, 20))
    paxs = [Polarplot(ax) for ax in axs.flatten()]
    for offset, (pax, ax) in enumerate(zip(paxs, axs.flatten())):
        pax.tricontourf(mlat.flatten(), mlt.flatten(), var.flatten(), cmap=cmap, levels=clvls)
        ax.set_title(offset)
        
        time_ = time + timedelta(hours=offset)
        pax.coastlines(time_, mag=Apex(time_))
                
    plt.suptitle(f"{time}")
    plt.savefig(fn, bbox_inches='tight')
    plt.close('all')
    plt.ion()











































