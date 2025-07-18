#%% Import

import os
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot
import h5py
from datetime import datetime
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

for key in dat[0].keys():
    print(key)
    print(dat[0][key].shape)


#%% Define grid

print('Defining CS grid')

position = (270, 79) # lon, lat for center of the grid
orientation = 0.
L = 45e6
Lres = 60e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6380e3 + 120e3)

#%% Convert to magnetic

for i in tqdm(range(len(dat)), total=len(dat)):
    time = dat[i]['time']    
    dpl = Dipole(epoch=time.year)
    
    _, _, Je, Jn = dpl.geo2mag(dat[i]['lat'], dat[i]['lon'], dat[i]['Je'], dat[i]['Jn'])
    dat[i]['Je'], dat[i]['Jn'] = Je, Jn
    
    _, _, Be, Bn = dpl.geo2mag(dat[i]['lat'], dat[i]['lon'], dat[i]['Be'], dat[i]['Bn'])
    dat[i]['Be'], dat[i]['Bn'] = Be, Bn

#%%

for key in tqdm(['SH', 'SP', 'FAC', 'JHe', 'JHn', 'JPe', 'JPn', 'Be', 'Bn', 'Bu'], total=10):
    folder = os.path.join(path_out, 'plot_Gamera', key)
    os.makedirs(folder, exist_ok=True)
    for i in range(len(dat)):
        fn = os.path.join(folder, f'{i}.png')
        
        time = dat[i]['time']
        
        dpl = Dipole(epoch=time.year)
        apx = Apex(time)
        
        mlat, mlon = dpl.geo2mag(dat[i]['lat'], dat[i]['lon'])
        mlt = dpl.mlon2mlt(mlon, dat[i]['time'])
        var = dat[i][key]
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
        pax.contourf(mlat, mlt, var, cmap=cmap, levels=clvls)
        plt.title(dat[i]['time'])
        plt.savefig(fn, bbox_inches='tight')
        plt.close('all')
        plt.ion()



