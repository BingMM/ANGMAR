#%% Import

import os
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot
import h5py
from datetime import datetime
from dipole import Dipole
from tqdm import tqdm

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', 'ionospheric_data_from_Gamera.h5')
path_out = os.path.join(base, 'figures')

#%% Import data

keys = ['lat', 'lon', 'SH', 'SP', 'FAC', 'JHe', 'JHn', 'JPe', 'JPn', 'Be', 'Bn', 'Bu']
dat = []
with h5py.File(path_in, 'r') as hf:
    for step in hf.values():
        record = {key: step[key][()] for key in keys}
        record['time'] = datetime.fromisoformat(step.attrs['time'])
        dat.append(record)

#%%

for i in tqdm(range(len(dat)), total=len(dat)):
    dpl = Dipole(epoch=dat[i]['time'].year)
    mlat, mlon = dpl.geo2mag(dat[i]['lat'], dat[i]['lon'])
    mlt = dpl.mlon2mlt(mlon, dat[i]['time'])
    var = dat[i]['FAC']
    vmax = np.max(abs(var))
    clvls = np.linspace(-vmax, vmax, 40)

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    pax = Polarplot(ax)
    pax.contourf(mlat, mlt, dat[i]['FAC'], cmap='bwr', levels=clvls)
    plt.title(dat[i]['time'])



