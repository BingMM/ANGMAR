#%% Import

import os
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot, subsol
import h5py
from datetime import datetime
from tqdm import tqdm

from dipole import Dipole
from apexpy import Apex

import lompe
from scipy.interpolate import griddata

#%% Paths

print('Setting paths')

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', 'ionospheric_data_from_Gamera.h5')
path_out = os.path.join(base, 'data', 'ionospheric_data_from_Lompe.h5')

#%% Import data

print('Loading Gamera data')

keys = ['lat', 'lon', 'SH', 'SP', 'FAC']
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
Lres = 200e3#30e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6380e3 + 120e3)

print('{} GB in single precision'.format(np.round(grid.xi_mesh.size**2 * 4 / 1024**3, 2)))
print('{} GB in double precision'.format(np.round(grid.xi_mesh.size**2 * 8 / 1024**3, 2)))

#%% Illustrate grid and data to see overlap

print('Making plot to check grid and data overlap')

i = 0
time = dat[i]['time']

lat = dat[i]['lat']

def lon2lt(lon, time):
    sslat, sslon = subsol(time)
    londiff = (lon - sslon + 180) % 360 - 180
    lon = (180. + londiff)
    return lon/15
    
lon = dat[i]['lon']
lt = lon2lt(lon, time)

var = dat[i]['FAC']
vmax = np.max(abs(var))
clvls = np.linspace(-vmax, vmax, 40)
cmap = 'bwr'

plt.ioff()
fig = plt.figure(figsize=(10,10))
ax = plt.gca()
pax = Polarplot(ax, minlat=30)

pax.contourf(lat, lt, var, cmap=cmap, levels=clvls)

pax.plot(grid.lat_mesh[0,  :], lon2lt(grid.lon_mesh[0,  :], time), color='k', linewidth=3)
pax.plot(grid.lat_mesh[-1, :], lon2lt(grid.lon_mesh[-1, :], time), color='k', linewidth=3)
pax.plot(grid.lat_mesh[:,  0], lon2lt(grid.lon_mesh[:,  0], time), color='k', linewidth=3)
pax.plot(grid.lat_mesh[:, -1], lon2lt(grid.lon_mesh[:, -1], time), color='k', linewidth=3)

pax.coastlines(time)
plt.title(time)

fn = os.path.join(base, 'figures', 'grid_size_check.png')
plt.savefig(fn, bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Interpolate data

def interp2grid(grid, lon, lat, data, mesh=False):        
    xiG, etaG = grid.projection.geo2cube(lon, lat)
    if mesh:
        datainterp = griddata((xiG.flatten(), etaG.flatten()), data.flatten(), 
                          (grid.xi_mesh.flatten(), grid.eta_mesh.flatten()), 
                          method='cubic')
        return datainterp.reshape(grid.xi_mesh.shape)
    else:
        datainterp = griddata((xiG.flatten(), etaG.flatten()), data.flatten(), 
                          (grid.xi.flatten(), grid.eta.flatten()), 
                          method='cubic')
        return datainterp.reshape(grid.shape)

dat_int = []
for i in tqdm(range(len(dat)), total=len(dat), desc='Interpolating data to CS'):
    SH_int  = interp2grid(grid, dat[i]['lon'], dat[i]['lat'], dat[i]['SH'], mesh=False)
    SP_int  = interp2grid(grid, dat[i]['lon'], dat[i]['lat'], dat[i]['SP'], mesh=False)
    FAC_int = interp2grid(grid, dat[i]['lon'], dat[i]['lat'], dat[i]['FAC'])
    
    dat_int.append({'time': dat[i]['time'],
                    'SH': SH_int, 'SP': SP_int, 'FAC': FAC_int}
                   )

#%% Define conductance function

SH_funs, SP_funs = [], []
for i in tqdm(range(len(dat_int)), total=len(dat_int), desc='Creating conductance functions'):
    SH_fun = lambda lon, lat, i=i: dat_int[i]['SH'].flatten()
    SP_fun = lambda lon, lat, i=i: dat_int[i]['SP'].flatten()
    SH_funs.append(SH_fun)
    SP_funs.append(SP_fun)

#%% Initiate model

print('\nInitiating Lompe model')

emodel = lompe.Emodel(grid, Hall_Pedersen_conductance=(SH_funs[0], SP_funs[0]), dipole = True, epoch=time.year)

#%% Loop

output = []
loop = tqdm(range(len(dat_int)), total=len(dat_int), desc='Lompe loop')
for i in loop:
    
    loop.set_description(f"[{i}] Resetting emodel".ljust(30))
    emodel.clear_model(Hall_Pedersen_conductance=(SH_funs[i], SP_funs[i]))

    loop.set_description(f"[{i}] Creating data object(s)".ljust(30))
    rs = np.full(grid.lat.size, 6380e3 + 1000e3)
    FAC_data = lompe.Data(dat_int[i]['FAC'].flatten() * grid.R / rs.flatten()[0], 
                          np.vstack((grid.lon.flatten(), grid.lat.flatten())), 
                          datatype = 'fac', error=1e-5, iweight=1)

    lon = np.hstack((grid.lon[0, :], grid.lon[-1, :], grid.lon[:, 0], grid.lon[-1, :]))
    lat = np.hstack((grid.lat[0, :], grid.lat[-1, :], grid.lat[:, 0], grid.lat[-1, :]))
    Ee, En = np.zeros(len(lon)), np.zeros(len(lon))
    E_data = lompe.Data(np.vstack((Ee, En)), np.vstack((lon, lat)), datatype = 'Efield', error=0.003, iweight=1)

    loop.set_description(f"[{i}] Passing data object(s)".ljust(30))
    emodel.add_data(E_data)
    emodel.add_data(FAC_data)

    loop.set_description(f"[{i}] Running inversion".ljust(30))
    _, _ = emodel.run_inversion(l1 = 0, l2 = 0)

    loop.set_description(f"[{i}] Storing relevant data".ljust(30))
    FAC = emodel.FAC().reshape(grid.shape)
    Je, Jn = emodel.j()
    Be, Bn, Bu = emodel.B_ground()
    S = emodel._B_df_matrix(return_poles=True)
    output.append({'time': dat_int[i]['time'],
                   'SH': dat[i]['SH'], 'SP': dat[i]['SP'],
                   'FAC_int': dat[i]['FAC'],
                   'FAC': FAC, 'S': S,
                   'Je': Je, 'Jn': Jn,
                   'Be': Be, 'Bn': Bn, 'Bu': Bu}
                 )

#%% Save to h5

print('Saving to file')
# Create HDF5 file
with h5py.File(path_out, 'w') as hf:
    for i, entry in enumerate(output):
        grp = hf.create_group(f"step_{i:04d}")

        # Save metadata (time) as an attribute
        grp.attrs['time'] = entry['time'].isoformat()

        # Save all array fields
        for key, val in entry.items():
            if key != 'time':
                grp.create_dataset(key, data=val)
