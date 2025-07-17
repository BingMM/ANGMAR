#%% Import

import os
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot, subsol
import h5py
from datetime import datetime
from tqdm import tqdm

import lompe
from scipy.interpolate import griddata

#%% Paths

print('Setting paths')

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', 'ionospheric_data_from_Gamera.h5')
path_out = os.path.join(base, 'figures')

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
Lres = 100e3#30e3
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6380e3 + 120e3)

print('{} GB in single precision'.format(int(grid.xi_mesh.size**2 * 4 / 1024**3)))
print('{} GB in double precision'.format(int(grid.xi_mesh.size**2 * 8 / 1024**3)))

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

#%% Fake E data

#%% Define conductance function

SH_funs, SP_funs = [], []
for i in tqdm(range(len(dat_int)), total=len(dat_int), desc='Creating conductance functions'):
    SH_fun = lambda lon, lat, i=i: dat_int[i]['SH']
    SP_fun = lambda lon, lat, i=i: dat_int[i]['SP']
    SH_funs.append(SH_fun)
    SP_funs.append(SP_fun)

#%% Initiate model

print('\nInitiating Lompe model')

emodel = lompe.Emodel(grid, Hall_Pedersen_conductance=(SH_funs[0], SP_funs[0]), dipole = True, epoch=time.year)

#%% Start with one model

i = 0

print('Creating data object(s)')

rs = np.full(grid.lat.size, 6380e3 + 1000e3)
FAC_data = lompe.Data(dat_int[i]['FAC'].flatten() * grid.R / rs.flatten()[0], np.vstack((grid.lon.flatten(), grid.lat.flatten())), datatype = 'fac')

#E_data = lompe.Data(np.vstack((dat_c['E_e'].to_numpy(), dat_c['E_n'].to_numpy())), np.vstack((dat_c['lon'].to_numpy(), dat_c['lat'].to_numpy())), datatype = 'Efield')

print('Passing data object(s)')
emodel.clear_model()
#emodel.add_data(E_data)
emodel.add_data(FAC_data)

print('Running inversion')
gtg, gtd = emodel.run_inversion(l1 = 0, l2 = 0)

#%% Plot it

lompe.lompeplot(emodel)
plt.savefig(os.path.join(path_out, 'lompe_test.png'), bbox_inches='tight')
plt.close('all')

#%% Loop over model













