#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polplot import Polarplot
import lompe
from apexpy import Apex

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', '20250717-14-57-supermag-stations.csv')
path_out = os.path.join(base, 'figures')

#%% Import gmag location

st_data = pd.read_csv(path_in, usecols=range(6))

lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()[lat>0]
lat = lat[lat>0]
lt = lon / 15

#%% Define grids

print('Defining CS grid')

position = (25, 68) # lon, lat for center of the grid
orientation = 8
L = 1.1e6
Lres = 10e3#60e3
grid_s = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)

position = (210, 66) # lon, lat for center of the grid
orientation = -12
L = 1.4e6
Lres = 10e3#60e3
grid_a = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, L, Lres, Lres, R = 6371.2e3 + 110e3)

position = (255, 62) # lon, lat for center of the grid
orientation = -12
L = 3e6
W = 2.4e6
Lres = 10e3#60e3
grid_c = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Lres, R = 6371.2e3 + 110e3)

#%% Plot all stations

plt.ioff()
fig = plt.figure(figsize=(15, 15))
ax = plt.gca()
pax = Polarplot(ax)
pax.coastlines(color='k', linewidth=2, zorder=1)
pax.coastlines(color='cyan', zorder=2)
pax.scatter(lat, lt, color='tab:orange', zorder=3)

apex = Apex(date=2013)
for la in np.arange(30, 90, 10):
    la, lo, _ = apex.apex2geo(np.ones(1000)*la, np.linspace(0, 360, 1000), height=0)
    pax.plot(la, (lo/15)%24, color='tab:red', zorder=0)
for lo in np.arange(0, 360, 30):
    la, lo, _ = apex.apex2geo(np.linspace(0, 90, 1000), np.ones(1000)*lo, height=0)
    pax.plot(la, (lo/15)%24, color='tab:red', zorder=0)

for grid in [grid_s, grid_a, grid_c]:
    pax.plot(grid.lat_mesh[0,  :], (grid.lon_mesh[0,  :]/15)%360, color='k', linewidth=3)
    pax.plot(grid.lat_mesh[-1, :], (grid.lon_mesh[-1, :]/15)%360, color='k', linewidth=3)
    pax.plot(grid.lat_mesh[:,  0], (grid.lon_mesh[:,  0]/15)%360, color='k', linewidth=3)
    pax.plot(grid.lat_mesh[:, -1], (grid.lon_mesh[:, -1]/15)%360, color='k', linewidth=3)

plt.savefig(os.path.join(path_out, 'all_stations.png'))
plt.close('all')
plt.ion()
