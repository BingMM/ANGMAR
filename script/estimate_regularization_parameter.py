#%% Import

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from secsy.cubedsphere import CSgrid, CSprojection
from secsy import get_SECS_B_G_matrices
import scipy
from tqdm import tqdm
from kneed import KneeLocator
import matplotlib.pyplot as plt

#%% Define grid

position = (23.5, 68.5)
orientation = 0
L = 650e3
W = 650e3
Lres = 20e3
Wres = 20e3
RI = 6371.2e3 + 110e3

grid = CSgrid(CSprojection(position, orientation), L, W, Lres, Lres, R = RI)

#%% Load magnetometer names and location

st_data = pd.read_csv('/home/bing/Dropbox/work/code/repos/ANGMAR/data/20250717-14-57-supermag-stations.csv', usecols=range(6))

lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()

f = grid.ingrid(lon, lat)
lon = lon[f]
lat = lat[f]
IAGA = st_data['IAGA'].to_numpy()[f]

#%% Find magnetometers inside the analysis region

# Radius of analysis circle
radius = 250e3

# Grid resolution
colat = 90 - grid.lat
lon_ = grid.lon
d2r = np.pi/180

x = grid.R * np.sin(colat*d2r) * np.cos(lon_*d2r)
y = grid.R * np.sin(colat*d2r) * np.sin(lon_*d2r)
z = grid.R * np.cos(colat*d2r)
del colat, lon_, d2r

res_xi = np.median(np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2 + np.diff(z, axis=1)**2))
res_eta = np.median(np.sqrt(np.diff(x, axis=0)**2 + np.diff(y, axis=0)**2 + np.diff(z, axis=0)**2))

# convert radius to xi eta units
AG_radius = radius / res_xi * grid.dxi
del res_xi, res_eta

# Find mags
xi, eta = grid.projection.geo2cube(lon, lat)
IAGA_AG = IAGA[np.sqrt(xi**2 + eta**2) <= AG_radius]
del xi, eta

#%% How many stations do we have and how much of the time do we have all of them

# Open the file
filepath = '/home/bing/Dropbox/work/data/all_stations_all2024.netcdf'
ds = Dataset(filepath, mode='r')

all_sm_st = ds['id'][0, :] # All stations in the supermag data

AG_IAGA_f = np.isin(all_sm_st, IAGA_AG) # Bool of all stations in the Supermag data that are in the AG

st_list = all_sm_st[AG_IAGA_f] # The station names in the AG

counts = np.zeros(st_list.size)

Be = ds['dbe_geo'][:].filled(np.nan)[:, AG_IAGA_f]
Bn = ds['dbn_geo'][:].filled(np.nan)[:, AG_IAGA_f]
Bu = ds['dbz_geo'][:].filled(np.nan)[:, AG_IAGA_f]

nan_f = np.isnan(Be) | np.isnan(Bn) | np.isnan(Bu)

number_of_st_per_min = np.sum(~nan_f, axis=1)

for i in range(11):
    print(f'#{i}: {np.sum(number_of_st_per_min == i)} : {np.sum(number_of_st_per_min == i)/number_of_st_per_min.size*100}')

#0: 311275 : 59.06098208864602
#1: 13108 : 2.487097753491196
#2: 1056 : 0.20036429872495445
#3: 0 : 0.0
#4: 0 : 0.0
#5: 55 : 0.01043564055859138
#6: 1760 : 0.33394049787492414
#7: 1187 : 0.22522009714632663
#8: 27195 : 5.159949908925319
#9: 76703 : 14.553544323011536
#10: 94701 : 17.96846539162113

# 60% of the time there is not data at all...
# 18% of the time all stations are present.

#%% How much of the time do we have all the stations in the MG?

IAGA_f = np.isin(all_sm_st, IAGA) # Bool of all stations in the Supermag data that are in the AG

st_list = all_sm_st[IAGA_f] # The station names in the AG

counts = np.zeros(st_list.size)

Be = ds['dbe_geo'][:].filled(np.nan)[:, IAGA_f]
Bn = ds['dbn_geo'][:].filled(np.nan)[:, IAGA_f]
Bu = ds['dbz_geo'][:].filled(np.nan)[:, IAGA_f]

nan_f = np.isnan(Be) | np.isnan(Bn) | np.isnan(Bu)

number_of_st_per_min = np.sum(~nan_f, axis=1)

for i in range(st_list.size + 1):
    print(f'#{i}: {np.sum(number_of_st_per_min == i)} : {np.sum(number_of_st_per_min == i)/number_of_st_per_min.size*100}')

#0: 311275 : 59.06098208864602
#1: 13108 : 2.487097753491196
#2: 1056 : 0.20036429872495445
#3: 0 : 0.0
#4: 0 : 0.0
#5: 0 : 0.0
#6: 55 : 0.01043564055859138
#7: 1760 : 0.33394049787492414
#8: 1065 : 0.20207194899817849
#9: 1442 : 0.2736035215543412
#10: 2881 : 0.5466378263509412
#11: 2032 : 0.38554948391013966
#12: 28770 : 5.458788706739527
#13: 69188 : 13.12765634486946
#14: 94408 : 17.912871888281725

# Again 18 percent of the time we have all 14 stations inside the MG
# We will base the analysis on these 17 stations

#%% Get data

ds.close()

Be = Be[number_of_st_per_min == st_list.size]
Bn = Bn[number_of_st_per_min == st_list.size]
Bu = Bu[number_of_st_per_min == st_list.size]

#%% Get G matrix

lat_st = np.array([lat[IAGA == st] for st in st_list])
lon_st = np.array([lon[IAGA == st] for st in st_list])
r_st = np.ones(lat_st.size)*6371.2e3

singularity_limit = np.min([grid.Wres, grid.Lres])/2

Ge, Gn, Gu = get_SECS_B_G_matrices(lat_st, lon_st, r_st, grid.lat, grid.lon, RI=RI,
                                   current_type = 'divergence_free',
                                   singularity_limit=singularity_limit)
G = np.vstack((Ge, Gn, Gu))
Qinv = np.diag(np.ones(3*lat_st.size) / ((1e-9)**2))
GTQ = G.T.dot(Qinv)
GTQG = GTQ.dot(G)
gmag = np.median(np.diag(GTQG))

#%% Regularization

lmin = -5
lmax = 5
lsize = 1000
lrange = np.linspace(lmin, lmax, lsize)

#%% Calc matrices

P = []
gcv_denom = []
for l in tqdm(lrange, total = lsize):
    Pl = scipy.linalg.lstsq(GTQG + 10**l * gmag * np.eye(GTQG.shape[0]), GTQ)[0]
    P.append(Pl)
    
    gcv_denom.append(np.sum(1 - np.diag(G.dot(Pl)))**2)

#%% Calculate residual and model norm

rnorm = np.zeros((Be.shape[0], lsize))
mnorm = np.zeros((Be.shape[0], lsize))
gcv = np.zeros((Be.shape[0], lsize))

for i in tqdm(range(rnorm.shape[0]), total=rnorm.shape[0]):
    d = np.hstack((Be[i], Bn[i], Bu[i])) * 1e-9
    for j in range(rnorm.shape[1]):
        m = P[j].dot(d)
        
        mnorm[i, j] = m.T.dot(m)
        
        res = d - G.dot(m)
        rnorm[i, j] = res.T.dot(Qinv).dot(res)
        
        gcv[i, j] = rnorm[i, j] / gcv_denom[j]

#%% Find optimal l with gcv for individual time steps

ls = np.zeros(rnorm.shape[0])
for i in tqdm(range(rnorm.shape[0]), total=rnorm.shape[0]):    
    ls[i] = lrange[np.argmin(gcv[i])]

#%% Plot histogram

fs = 14

plt.figure(figsize=(15,10))
ax = plt.gca()

hist = plt.hist(ls, bins=lrange, density=True, alpha=.9)

plt.xlabel('x   ($\lambda$=10$^x$)', fontsize=fs)
plt.ylabel('Probability density', fontsize=fs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim(lrange[2], lrange[-2])
plt.ylim(0, .5)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)

opt_id = np.argmax(hist[0][1:-1]) + 1
x = hist[1][opt_id]
y = hist[0][opt_id]
plt.vlines(x, 0, y, color='tab:orange')

plt.title(f'Mode at x={x}', fontsize=1.5*fs)

#%% Plot gcv examples

fig, axs = plt.subplots(3, 4, figsize=(20, 15), sharex=True)
for i, ax in enumerate(axs.flatten()):
    ii = i*100
    
    gcv_ = gcv[ii]    
    
    gcv_id = np.argmin(gcv_)
    
    ax.plot(lrange, gcv_)
    ax.plot(lrange[gcv_id], gcv_[gcv_id], '.')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax in axs[-1, :]:
    ax.set_xlabel('x   ($\lambda$=10$^x$)')

for ax in axs[:, 0]:
    ax.set_ylabel('GCV score')

#%% Plot median

plt.figure(figsize=(15,10))
ax = plt.gca()

median = np.median(gcv, axis=0)

plt.plot(lrange, median)

plt.xlabel('x   ($\lambda$=10$^x$)', fontsize=fs)
plt.ylabel('Median GCV score', fontsize=fs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)

opt_id = np.argmin(median)
x = lrange[opt_id]
y = median[opt_id]
plt.plot(x, y, '*', color='tab:orange', markersize=10)

plt.title(f'Minimum at x={x}', fontsize=1.5*fs)

#%% Plot mean

plt.figure(figsize=(15,10))
ax = plt.gca()

median = np.mean(gcv, axis=0)

plt.plot(lrange, median)

plt.xlabel('x   ($\lambda$=10$^x$)', fontsize=fs)
plt.ylabel('Mean GCV score', fontsize=fs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs)

opt_id = np.argmin(median)
x = lrange[opt_id]
y = median[opt_id]
plt.plot(x, y, '*', color='tab:orange', markersize=10)

plt.title(f'Minimum at x={x}', fontsize=1.5*fs)
