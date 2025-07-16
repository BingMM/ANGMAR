#%% Import

import os
import matplotlib.pyplot as plt
from pickle import load
from polplot import Polarplot

#%% Paths

cwd = os.getcwd()
#base = os.path.abspath(os.path.join(cwd, '..'))
base = cwd
path_in = os.path.join(base, 'data', 'ionospheric_data_from_Gamera.pkl')
path_out = os.path.join(base, 'figures')

#%% Import data

with open(path_in, 'rb') as file:
    dat = load(file)

'''
print(dat[0].keys())

#%%

fig = plt.figure(figsize=(10,10))
ax = plt.gca()
pax = Polarplot(ax)
'''