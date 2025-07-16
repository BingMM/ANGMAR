#%% Import

import os
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

#%% Path

path_in = '/disk/Gamera_Kareem/'
path_out = '/Home/siv32/mih008/repos/ANGMAR/figures/Kaipy_plots/'
        
#%% Load Gamera

# Define simulation directory
ftag = 'msphere'

# Initiate magnetosphere
gsph = msph.GamsphPipe(path_in,ftag,doFast=False)
nt = len(gsph.UT)
#nt = gsph.sFin-gsph.s0

# Remix files
mixFiles = os.path.join(path_in,"%s.mix.h5"%(ftag))

for i in tqdm(range(0, nt), total=nt): # skipping the first because TIEGCM does not have the first time step
    # get time index, date time, epoch and init dipole
    t = gsph.s0 + i
    
    # init ionosphere
    ion = remix.remix(mixFiles, t)
    ion.init_vars('NORTH')

    ion.plot('current')
    
    plt.savefig(path_out + f'{i}.png', bbox_inches='tight')
    plt.close('all')
