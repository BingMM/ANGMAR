#%% Import

import os
import numpy as np
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix

from tqdm import tqdm
import dipole
import datetime

import pickle

#%% Fun

def datetime_to_decimal_year(dt):
    year_start = datetime.datetime(dt.year, 1, 1)
    next_year_start = datetime.datetime(dt.year + 1, 1, 1)
    year_length = (next_year_start - year_start).total_seconds()
    seconds_since_start = (dt - year_start).total_seconds()
    return dt.year + seconds_since_start / year_length

#%% Path

path_in = '/disk/Gamera_Kareem/'
path_out = '/Home/siv32/mih008/repos/ANGMAR/data/'
        
#%% Load Gamera

# Define simulation directory
ftag = 'msphere'

# Initiate magnetosphere
gsph = msph.GamsphPipe(path_in,ftag,doFast=False)
nt = len(gsph.UT)
#nt = gsph.sFin-gsph.s0

# Remix files
mixFiles = os.path.join(path_in,"%s.mix.h5"%(ftag))

# Allocate space
SH_all,   SP_all        = [], []
FAC_all                 = []
JHe_all, JHn_all        = [], []
JPe_all, JPn_all        = [], []
Be_all, Bn_all, Bu_all  = [], [], []
dat = []
for i in tqdm(range(0, nt), total=nt): # skipping the first because TIEGCM does not have the first time step
    # get time index, date time, epoch and init dipole
    t = gsph.s0 + i
    dt = gsph.UT[i]
    epoch = datetime_to_decimal_year(dt)
    dpl = dipole.Dipole(epoch)
    
    # init ionosphere
    ion = remix.remix(mixFiles, t)

    # Get coordinates
    xc, yc, theta, phi = ion.cartesianCellCenters()
    mlat = 90 - theta/np.pi*180
    lon = phi/np.pi*180
    
    # Calculate mlt and calc mlon
    mlt = (lon/15 + 12)%24
    mlon = dpl.mlt2mlon(mlt, dt)
    
    ### NORTH
    ion.init_vars('NORTH')
    # Get variables    
    SH   = ion.variables['sigmah']['data']
    SP   = ion.variables['sigmap']['data']
    FAC  = ion.variables['current']['data']
    hc   = ion.hCurrents()
    JHn, JHe = -hc[6], hc[7]
    JPn, JPe = -hc[8], hc[9]
    
    # Convert to geographic
    lat, lon, JHe, JHn = dpl.mag2geo(mlat, mlon, JHe, JHn)
    _  , _  , JHe, JHn = dpl.mag2geo(mlat, mlon, JHe, JHn)
    
    from kaipy.kdefs import RionE, REarth
    r = REarth*1.e-6 / RionE # dB say to use units of Ri    
    x = r * np.sin(np.deg2rad(90-mlat)) * np.cos(np.deg2rad(lon))
    y = r * np.sin(np.deg2rad(90-mlat)) * np.sin(np.deg2rad(lon))
    z = r * np.cos(np.deg2rad(90-mlat))
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    N = xyz.shape[0]
    dBr_list = []
    dBtheta_list = []
    dBphi_list = []
    chunk_size = 10

    for j in tqdm(range(0, N, chunk_size), total=np.arange(0, N, chunk_size).size, leave=False):
        xyz_chunk = xyz[j:j+chunk_size]
        dBr_chunk, dBtheta_chunk, dBphi_chunk = ion.dB(xyz_chunk)
        dBr_list.append(dBr_chunk)
        dBtheta_list.append(dBtheta_chunk)
        dBphi_list.append(dBphi_chunk)

    dBr = np.concatenate(dBr_list)
    dBtheta = np.concatenate(dBtheta_list)
    dBphi = np.concatenate(dBphi_list)
    
    _  , _  , dBe, dBn = dpl.mag2geo(mlat, mlon, dBphi.reshape(x.shape), -dBtheta.reshape(x.shape))
    dBu = dBr.reshape(x.shape)
    
    #dBe, dBn, dBu = 1, 1, 1
    
    # Save it
    dat.append({'time': dt, 'lat': lat, 'lon': lon,
                'SH': SH, 'SP': SP,
                'FAC': FAC,
                'JHe': JHe, 'JHn': JHn,
                'JPe': JPe, 'JPn': JPn,
                'Be': dBe, 'Bn': dBn, 'Bu': dBu}
               )

#%% Save it

with open(path_out + 'ionospheric_data_from_Gamera.pkl', 'wb') as file:
    pickle.dump(dat, file)

