#%% Import

import os
import numpy as np
import kaipy.gamera.magsphere as msph
import kaipy.remix.remix as remix

from tqdm import tqdm
import dipole
import datetime

from multiprocessing import Pool, cpu_count
from functools import partial

import h5py

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
    '''
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
    '''
    
    # 1. Precompute xyz as before
    from kaipy.kdefs import RionE, REarth
    r = REarth * 1.e-6 / RionE
    x = r * np.sin(np.deg2rad(90 - mlat)) * np.cos(np.deg2rad(lon))
    y = r * np.sin(np.deg2rad(90 - mlat)) * np.sin(np.deg2rad(lon))
    z = r * np.cos(np.deg2rad(90 - mlat))
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    N = xyz.shape[0]

    # 2. Split xyz into chunks
    chunk_size = 10
    xyz_chunks = [xyz[i:i + chunk_size] for i in range(0, N, chunk_size)]

    # 3. Define a wrapper for ion.dB
    def compute_dB(chunk, ion):
        return ion.dB(chunk)

    # 4. Use multiprocessing
    # This part is tricky: ion must be accessible inside the worker
    # Workaround: use a global variable + initializer

    # Global placeholder
    _global_ion = None

    def init_worker(ion_instance):
        global _global_ion
        _global_ion = ion_instance

    def compute_dB_worker(chunk):
        return _global_ion.dB(chunk)

    # 5. Run with multiprocessing
    if __name__ == "__main__":
        nproc = min(cpu_count(), 48)  # don't use all 192 unless needed

        with Pool(processes=nproc, initializer=init_worker, initargs=(ion,)) as pool:
            results = list(tqdm(pool.imap(compute_dB_worker, xyz_chunks), total=len(xyz_chunks), leave=False))

        # 6. Unpack results
        dBr_list, dBtheta_list, dBphi_list = zip(*results)
        dBr = np.concatenate(dBr_list)
        dBtheta = np.concatenate(dBtheta_list)
        dBphi = np.concatenate(dBphi_list)
    
    _  , _  , dBe, dBn = dpl.mag2geo(mlat, mlon, dBphi.reshape(x.shape), -dBtheta.reshape(x.shape))
    dBu = dBr.reshape(x.shape)
    
    # Save it
    dat.append({'time': dt, 'lat': lat, 'lon': lon,
                'SH': SH, 'SP': SP,
                'FAC': FAC,
                'JHe': JHe, 'JHn': JHn,
                'JPe': JPe, 'JPn': JPn,
                'Be': dBe, 'Bn': dBn, 'Bu': dBu}
               )

#%% Save it

# Create HDF5 file
with h5py.File(path_out + 'ionospheric_data_from_Gamera.h5', 'w') as hf:
    for i, entry in enumerate(dat):
        grp = hf.create_group(f"step_{i:04d}")

        # Save metadata (time) as an attribute
        grp.attrs['time'] = entry['time'].isoformat()

        # Save all array fields
        for key, val in entry.items():
            if key != 'time':
                grp.create_dataset(key, data=val)


