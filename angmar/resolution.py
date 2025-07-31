#%% Import

import numpy as np
from secsy import get_SECS_B_G_matrices
import scipy

#%% Resolution

class Resolution:
    def __init__(self, lat, lon, r, q, lreg, MG, AG):
        self.lat = lat
        self.lon = lon
        
        if isinstance(r, np.ndarray):
            self.r = r
        else:
            self.r = np.ones(self.lat.size)*r
        
        if isinstance(q, np.ndarray):
            self.q = q
        else:
            self.q = np.ones(3*self.lat.size)*q
        
        self.lreg = lreg
        self.MG = MG
        self.AG = AG

    def compute_resolution_matrix(self):
        """Returns a 2D array of resolution values on the model grid."""
        
        Ge, Gn, Gu = get_SECS_B_G_matrices(self.lat, self.lon, self.r, 
                                           self.MG.grid.lat, self.MG.grid.lon, RI=self.MG.grid.R,
                                           current_type = 'divergence_free',
                                           singularity_limit=self.MG.singularity_limit)
        
        G = np.vstack((Ge, Gn, Gu))
        Qinv = np.diag(1 / self.q**2)
        
        GTQG = G.T.dot(Qinv).dot(G)
        gmag = np.median(np.diag(GTQG))
        #self.R = np.linalg.lstsq(GTQG + self.lreg * gmag * np.eye(GTQG.shape[0]), GTQG)[0]
        self.R = scipy.linalg.lstsq(GTQG + self.lreg * gmag * np.eye(GTQG.shape[0]), GTQG, lapack_driver='gelsy')[0]
    
    def compute_resolution(self):
        
        # Left right function
        def left_right(PSF_i, fraq=0.5):
        
            i_max = np.argmax(PSF_i)    
            PSF_max = PSF_i[i_max]
            
            j = 0
            i_left = 0
            left_edge = True
            while (i_max - j) >= 0:
                if PSF_i[i_max - j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
                    dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
                    i_left = i_max - j + dx
                
                    left_edge = False
                
                    break
                else:
                    j += 1

            j = 0
            i_right = len(PSF_i) - 1
            right_edge = True
            while (i_max + j) < len(PSF_i):
                if PSF_i[i_max + j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
                    dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
                    i_right = i_max + j - 1 + dx 
                
                    right_edge = False
                
                    break
                else:
                    j += 1
        
            flag = True
            if left_edge and right_edge:
                print('I think something is wrong')
                flag = False
            elif left_edge:
                i_left = i_max - (i_right - i_max)
                flag = False
            elif right_edge:
                i_right = i_max + (i_max - i_left)
                flag = False
        
            return i_left, i_right, i_max, flag
        
        # Allocate space
        xiRes = np.zeros(self.MG.grid.shape)
        etaRes = np.zeros(self.MG.grid.shape)
        xiResFlag = np.zeros(self.MG.grid.shape)
        etaResFlag = np.zeros(self.MG.grid.shape)
        resL = np.zeros(self.MG.grid.shape)
        
        # Loop over all PSFs
        for i in range(xiRes.size):
                        
            row = i//xiRes.shape[1]
            col = i%xiRes.shape[1]
            
            PSF = abs(self.R[:, i]).reshape(self.MG.grid.shape)
            
            ii = np.argmax(PSF)
            rowPSF = ii//self.MG.grid.shape[1]
            colPSF = ii%self.MG.grid.shape[1]
            
            dxi = abs(colPSF - col) * self.MG.res_xi
            deta = abs(rowPSF - row) * self.MG.res_eta
            
            resL[row, col] = np.sqrt(dxi**2 + deta**2)
            
            PSF_xi = np.sum(PSF, axis=0)
            i_left, i_right, i_max, flag = left_right(PSF_xi)
            xiRes[row, col] = self.MG.res_xi * (i_right - i_left)
            xiResFlag[row, col] = flag
            
            PSF_eta = np.sum(PSF, axis=1)
            i_left, i_right, i_max, flag = left_right(PSF_eta)
            etaRes[row, col] = self.MG.res_eta * (i_right - i_left)
            etaResFlag[row, col] = flag
        
        self.xiRes = xiRes
        self.etaRes = etaRes
        self.xiResFlag = xiResFlag
        self.etaResFlag = etaResFlag
    
    def extract_analysis_resolution(self, analysis_grid):
        """Masks the resolution to the analysis grid region."""
        pass