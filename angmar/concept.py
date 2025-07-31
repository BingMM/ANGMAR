#%% Import

import numpy as np
from secsy.cubedsphere import CSgrid, CSprojection
import matplotlib.pyplot as plt
from secsy import get_SECS_B_G_matrices
import os
import pandas as pd
import scipy
from scipy.optimize import dual_annealing

#%% Fixed sensor array

class FixedSensorArray:
    def __init__(self, lat, lon, precision, r=6371.2e3, grid = None):
        self.lat = lat
        self.lon = lon
        
        if isinstance(r, np.ndarray):
            self.r = r
        else:
            self.r = np.ones(self.lat.size)*r
        
        if isinstance(precision, np.ndarray):
            self.precision = precision
        else:
            self.precision = np.ones(self.lat.size)*precision
        
        self.grid = grid

        self._xi = None
        self._eta = None
    
    @property
    def xi(self):
        if self.grid == None:
            raise ValueError('Cannot call xi or eta without defining grid first')
        if self._xi is None:
            xi, eta = self.grid.projection.geo2cube(self.lon, self.lat)
            self._xi, self._eta = xi, eta
        return self._xi
    
    @property
    def eta(self):
        if self.grid == None:
            raise ValueError('Cannot call xi or eta without defining grid first')
        if self._eta is None:
            xi, eta = self.grid.projection.geo2cube(self.lon, self.lat)
            self._xi, self._eta = xi, eta
        return self._eta

#%% Model grid

class ModelGrid:
    def __init__(self, center_lat, center_lon, radius=6371.2e3 + 110e3, orientation=0, L=10e4, W=10e4, Lres=10e3, Wres=10e3):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.orientation = orientation
        self.L, self.W = L, W
        self.Lres, self.Wres = Lres, Wres
        
        self.generate_CS()
    
    def calculate_grid_resolution(self):
        # Get res in km
        colat = 90 - self.grid.lat
        lon = self.grid.lon
        d2r = np.pi/180

        x = self.radius * np.sin(colat*d2r) * np.cos(lon*d2r)
        y = self.radius * np.sin(colat*d2r) * np.sin(lon*d2r)
        z = self.radius * np.cos(colat*d2r)

        self.res_xi = np.median(np.sqrt(np.diff(x, axis=1)**2 + np.diff(y, axis=1)**2 + np.diff(z, axis=1)**2))
        self.res_eta = np.median(np.sqrt(np.diff(x, axis=0)**2 + np.diff(y, axis=0)**2 + np.diff(z, axis=0)**2))
    
    def generate_CS(self):
        
        position = (self.center_lon, self.center_lat)
        self.grid = CSgrid(CSprojection(position, self.orientation), 
                           self.L, self.W, self.Lres, self.Lres, R = self.radius)
        
        self.singularity_limit = np.min([self.grid.Wres, self.grid.Lres])/2
        
        self.calculate_grid_resolution()
    
    def illustrate_grid(self, FSA=None, AG=None, coastline=True, lim=2):
        fig = plt.figure(figsize=(10,10))
        for i in range(self.grid.xi_mesh.shape[0]):
            if i == 0 or i == self.grid.shape[0]:
                lw = 2
            else:
                lw = .5
            plt.plot(self.grid.xi_mesh[i, :], self.grid.eta_mesh[i, :], color='k', linewidth=lw)
            
        for i in range(self.grid.xi_mesh.shape[1]):
            if i == 0 or i == self.grid.shape[1]:
                lw = 2
            else:
                lw = .5
            plt.plot(self.grid.xi_mesh[:, i], self.grid.eta_mesh[:, i], color='k', linewidth=lw)

        if FSA is not None:            
            xi, eta = self.grid.projection.geo2cube(FSA.lon, FSA.lat)
            plt.plot(xi, eta, '*', color='tab:red', markersize=10)
        
        if AG is not None:
            xi, eta = self.grid.projection.geo2cube(AG.center_lon+0.001, AG.center_lat)

            sin = np.sin(np.linspace(0, 2*np.pi, 1000))
            cos = np.cos(np.linspace(0, 2*np.pi, 1000))
            radius = AG.radius / MG.res_xi * MG.grid.dxi
            xi = xi + radius * sin
            eta = eta + radius * cos
            plt.plot(xi, eta, '--', color='tab:blue')
        
        if coastline:
            coastlines = np.load('/home/bing/Dropbox/work/code/repos/ANGMAR/data/coastlines_50m.npz')
            for cl in coastlines:
                lat, lon = coastlines[cl]
                xi, eta = self.grid.projection.geo2cube(lon, lat)
                plt.plot(xi, eta, color='k', linewidth=2)
                plt.plot(xi, eta, color='cyan', linewidth=1)
        
        ximin, ximax = self.grid.xi_mesh.min(), self.grid.xi_mesh.max()
        etamin, etamax = self.grid.eta_mesh.min(), self.grid.eta_mesh.max()
        dxi, deta = (ximax-ximin)/2, (etamax-etamin)/2
        xic, etac = ximin + dxi, etamin + deta
        ximin, ximax = xic - dxi * lim, xic + dxi * lim
        etamin, etamax = etac - deta * lim, etac + deta * lim
        
        plt.xlim(ximin, ximax)
        plt.ylim(etamin, etamax)
        
        return fig

#%% Analysis grid class

class AnalysisGrid:
    def __init__(self, grid, center_lat=None, center_lon=None, radius=None):
        self.grid = grid
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius # Same units a grid.L, grid.W, grid.Lres etc.
        #self.radius = self._radius / grid.Lres * np.median(np.diff(grid.eta_mesh, axis=0)) # Radius in CS units        

#%% Candidate sensor array

class CandidateSensorArray:
    def __init__(self, N, precision):
        self.N = N
        self.precision = precision

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

#%% Sensor optimizer

class SensorOptimizer:
    def __init__(self, MG, AG, FSA, CSA):
        self.MG = MG
        self.AG = AG
        self.FSA = FSA
        self.CSA = CSA
        self.AG_radius = self.AG.radius / self.MG.res_xi * self.MG.grid.dxi
        
        self.latF = np.copy(self.FSA.lat)
        self.lonF = np.copy(self.FSA.lon)
        self.rF = np.copy(self.FSA.r)
        self.qF = np.tile(self.FSA.precision, (3, 1)).T.flatten()
        
        self.latC = None
        self.lonC = None
        self.rC = None
        self.qC = None
        
        self.lreg = 1e0

    def compute_resolution(self):
        
        lat = np.concatenate((self.latF, self.latC))
        lon = np.concatenate((self.lonF, self.lonC))
        r = np.concatenate((self.rF, self.rC))
        q = np.concatenate((self.qF, self.qC))
        
        self.Res = Resolution(lat, lon, r, q, self.lreg, self.MG, self.AG)
        self.Res.compute_resolution_matrix()
        self.Res.compute_resolution()
        self.res = (self.Res.xiRes + self.Res.etaRes)/2
        f = (self.Res.xiResFlag == 1) & (self.Res.etaResFlag == 1)
        resMax = np.max(self.res[f])
        self.res[~f] = resMax + 1
        rs = np.sqrt(self.MG.grid.xi**2 + self.MG.grid.eta**2)
        self.res[rs > self.AG_radius] = 0
        
    def compute_initial_guess(self):
        """Greedy initialization based on current max resolution."""
        
        self.latC = []
        self.lonC = []
        self.rC = []
        self.qC = []
        
        res = []
        for i in range(self.CSA.N):
            print('iter: ', i)
            self.compute_resolution()
            res.append(self.res)
            id_max = np.argmax(self.res)
            
            self.latC.append(self.MG.grid.lat.flatten()[id_max])
            self.lonC.append(self.MG.grid.lon.flatten()[id_max]+0.001)
            self.rC.append(self.rF[0])
            self.qC.extend([self.CSA.precision]*3)
        
        self.latC = np.array(self.latC)
        self.lonC = np.array(self.lonC)
        self.rC = np.array(self.rC)
        self.qC = np.array(self.qC)
        self.res_all = res

    def calculate_forces(self, xi, eta):
        dxi = self.MG.grid.xi - xi
        deta = self.MG.grid.eta - eta
        r = np.sqrt(dxi**2 + deta**2)
        
        Fxi = np.sum(self.res * dxi / r**3)
        Feta = np.sum(self.res * deta / r**3)
        
        return Fxi, Feta
    
    def calculate_forces2(self, xi, eta): # towards max
        maxid = np.argmax(self.res)
        dxi = self.MG.grid.xi.flatten()[maxid] - xi
        deta = self.MG.grid.eta.flatten()[maxid] - eta
        return dxi, deta
        
    def optimize_positions(self, max_iter=100):
        """PSO-like optimization algorithm."""
        
        self.lonCs = []
        self.latCs = []
        self.lonCs.append(np.copy(self.lonC))
        self.latCs.append(np.copy(self.latC))
        self.res_all_opt = []
        
        for i in range(max_iter):
            print(i)
            for j, (lati, loni) in enumerate(zip(self.latC, self.lonC)):
                xii, etai = self.MG.grid.projection.geo2cube(loni, lati)
                Fxi, Feta = self.calculate_forces(xii, etai)
                F = np.sqrt(Fxi**2 + Feta**2)
                xi = xii + Fxi / F * self.MG.grid.dxi
                eta = etai + Feta / F * self.MG.grid.deta
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.latC[j] = lat
                self.lonC[j] = lon
                self.compute_resolution()
            self.lonCs.append(np.copy(self.lonC))
            self.latCs.append(np.copy(self.latC))
            self.res_all_opt.append(np.copy(self.res))

#%%

class SensorOptimizer:
    def __init__(self, MG, AG, FSA, CSA):
        self.MG = MG
        self.AG = AG
        self.FSA = FSA
        self.CSA = CSA
        self.AG_radius = self.AG.radius / self.MG.res_xi * self.MG.grid.dxi
        
        self.latF = np.copy(self.FSA.lat)
        self.lonF = np.copy(self.FSA.lon)
        self.rF = np.copy(self.FSA.r)
        self.qF = np.tile(self.FSA.precision, (3, 1)).T.flatten()
        
        self.latC = None
        self.lonC = None
        self.rC = None
        self.qC = None
        
        self.lreg = 1e0

    def compute_resolution(self):
        
        lat = np.concatenate((self.latF, self.latC))
        lon = np.concatenate((self.lonF, self.lonC))
        r = np.concatenate((self.rF, self.rC))
        q = np.concatenate((self.qF, self.qC))
        
        self.Res = Resolution(lat, lon, r, q, self.lreg, self.MG, self.AG)
        self.Res.compute_resolution_matrix()
        self.Res.compute_resolution()
        self.res = (self.Res.xiRes + self.Res.etaRes)/2
        f = (self.Res.xiResFlag == 1) & (self.Res.etaResFlag == 1)
        resMax = np.max(self.res[f])
        self.res[~f] = resMax + 1
        rs = np.sqrt(self.MG.grid.xi**2 + self.MG.grid.eta**2)
        self.res[rs > self.AG_radius] = 0    

    def compute_max_resolution(self):        
        self.compute_resolution()
        return np.max(self.res)

    def compute_initial_guess(self):
        """Greedy initialization based on current max resolution."""
        
        self.latC = []
        self.lonC = []
        self.rC = []
        self.qC = []
        
        res = []
        for i in range(self.CSA.N):
            #print('iter: ', i)
            self.compute_resolution()
            res.append(self.res)
            id_max = np.argmax(self.res)
            
            self.latC.append(self.MG.grid.lat.flatten()[id_max])
            self.lonC.append(self.MG.grid.lon.flatten()[id_max]+0.001)
            self.rC.append(self.rF[0])
            self.qC.extend([self.CSA.precision]*3)
        
        self.latC = np.array(self.latC)
        self.lonC = np.array(self.lonC)
        self.rC = np.array(self.rC)
        self.qC = np.array(self.qC)
        self.res_all = res
        
    def optimize_positions(self):
        
        self.best_so_far = 1e11
        self.eval_count = 0
        
        # Define bounds for each sensor (2 coords per sensor)
        bounds = []
        for i in range(self.CSA.N):
            # Get bounds in xi/eta space (easier than lat/lon)
            bounds.extend([(-self.AG_radius, self.AG_radius),  # xi
                           (-self.AG_radius, self.AG_radius)]) # eta
        
        def objective(x):
            # Unpack positions
            for i in range(self.CSA.N):
                xi = x[2*i]
                eta = x[2*i + 1]
                
                # Skip if outside valid region
                #if xi**2 + eta**2 > self.AG_radius**2:
                #    return 1e10
                
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.lonC[i] = lon
                self.latC[i] = lat
            
            #return self.compute_max_resolution()
            
            obj_value = self.compute_max_resolution()
        
            self.eval_count += 1
            if obj_value < self.best_so_far:
                self.best_so_far = obj_value
                print(f"Eval {self.eval_count}: New best = {obj_value:.6f}")
            elif self.eval_count % 100 == 0:
                print(f"Eval {self.eval_count}: Current = {obj_value:.6f}, Best = {self.best_so_far:.6f}")
        
            return obj_value
        
        # Initial guess in xi/eta
        if self.latC is None:
            self.compute_initial_guess()
        elif len(self.latC) != self.CSA.N:
            self.compute_initial_guess()
            
        x0 = []
        for i in range(self.CSA.N):
            xi, eta = self.MG.grid.projection.geo2cube(self.lonC[i], self.latC[i])
            x0.extend([xi, eta])
        
        # Run optimization
        result = dual_annealing(objective, bounds, x0=x0, maxiter=100, no_local_search=True)
        #result = dual_annealing(objective, bounds, x0=x0, no_local_search=True)
        
        #from scipy.optimize import minimize
        #result = minimize(objective, x0=x0, method='Nelder-Mead', options={'maxfev': 1000, 'disp': True})
        
        # Extract final positions
        for i in range(self.CSA.N):
            xi = result.x[2*i]
            eta = result.x[2*i + 1]
            self.lonC[i], self.latC[i] = self.MG.grid.projection.cube2geo(xi, eta)

        
    def optimize_positions2(self, n_restarts=5, iters_per_restart=1000):
        """SA with multiple restarts from different initial conditions."""
        self.compute_initial_guess()        
        
        best_global = (self.latC.copy(), self.lonC.copy(), self.compute_max_resolution())
        
        bounds = []
        for i in range(self.CSA.N):
            # Get bounds in xi/eta space (easier than lat/lon)
            bounds.extend([(-self.AG_radius, self.AG_radius),  # xi
                           (-self.AG_radius, self.AG_radius)]) # eta
        
        def objective(x):
            # Unpack positions
            for i in range(self.CSA.N):
                xi = x[2*i]
                eta = x[2*i + 1]
                
                # Skip if outside valid region
                #if xi**2 + eta**2 > self.AG_radius**2:
                #    return 1e10
                
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.lonC[i] = lon
                self.latC[i] = lat
            
            #return self.compute_max_resolution()
            
            obj_value = self.compute_max_resolution()
        
            self.eval_count += 1
            if obj_value < self.best_so_far:
                self.best_so_far = obj_value
                print(f"Eval {self.eval_count}: New best = {obj_value:.6f}")
            elif self.eval_count % 100 == 0:
                print(f"Eval {self.eval_count}: Current = {obj_value:.6f}, Best = {self.best_so_far:.6f}")
        
            return obj_value
        
        for restart in range(n_restarts):
            self.eval_count = 0
            self.best_so_far = 1e11
            # Random restart (except first)
            if restart > 0:
                self.compute_initial_guess()  # Reset to greedy
                # Add random perturbation
                for i in range(self.CSA.N):
                    xi, eta = self.MG.grid.projection.geo2cube(self.lonC[i], self.latC[i])
                    xi += np.random.normal(0, self.AG_radius * 0.2)
                    eta += np.random.normal(0, self.AG_radius * 0.2)
                    r = np.sqrt(xi**2 + eta**2)
                    if r < self.AG_radius:
                        self.lonC[i], self.latC[i] = self.MG.grid.projection.cube2geo(xi, eta)
            
            # Run SA
            result = dual_annealing(objective, bounds, 
                                  maxiter=100,
                                  no_local_search=True,
                                  initial_temp=5000,  # Tune this based on your objective scale
                                  restart_temp_ratio=2e-5)
            
            if result.fun < best_global[2]:
                best_global = (self.latC.copy(), self.lonC.copy(), result.fun)
            
            print(f"Restart {restart}: {result.fun:.1f} (best: {best_global[2]:.1f})")
        
        self.latC, self.lonC = best_global[0], best_global[1]

    def optimize_positions2_fix(self, n_restarts=5, iters_per_restart=1000):
        """SA with multiple restarts from different initial conditions."""
        self.compute_initial_guess()        
        
        best_global = (self.latC.copy(), self.lonC.copy(), self.compute_max_resolution()*1e-3)
        
        bounds = []
        for i in range(self.CSA.N):
            bounds.extend([(-self.AG_radius, self.AG_radius),  # xi
                           (-self.AG_radius, self.AG_radius)]) # eta
        
        def objective(x):
            # Unpack positions
            for i in range(self.CSA.N):
                xi = x[2*i]
                eta = x[2*i + 1]
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.lonC[i] = lon
                self.latC[i] = lat
            
            obj_value = self.compute_max_resolution()*1e-3
        
            self.eval_count += 1
            if obj_value < self.best_so_far:
                self.best_so_far = obj_value
                # SAVE THE BEST CONFIGURATION HERE
                self.best_latC = self.latC.copy()
                self.best_lonC = self.lonC.copy()
                #print(f"Eval {self.eval_count}: New best = {obj_value:.6f}")
            #elif self.eval_count % 100 == 0:
                #print(f"Eval {self.eval_count}: Current = {obj_value:.6f}, Best = {self.best_so_far:.6f}")                
        
            return obj_value
        
        for restart in range(n_restarts):
            self.eval_count = 0
            self.best_so_far = 1e11
            self.best_latC = None
            self.best_lonC = None
            
            if restart > 0:
                self.compute_initial_guess()
                for i in range(self.CSA.N):
                    xi, eta = self.MG.grid.projection.geo2cube(self.lonC[i], self.latC[i])
                    xi += np.random.normal(0, self.AG_radius * 0.2)
                    eta += np.random.normal(0, self.AG_radius * 0.2)
                    self.lonC[i], self.latC[i] = self.MG.grid.projection.cube2geo(xi, eta)
            
            # Run SA
            result = dual_annealing(objective, bounds, 
                                  maxiter=100,
                                  no_local_search=True,
                                  initial_temp=5000,
                                  restart_temp_ratio=2e-5)
            
            # Use the saved best configuration for this restart
            if self.best_latC is not None:
                if self.best_so_far < best_global[2]:
                    best_global = (self.best_latC.copy(), self.best_lonC.copy(), self.best_so_far)
            
            print(f"Restart {restart}: {self.best_so_far:.1f} (best: {best_global[2]:.1f})")
        
        self.latC, self.lonC = best_global[0], best_global[1]


#%% Manually setting grids

# Import supermag stations
base = os.getcwd()
path_in = os.path.join(base, '..', 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))
lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()

# Defined fixed sensor array
FSA = FixedSensorArray(lat, lon, 5e-9)

# Define model grid
MG = ModelGrid(68.5, 23.5, L=900e3, W=900e3, Lres=20e3, Wres=20e3)

# Define analysis grid
AG = AnalysisGrid(MG.grid, 68.5, 23.5, 250e3)

MG.illustrate_grid(FSA=FSA, AG=AG, lim=1.2)

#%% Resolution test

f = MG.grid.ingrid(lon, lat)
lat, lon = lat[f], lon[f]

Res = Resolution(lat, lon, 6371.2e3, 5e-9, 1e0, MG, AG)

Res.compute_resolution_matrix()
Res.compute_resolution()

f = (Res.xiResFlag == 1) & (Res.etaResFlag == 1)
vmin=0
vmax = 400
clvls = np.linspace(vmin, vmax, 40)

fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,5))
cc = axs[0].tricontourf(MG.grid.xi[f], MG.grid.eta[f], Res.xiRes[f]*1e-3, cmap='Reds', levels=clvls)
axs[1].tricontourf(MG.grid.xi[f], MG.grid.eta[f], Res.etaRes[f]*1e-3, cmap='Reds', levels=clvls)
axs[2].tricontourf(MG.grid.xi[f], MG.grid.eta[f], (Res.xiRes[f]+Res.etaRes[f])/2*1e-3, cmap='Reds', levels=clvls)
axs[0].set_xlim(MG.grid.xi_mesh.min(), MG.grid.xi_mesh.max())
axs[0].set_ylim(MG.grid.eta_mesh.min(), MG.grid.eta_mesh.max())

xi, eta = MG.grid.projection.geo2cube(AG.center_lon+0.001, AG.center_lat)
sin = np.sin(np.linspace(0, 2*np.pi, 1000))
cos = np.cos(np.linspace(0, 2*np.pi, 1000))
radius = AG.radius / MG.res_xi * MG.grid.dxi
xi = xi + radius * sin
eta = eta + radius * cos
for ax in axs:
    ax.plot(xi, eta, '--', color='tab:blue')
    ax.set_aspect('equal')
cbar = fig.colorbar(cc, ax=axs.ravel().tolist(), shrink=0.9, orientation='vertical')


fig = plt.figure(figsize=(10,10))
plt.tricontourf(MG.grid.xi[f], MG.grid.eta[f], (Res.xiRes[f]+Res.etaRes[f])/2*1e-3, cmap='Reds', levels=clvls)
plt.xlim(MG.grid.xi_mesh.min(), MG.grid.xi_mesh.max())
plt.ylim(MG.grid.eta_mesh.min(), MG.grid.eta_mesh.max())

xi, eta = MG.grid.projection.geo2cube(AG.center_lon+0.001, AG.center_lat)
sin = np.sin(np.linspace(0, 2*np.pi, 1000))
cos = np.cos(np.linspace(0, 2*np.pi, 1000))
radius = AG.radius / MG.res_xi * MG.grid.dxi
xi = xi + radius * sin
eta = eta + radius * cos
plt.plot(xi, eta, '--', color='tab:blue')
plt.gca().set_aspect('equal')
plt.colorbar()

#%% Test greedy algorithm

MG = ModelGrid(68.5, 23.5, L=900e3, W=900e3, Lres=40e3, Wres=40e3)

# Import supermag stations
base = os.getcwd()
path_in = os.path.join(base, '..', 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))
lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()

f = MG.grid.ingrid(lon, lat)
lon = lon[f]
lat = lat[f]

# Defined fixed sensor array
FSA = FixedSensorArray(lat, lon, 5e-9)

AG = AnalysisGrid(MG.grid, 68.5, 23.5, 250e3)

MG.illustrate_grid(FSA=FSA, AG=AG, lim=1.2)

CSA = CandidateSensorArray(2, 15e-9)
#CSA = CandidateSensorArray(36, 5e-9)

SO = SensorOptimizer(MG, AG, FSA, CSA)

SO.compute_initial_guess()

SO.optimize_positions(max_iter=49)

SO.compute_resolution()

#%%

fig, axs = plt.subplots(6,6, figsize=(20, 20), sharex=True, sharey=True)
for i, (ax, res) in enumerate(zip(axs.flatten(), SO.res_all)):
    vmin=100e3
    vmax=300e3
    clvls = np.linspace(vmin, vmax, 40)
    res_ = np.copy(res)
    ax.contourf(MG.grid.xi, MG.grid.eta, res, cmap='Reds', levels=clvls)
    xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
    ax.plot(xi, eta, '*', markersize=5, color='tab:blue')
    xi, eta = MG.grid.projection.geo2cube(SO.lonC[:i], SO.latC[:i])
    ax.plot(xi, eta, '*', markersize=5, color='white')
    
    ax.set_axis_off()
    ax.set_xlim(-SO.AG_radius, SO.AG_radius)
    ax.set_ylim(-SO.AG_radius, SO.AG_radius)

#%%
vmin=150e3
vmax=300e3
clvls = np.linspace(vmin, vmax, 40)

fig = plt.figure(figsize=(15,15))
plt.tricontourf(MG.grid.xi.flatten(), MG.grid.eta.flatten(), SO.res.flatten(), cmap='Reds', levels=clvls)
xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
plt.plot(xi, eta, '*', markersize=5, color='tab:blue')
xi, eta = MG.grid.projection.geo2cube(SO.lonC, SO.latC)
plt.plot(xi, eta, '*', markersize=5, color='white')

plt.xlim(-SO.AG_radius, SO.AG_radius)
plt.ylim(-SO.AG_radius, SO.AG_radius)

#%%
vmin=150e3
vmax=300e3
clvls = np.linspace(vmin, vmax, 40)

fig, axs = plt.subplots(7,7, figsize=(20, 20), sharex=True, sharey=True)
for i, (ax, res) in enumerate(zip(axs.flatten(), SO.res_all_opt)):
    ax.tricontourf(MG.grid.xi.flatten(), MG.grid.eta.flatten(), res.flatten(), cmap='Reds', levels=clvls)
    xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
    ax.plot(xi, eta, '*', markersize=5, color='tab:blue')
    xi, eta = MG.grid.projection.geo2cube(SO.lonCs[i], SO.latCs[i])
    ax.plot(xi, eta, '*', markersize=5, color='white')

ax.set_xlim(-SO.AG_radius, SO.AG_radius)
ax.set_ylim(-SO.AG_radius, SO.AG_radius)

    
    
#%% Test SA algorithm

MG = ModelGrid(68.5, 23.5, L=900e3, W=900e3, Lres=40e3, Wres=40e3)

# Import supermag stations
base = os.getcwd()
path_in = os.path.join(base, '..', 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))
lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()

f = MG.grid.ingrid(lon, lat)
lon = lon[f]
lat = lat[f]

# Defined fixed sensor array
FSA = FixedSensorArray(lat, lon, 5e-9)

AG = AnalysisGrid(MG.grid, 68.5, 23.5, 250e3)

MG.illustrate_grid(FSA=FSA, AG=AG, lim=1.2)

CSA = CandidateSensorArray(36, 15e-9)
#CSA = CandidateSensorArray(36, 5e-9)

SO = SensorOptimizer(MG, AG, FSA, CSA)

#SO.compute_initial_guess()

#SO.optimize_positions2()
SO.optimize_positions2_fix()

SO.compute_resolution()

#%%

fig, axs = plt.subplots(6,6, figsize=(20, 20), sharex=True, sharey=True)
for i, (ax, res) in enumerate(zip(axs.flatten(), SO.res_all)):
    vmin=100e3
    vmax=300e3
    clvls = np.linspace(vmin, vmax, 40)
    res_ = np.copy(res)
    ax.contourf(MG.grid.xi, MG.grid.eta, res, cmap='Reds', levels=clvls)
    xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
    ax.plot(xi, eta, '*', markersize=5, color='tab:blue')
    xi, eta = MG.grid.projection.geo2cube(SO.lonC[:i], SO.latC[:i])
    ax.plot(xi, eta, '*', markersize=5, color='white')
    
    ax.set_axis_off()
    ax.set_xlim(-SO.AG_radius, SO.AG_radius)
    ax.set_ylim(-SO.AG_radius, SO.AG_radius)

#%%
vmin=150e3
vmax=300e3
clvls = np.linspace(vmin, vmax, 40)

fig = plt.figure(figsize=(15,15))
plt.tricontourf(MG.grid.xi.flatten(), MG.grid.eta.flatten(), SO.res.flatten(), cmap='Reds', levels=clvls)
xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
plt.plot(xi, eta, '*', markersize=5, color='tab:blue')
xi, eta = MG.grid.projection.geo2cube(SO.best_lonC, SO.best_latC)
plt.plot(xi, eta, '*', markersize=5, color='white')

plt.xlim(-SO.AG_radius, SO.AG_radius)
plt.ylim(-SO.AG_radius, SO.AG_radius)


#%%

MG = ModelGrid(68.5, 23.5, L=900e3, W=900e3, Lres=40e3, Wres=40e3)

# Import supermag stations
base = os.getcwd()
path_in = os.path.join(base, '..', 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))
lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()

f = MG.grid.ingrid(lon, lat)
lon = lon[f]
lat = lat[f]

# Defined fixed sensor array
FSA = FixedSensorArray(lat, lon, 5e-9)

AG = AnalysisGrid(MG.grid, 68.5, 23.5, 250e3)

MG.illustrate_grid(FSA=FSA, AG=AG, lim=1.2)

output = []

for sensors in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print(sensors)
    CSA = CandidateSensorArray(sensors, 15e-9)
    SO = SensorOptimizer(MG, AG, FSA, CSA)
    SO.optimize_positions2_fix()
    SO.compute_resolution()
    output.append({'lat': SO.best_latC, 'lon': SO.best_lonC,
                   'maxres': SO.best_so_far, 'R': SO.res})


#%%

plt.figure()
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80], [out['maxres'] for out in output])

i = 11
vmin=150e3
vmax=300e3
clvls = np.linspace(vmin, vmax, 40)

fig = plt.figure(figsize=(15,15))
plt.tricontourf(MG.grid.xi.flatten(), MG.grid.eta.flatten(), output[i]['R'].flatten(), cmap='Reds', levels=clvls)
xi, eta = MG.grid.projection.geo2cube(FSA.lon, FSA.lat)
plt.plot(xi, eta, '*', markersize=5, color='white')
xi, eta = MG.grid.projection.geo2cube(output[i]['lon'], output[i]['lat'])
plt.plot(xi, eta, '*', markersize=5, color='tab:blue')

#plt.xlim(-SO.AG_radius, SO.AG_radius)
#plt.ylim(-SO.AG_radius, SO.AG_radius)

