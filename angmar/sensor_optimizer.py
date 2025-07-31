#%% Import

import numpy as np
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution

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
        self.res = (self.Res.xiRes + self.Res.etaRes)/2 * 1e-3 # convert from m to km
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
        
    def optimize_positions_SA(self, init=None, maxiter=100):
        """Simulated annealing"""
        
        def objective(x):
            # Unpack positions
            for i in range(self.CSA.N):
                xi = x[2*i]
                eta = x[2*i + 1]
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.lonC[i] = lon
                self.latC[i] = lat
            
            obj_value = self.compute_max_resolution()
        
            self.eval_count += 1
            if obj_value < self.best_obj:
                self.best_obj = obj_value
                self.best_res = self.res.copy()
                # SAVE THE BEST CONFIGURATION HERE
                self.best_latC = self.latC.copy()
                self.best_lonC = self.lonC.copy()
            
            if self.eval_count % 100 == 0 or self.eval_count == 1:
                print(f"Eval {self.eval_count}: Best = {self.best_obj:.6f}")
        
            self.latCs.append(self.latC.copy())
            self.lonCs.append(self.lonC.copy())
            self.objs.append(np.copy(obj_value))
        
            return obj_value
        
        # get initial guess
        if init is None:
            print('No init provided, using sequential greedy algorithm')
            self.compute_initial_guess()
        else:
            print('Init provided')
            self.latC = init['lat']
            self.lonC = init['lon']
            self.rC = init['r']
            self.qC = init['q']
            self.compute_resolution()
            self.res_all = np.copy(self.res)
        
        print('Setting boundaries')
        bounds = []
        for i in range(self.CSA.N):
            bounds.extend([(-self.AG_radius, self.AG_radius),  # xi
                           (-self.AG_radius, self.AG_radius)]) # eta
        
        self.eval_count = 0
        self.best_obj = 1e11
        self.best_latC = None
        self.best_lonC = None
        self.best_res = None
        
        self.latCs = []
        self.lonCs = []
        self.objs = []
                
        result = dual_annealing(objective, bounds, maxiter=maxiter, no_local_search=True)      
        #result = dual_annealing(objective, bounds, maxiter=maxiter, no_local_search=False)
        
        self.result = result

    def optimize_positions_DE(self, init=None, maxiter=100):
        """Differential evolution."""
        
        def objective(x):
            # Unpack positions
            for i in range(self.CSA.N):
                xi = x[2*i]
                eta = x[2*i + 1]
                lon, lat = self.MG.grid.projection.cube2geo(xi, eta)
                self.lonC[i] = lon
                self.latC[i] = lat
            
            obj_value = self.compute_max_resolution()
        
            self.eval_count += 1
            if obj_value < self.best_obj:
                self.best_obj = obj_value
                self.best_res = self.res.copy()
                # SAVE THE BEST CONFIGURATION HERE
                self.best_latC = self.latC.copy()
                self.best_lonC = self.lonC.copy()
            
            if self.eval_count % 100 == 0 or self.eval_count == 1:
                print(f"Eval {self.eval_count}: Best = {self.best_obj:.6f}")
        
            self.latCs.append(self.latC.copy())
            self.lonCs.append(self.lonC.copy())
            self.objs.append(np.copy(obj_value))
        
            return obj_value
        
        # get initial guess
        if init is None:
            print('No init provided, using sequential greedy algorithm')
            self.compute_initial_guess()
        else:
            print('Init provided')
            self.latC = init['lat']
            self.lonC = init['lon']
            self.rC = init['r']
            self.qC = init['q']
            self.compute_resolution()
            self.res_all = np.copy(self.res)
        
        print('Setting boundaries')
        bounds = []
        for i in range(self.CSA.N):
            bounds.extend([(-self.AG_radius, self.AG_radius),  # xi
                           (-self.AG_radius, self.AG_radius)]) # eta
        
        self.eval_count = 0
        self.best_obj = 1e11
        self.best_latC = None
        self.best_lonC = None
        self.best_res = None
        
        self.latCs = []
        self.lonCs = []
        self.objs = []
        
        # DE actually returns the best found, not the last position!
        result = differential_evolution(objective, bounds,
                                        popsize=15,  # Population size
                                        maxiter=maxiter,
                                        disp=True,
                                        polish=False,  # No local search
                                        workers=1)    # Parallel evaluation
        
        self.result = result
