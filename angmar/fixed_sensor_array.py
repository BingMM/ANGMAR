#%% Import

import numpy as np

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