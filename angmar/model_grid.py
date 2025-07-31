#%% Import

import numpy as np
from secsy.cubedsphere import CSgrid, CSprojection
import matplotlib.pyplot as plt

#%%

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
            radius = AG.radius / self.res_xi * self.grid.dxi
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