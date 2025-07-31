#%% Import

#%% Analysis grid class

class AnalysisGrid:
    def __init__(self, grid, center_lat=None, center_lon=None, radius=None):
        self.grid = grid
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius # Same units a grid.L, grid.W, grid.Lres etc.
        #self.radius = self._radius / grid.Lres * np.median(np.diff(grid.eta_mesh, axis=0)) # Radius in CS units  