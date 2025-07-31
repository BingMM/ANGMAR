import matplotlib.pyplot as plt
import numpy as np
import requests
import time
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import transform
import json
from secsy.cubedsphere import CSgrid, CSprojection

class OSMDataFetcher:
    """Fetches geographic data from OpenStreetMap using Overpass API"""
    
    def __init__(self, chunk_size=0.5):
        """
        Initialize the OSM data fetcher
        
        Args:
            chunk_size: Size of each chunk in degrees (default 0.5)
        """
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.chunk_size = chunk_size
        
    def get_bounds_from_grid(self, grid):
        """Extract bounds from grid lat/lon arrays"""
        min_lat = np.min(grid.lat)
        max_lat = np.max(grid.lat)
        min_lon = np.min(grid.lon)
        max_lon = np.max(grid.lon)
        return min_lat, max_lat, min_lon, max_lon
    
    def create_chunks(self, min_lat, max_lat, min_lon, max_lon):
        """Divide the area into smaller chunks"""
        chunks = []
        
        lat_chunks = int(np.ceil((max_lat - min_lat) / self.chunk_size))
        lon_chunks = int(np.ceil((max_lon - min_lon) / self.chunk_size))
        
        for i in range(lat_chunks):
            for j in range(lon_chunks):
                chunk_min_lat = min_lat + i * self.chunk_size
                chunk_max_lat = min(min_lat + (i + 1) * self.chunk_size, max_lat)
                chunk_min_lon = min_lon + j * self.chunk_size
                chunk_max_lon = min(min_lon + (j + 1) * self.chunk_size, max_lon)
                
                chunks.append({
                    'min_lat': chunk_min_lat,
                    'max_lat': chunk_max_lat,
                    'min_lon': chunk_min_lon,
                    'max_lon': chunk_max_lon
                })
        
        return chunks
    
    def fetch_coastlines(self, min_lat, max_lat, min_lon, max_lon):
        """Fetch coastline data from OSM"""
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        return self._execute_query(query)
    
    def fetch_borders(self, min_lat, max_lat, min_lon, max_lon, admin_level=2):
        """Fetch political borders from OSM
        
        Args:
            admin_level: Administrative level (2=country, 4=state/province, etc.)
        """
        query = f"""
        [out:json][timeout:25];
        (
          way["boundary"="administrative"]["admin_level"="{admin_level}"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["boundary"="administrative"]["admin_level"="{admin_level}"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        return self._execute_query(query)
    
    def fetch_roads(self, min_lat, max_lat, min_lon, max_lon, road_types=None):
        """Fetch road data from OSM
        
        Args:
            road_types: List of road types to fetch (motorway, trunk, primary, etc.)
                       If None, fetches major roads
        """
        if road_types is None:
            road_types = ['motorway', 'trunk', 'primary', 'secondary']
        
        road_filters = ''.join([f'way["highway"="{rt}"]({min_lat},{min_lon},{max_lat},{max_lon});' 
                               for rt in road_types])
        
        query = f"""
        [out:json][timeout:25];
        (
          {road_filters}
        );
        out geom;
        """
        return self._execute_query(query)
    
    def _execute_query(self, query):
        """Execute an Overpass API query with retry logic"""
        max_retries = 5  # Increased from 3
        retry_delay = 5  # Increased from 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.overpass_url, data=query, timeout=45)  # Increased timeout
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too many requests
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                elif response.status_code == 504:  # Gateway timeout
                    wait_time = retry_delay * (attempt + 1) * 2  # Longer wait for timeout errors
                    print(f"Server timeout (504), waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Error {response.status_code} on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        print(f"Failed to fetch data after {max_retries} attempts")
        return None
    
    def parse_osm_data(self, data):
        """Parse OSM data into coordinate lists"""
        if not data or 'elements' not in data:
            return []
        
        features = []
        
        for element in data['elements']:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                features.append(coords)
            elif element['type'] == 'relation' and 'members' in element:
                # Handle relations (more complex geometries)
                for member in element['members']:
                    if member['type'] == 'way' and 'geometry' in member:
                        coords = [(node['lon'], node['lat']) for node in member['geometry']]
                        features.append(coords)
        
        return features


class OSMPlotter:
    """Plot OSM data using custom projection"""
    
    def __init__(self, grid):
        """Initialize plotter with grid object containing projection"""
        self.grid = grid
        self.projection = grid.projection
        
    def project_coordinates(self, coords):
        """Project a list of (lon, lat) coordinates to (xi, eta)"""
        if not coords:
            return [], []
        
        lons, lats = zip(*coords)
        xi, eta = self.projection.geo2cube(np.array(lons), np.array(lats))
        return xi, eta
    
    def plot_features(self, features, ax=None, **kwargs):
        """Plot a list of features (coordinate lists)"""
        if ax is None:
            ax = plt.gca()
        
        for feature in features:
            if len(feature) < 2:
                continue
            
            xi, eta = self.project_coordinates(feature)
            ax.plot(xi, eta, **kwargs)
    
    def plot_osm_data(self, grid, coastlines=True, borders=True, roads=True, 
                      admin_level=2, road_types=None, chunk_size=0.5):
        """Main function to fetch and plot OSM data"""
        
        fetcher = OSMDataFetcher(chunk_size=chunk_size)
        min_lat, max_lat, min_lon, max_lon = fetcher.get_bounds_from_grid(grid)
        chunks = fetcher.create_chunks(min_lat, max_lat, min_lon, max_lon)
        
        print(f"Area divided into {len(chunks)} chunks")
        
        # Storage for all features
        all_coastlines = []
        all_borders = []
        all_roads = []
        
        # Track failed chunks for potential retry
        failed_chunks = []
        
        # Fetch data for each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            chunk_success = True
            
            if coastlines:
                coast_data = fetcher.fetch_coastlines(**chunk)
                if coast_data:
                    all_coastlines.extend(fetcher.parse_osm_data(coast_data))
                else:
                    chunk_success = False
            
            if borders:
                border_data = fetcher.fetch_borders(**chunk, admin_level=admin_level)
                if border_data:
                    all_borders.extend(fetcher.parse_osm_data(border_data))
                else:
                    chunk_success = False
            
            if roads:
                road_data = fetcher.fetch_roads(**chunk, road_types=road_types)
                if road_data:
                    all_roads.extend(fetcher.parse_osm_data(road_data))
                else:
                    chunk_success = False
            
            if not chunk_success:
                failed_chunks.append((i, chunk))
                print(f"  Warning: Some data failed to load for chunk {i+1}")
            
            # Adaptive delay - longer if we had failures
            if chunk_success:
                time.sleep(0.5)
            else:
                time.sleep(2.0)  # Longer delay after failures
        
        # Report on any failed chunks
        if failed_chunks:
            print(f"\nWarning: {len(failed_chunks)} chunks had failures and may be missing data")
            print("Consider re-running with a smaller chunk_size or during off-peak hours")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot features with different styles
        if all_coastlines:
            print(f"Plotting {len(all_coastlines)} coastline features...")
            self.plot_features(all_coastlines, ax, color='blue', linewidth=1.5, 
                             label='Coastlines')
        
        if all_borders:
            print(f"Plotting {len(all_borders)} border features...")
            self.plot_features(all_borders, ax, color='red', linewidth=2, 
                             linestyle='--', label='Borders')
        
        if all_roads:
            print(f"Plotting {len(all_roads)} road features...")
            self.plot_features(all_roads, ax, color='gray', linewidth=0.5, 
                             alpha=0.7, label='Roads')
        
        # Add grid outline
        grid_xi, grid_eta = self.projection.geo2cube(grid.lon.flatten(), 
                                                     grid.lat.flatten())
        ax.scatter(grid_xi[::100], grid_eta[::100], s=1, alpha=0.3, 
                  color='black', label='Grid points')
        
        ax.set_xlabel('Xi')
        ax.set_ylabel('Eta')
        ax.set_title('OpenStreetMap Data with Custom Projection')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return fig, ax


# Example usage function
def plot_osm_on_grid(grid, **kwargs):
    """
    Convenience function to plot OSM data on your grid
    
    Args:
        grid: Your grid object with .lat, .lon arrays and .projection.geo2cube method
        **kwargs: Additional arguments passed to plot_osm_data
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    plotter = OSMPlotter(grid)
    return plotter.plot_osm_data(grid, **kwargs)


# Example of how to use with your grid:
position = (25.5, 68.5)    
orientation = 0
L, W = 500e3, 500e3
Lres, Wres = 10e3, 10e3
R = 6371.2e3 + 110e3

grid = CSgrid(CSprojection(position, orientation), L, W, Lres, Lres, R = R)    

# Assuming you have your grid object ready:
fig, ax = plot_osm_on_grid(
    grid,
    coastlines=True,
    borders=True,
    roads=True,
    admin_level=2,  # Country borders
    road_types=['motorway', 'trunk', 'primary'],  # Major roads only
    chunk_size=2  # Chunk size in degrees
)

# You can also plot specific features separately:
plotter = OSMPlotter(grid)
fetcher = OSMDataFetcher()

# Get bounds from your grid
min_lat, max_lat, min_lon, max_lon = fetcher.get_bounds_from_grid(grid)

# Fetch and plot only coastlines
coast_data = fetcher.fetch_coastlines(min_lat, max_lat, min_lon, max_lon)
coastlines = fetcher.parse_osm_data(coast_data)
plotter.plot_features(coastlines, color='blue', linewidth=2)

plt.show()
