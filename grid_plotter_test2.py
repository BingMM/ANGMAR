import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from secsy.cubedsphere import CSgrid, CSprojection

@dataclass
class OSMFeature:
    """Container for OSM feature data"""
    feature_type: str  # 'coastline', 'border', 'road'
    coordinates: List[Tuple[float, float]]  # List of (lon, lat) tuples
    properties: Dict = None  # Optional properties


class OSMDataFetcher:
    """Fetches and stores geographic data from OpenStreetMap using Overpass API"""
    
    def __init__(self, grid, chunk_size=0.5):
        """
        Initialize the OSM data fetcher
        
        Args:
            grid: Grid object containing lat/lon arrays
            chunk_size: Size of each chunk in degrees (default 0.5)
        """
        self.grid = grid
        self.chunk_size = chunk_size
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Extract bounds from grid
        self.min_lat = np.min(grid.lat)
        self.max_lat = np.max(grid.lat)
        self.min_lon = np.min(grid.lon)
        self.max_lon = np.max(grid.lon)
        
        # Storage for fetched data
        self.coastlines = []
        self.borders = []
        self.roads = []
        
        # Metadata
        self.fetch_status = {
            'coastlines': False,
            'borders': False,
            'roads': False,
            'failed_chunks': []
        }
    
    def create_chunks(self):
        """Divide the area into smaller chunks"""
        chunks = []
        
        lat_chunks = int(np.ceil((self.max_lat - self.min_lat) / self.chunk_size))
        lon_chunks = int(np.ceil((self.max_lon - self.min_lon) / self.chunk_size))
        
        for i in range(lat_chunks):
            for j in range(lon_chunks):
                chunk_min_lat = self.min_lat + i * self.chunk_size
                chunk_max_lat = min(self.min_lat + (i + 1) * self.chunk_size, self.max_lat)
                chunk_min_lon = self.min_lon + j * self.chunk_size
                chunk_max_lon = min(self.min_lon + (j + 1) * self.chunk_size, self.max_lon)
                
                chunks.append({
                    'min_lat': chunk_min_lat,
                    'max_lat': chunk_max_lat,
                    'min_lon': chunk_min_lon,
                    'max_lon': chunk_max_lon
                })
        
        return chunks
    
    def fetch_all(self, coastlines=True, borders=True, roads=True, 
                  admin_level=2, road_types=None):
        """
        Fetch all requested data types and store them
        
        Args:
            coastlines: Whether to fetch coastlines
            borders: Whether to fetch political borders
            roads: Whether to fetch roads
            admin_level: Administrative level for borders (2=country, 4=state/province)
            road_types: List of road types to fetch (default: major roads)
        
        Returns:
            bool: True if all requested data was fetched successfully
        """
        if road_types is None:
            road_types = ['motorway', 'trunk', 'primary', 'secondary']
        
        chunks = self.create_chunks()
        print(f"Area divided into {len(chunks)} chunks")
        
        # Clear previous data
        self.coastlines.clear()
        self.borders.clear()
        self.roads.clear()
        self.fetch_status['failed_chunks'].clear()
        
        # Fetch data for each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_success = True
            
            if coastlines:
                features = self._fetch_coastlines_chunk(**chunk)
                if features is not None:
                    self.coastlines.extend(features)
                else:
                    chunk_success = False
            
            if borders:
                features = self._fetch_borders_chunk(**chunk, admin_level=admin_level)
                if features is not None:
                    self.borders.extend(features)
                else:
                    chunk_success = False
            
            if roads:
                features = self._fetch_roads_chunk(**chunk, road_types=road_types)
                if features is not None:
                    self.roads.extend(features)
                else:
                    chunk_success = False
            
            if not chunk_success:
                self.fetch_status['failed_chunks'].append(i)
                print(f"  Warning: Some data failed to load for chunk {i+1}")
            
            # Adaptive delay
            time.sleep(0.5 if chunk_success else 2.0)
        
        # Update fetch status
        self.fetch_status['coastlines'] = coastlines and len(self.coastlines) > 0
        self.fetch_status['borders'] = borders and len(self.borders) > 0
        self.fetch_status['roads'] = roads and len(self.roads) > 0
        
        # Report results
        print(f"\nFetch complete:")
        print(f"  Coastlines: {len(self.coastlines)} features")
        print(f"  Borders: {len(self.borders)} features")
        print(f"  Roads: {len(self.roads)} features")
        
        if self.fetch_status['failed_chunks']:
            print(f"  Warning: {len(self.fetch_status['failed_chunks'])} chunks had failures")
        
        return len(self.fetch_status['failed_chunks']) == 0
    
    def _fetch_coastlines_chunk(self, min_lat, max_lat, min_lon, max_lon):
        """Fetch coastline data for a chunk"""
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        data = self._execute_query(query)
        return self._parse_osm_data(data, 'coastline') if data else None
    
    def _fetch_borders_chunk(self, min_lat, max_lat, min_lon, max_lon, admin_level=2):
        """Fetch political borders for a chunk"""
        query = f"""
        [out:json][timeout:25];
        (
          way["boundary"="administrative"]["admin_level"="{admin_level}"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["boundary"="administrative"]["admin_level"="{admin_level}"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        data = self._execute_query(query)
        return self._parse_osm_data(data, 'border') if data else None
    
    def _fetch_roads_chunk(self, min_lat, max_lat, min_lon, max_lon, road_types):
        """Fetch road data for a chunk"""
        road_filters = ''.join([f'way["highway"="{rt}"]({min_lat},{min_lon},{max_lat},{max_lon});' 
                               for rt in road_types])
        
        query = f"""
        [out:json][timeout:25];
        (
          {road_filters}
        );
        out geom;
        """
        data = self._execute_query(query)
        return self._parse_osm_data(data, 'road') if data else None
    
    def _execute_query(self, query):
        """Execute an Overpass API query with retry logic"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.overpass_url, data=query, timeout=45)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too many requests
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                elif response.status_code == 504:  # Gateway timeout
                    wait_time = retry_delay * (attempt + 1) * 2
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
    
    def _parse_osm_data(self, data, feature_type):
        """Parse OSM data into OSMFeature objects"""
        if not data or 'elements' not in data:
            return []
        
        features = []
        
        for element in data['elements']:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                feature = OSMFeature(
                    feature_type=feature_type,
                    coordinates=coords,
                    properties=element.get('tags', {})
                )
                features.append(feature)
            elif element['type'] == 'relation' and 'members' in element:
                for member in element['members']:
                    if member['type'] == 'way' and 'geometry' in member:
                        coords = [(node['lon'], node['lat']) for node in member['geometry']]
                        feature = OSMFeature(
                            feature_type=feature_type,
                            coordinates=coords,
                            properties=element.get('tags', {})
                        )
                        features.append(feature)
        
        return features
    
def plot_osm_data(grid, fetcher, ax=None, coastline_style=None, border_style=None, 
                  road_style=None, show_grid_points=True):
    """
    Plot OSM data using custom projection
    
    Args:
        grid: Grid object with projection.geo2cube method
        fetcher: OSMDataFetcher object containing the data
        ax: Matplotlib axes (creates new if None)
        coastline_style: Dict of matplotlib plot kwargs for coastlines
        border_style: Dict of matplotlib plot kwargs for borders
        road_style: Dict of matplotlib plot kwargs for roads
        show_grid_points: Whether to show grid points
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Default styles
    if coastline_style is None:
        coastline_style = {'color': 'blue', 'linewidth': 1.5, 'label': 'Coastlines'}
    if border_style is None:
        border_style = {'color': 'red', 'linewidth': 2, 'linestyle': '--', 'label': 'Borders'}
    if road_style is None:
        road_style = {'color': 'gray', 'linewidth': 0.5, 'alpha': 0.7, 'label': 'Roads'}
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Helper function to project and plot features
    def plot_features(features, style):
        for feature in features:
            if len(feature.coordinates) < 2:
                continue
            
            lons, lats = zip(*feature.coordinates)
            xi, eta = grid.projection.geo2cube(np.array(lons), np.array(lats))
            
            # Remove label after first plot to avoid duplicate labels
            plot_style = style.copy()
            if 'label' in plot_style:
                ax.plot(xi, eta, **plot_style)
                plot_style.pop('label')
            else:
                ax.plot(xi, eta, **plot_style)
    
    # Plot features
    if fetcher.coastlines:
        print(f"Plotting {len(fetcher.coastlines)} coastline features...")
        plot_features(fetcher.coastlines, coastline_style)
    
    if fetcher.borders:
        print(f"Plotting {len(fetcher.borders)} border features...")
        plot_features(fetcher.borders, border_style)
    
    if fetcher.roads:
        print(f"Plotting {len(fetcher.roads)} road features...")
        plot_features(fetcher.roads, road_style)
    
    # Add grid points
    if show_grid_points:
        grid_xi, grid_eta = grid.projection.geo2cube(grid.lon.flatten(), grid.lat.flatten())
        ax.scatter(grid_xi[::100], grid_eta[::100], s=1, alpha=0.3, 
                  color='black', label='Grid points')
    
    ax.set_xlabel('Xi')
    ax.set_ylabel('Eta')
    ax.set_title('OpenStreetMap Data with Custom Projection')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


# Example usage:
# Example of how to use with your grid:
position = (25.5, 68.5)    
orientation = 0
L, W = 500e3, 500e3
Lres, Wres = 10e3, 10e3
R = 6371.2e3 + 110e3

grid = CSgrid(CSprojection(position, orientation), L, W, Lres, Lres, R = R)    
    

# Initialize fetcher with your grid
fetcher = OSMDataFetcher(grid, chunk_size=2)

# Fetch all data (this takes time but only needs to be done once)
fetcher.fetch_all(
    coastlines=True,
    borders=False,
    roads=True,
    admin_level=2,  # Country borders
    road_types=['motorway', 'trunk', 'primary']
)

# Save data for later use
#fetcher.save_data('osm_data.json')

# Later, you can load the data instead of fetching
# fetcher.load_data('osm_data.json')

# Plot the data (can be done multiple times with different styles)
fig, ax = plot_osm_data(
    grid, 
    fetcher,
    coastline_style={'color': 'darkblue', 'linewidth': 2},
    border_style={'color': 'darkred', 'linewidth': 2.5, 'linestyle': '-'},
    road_style={'color': 'orange', 'linewidth': 1, 'alpha': 0.8}
)

# You can also plot on existing axes
fig2, ax2 = plt.subplots(figsize=(10, 8))
plot_osm_data(grid, fetcher, ax=ax2, show_grid_points=False)

plt.show()
