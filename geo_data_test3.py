import os
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import pandas as pd

def create_hexagonal_grid(center_x, center_y, spacing_km, n_rings, transformer):
    """
    Create a hexagonal grid of magnetometer positions
    
    Parameters:
    -----------
    center_x, center_y : float
        Center coordinates in projected system (meters)
    spacing_km : float
        Spacing between magnetometers in kilometers
    n_rings : int
        Number of rings around center (radius = n_rings * spacing_km)
    transformer : pyproj.Transformer
        Coordinate transformer for converting back to lat/lon
    
    Returns:
    --------
    proposed_sites : list
        List of tuples (x, y, lon, lat) for each proposed site
    """
    spacing_m = spacing_km * 1000  # Convert to meters
    proposed_sites = []
    
    for ring in range(n_rings + 1):
        if ring == 0:
            # Center point
            lon, lat = transformer.transform(center_x, center_y, direction='INVERSE')
            proposed_sites.append((center_x, center_y, lon, lat))
        else:
            # Points in hexagonal rings - 6 * ring points per ring
            n_points = 6 * ring
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = center_x + ring * spacing_m * np.cos(angle)
                y = center_y + ring * spacing_m * np.sin(angle)
                
                lon, lat = transformer.transform(x, y, direction='INVERSE')
                proposed_sites.append((x, y, lon, lat))
    
    return proposed_sites

def fetch_osm_data(center_lat, center_lon, radius_km):
    """
    Fetch road and coastline data from OpenStreetMap using Overpass API
    
    Parameters:
    -----------
    center_lat, center_lon : float
        Center coordinates
    radius_km : float
        Radius in kilometers
    
    Returns:
    --------
    roads_gdf : GeoDataFrame with road geometries
    coastlines_gdf : GeoDataFrame with coastline geometries
    """
    
    # For large areas, we need to be more selective or chunk the download
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    all_roads = []
    all_coastlines = []
    
    if radius_km > 200:
        # For large areas, download in chunks
        print(f"Large area detected ({radius_km} km radius). Downloading in chunks...")
        
        # Create a grid of smaller download areas
        chunk_radius = 150  # km
        n_chunks = int(radius_km / chunk_radius) + 1
        
        for i in range(-n_chunks, n_chunks + 1):
            for j in range(-n_chunks, n_chunks + 1):
                # Calculate chunk center
                # Approximate: 1 degree latitude = 111 km
                # At 68°N, 1 degree longitude ≈ 41 km
                chunk_lat = center_lat + (i * chunk_radius / 111)
                chunk_lon = center_lon + (j * chunk_radius / 41)
                
                # Check if chunk center is within overall radius
                dist_km = np.sqrt((i * chunk_radius)**2 + (j * chunk_radius)**2)
                if dist_km > radius_km:
                    continue
                
                # Query for this chunk - only major roads to reduce data
                query = f"""
                [out:json][timeout:25];
                (
                  way["highway"~"motorway|trunk|primary|secondary"]
                  (around:{chunk_radius * 1000},{chunk_lat},{chunk_lon});
                  way["natural"="coastline"]
                  (around:{chunk_radius * 1000},{chunk_lat},{chunk_lon});
                );
                out geom;
                """
                
                try:
                    response = requests.get(overpass_url, params={'data': query}, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        
                        for element in data.get('elements', []):
                            if element['type'] == 'way' and 'geometry' in element:
                                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                                tags = element.get('tags', {})
                                
                                if 'highway' in tags and len(coords) >= 2:
                                    all_roads.append({
                                        'geometry': LineString(coords),
                                        'highway': tags.get('highway', 'unclassified'),
                                        'name': tags.get('name', ''),
                                        'ref': tags.get('ref', '')
                                    })
                                elif tags.get('natural') == 'coastline' and len(coords) >= 2:
                                    all_coastlines.append({
                                        'geometry': LineString(coords),
                                        'name': tags.get('name', '')
                                    })
                    
                except Exception as e:
                    print(f"Error downloading chunk at ({chunk_lat:.2f}, {chunk_lon:.2f}): {e}")
                    continue
        
        print(f"Downloaded {len(all_roads)} road segments and {len(all_coastlines)} coastline segments")
    
    else:
        # For smaller areas, single query
        print(f"Fetching roads and coastlines within {radius_km} km of ({center_lat}, {center_lon})...")
        
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary"]
          (around:{radius_km * 1000},{center_lat},{center_lon});
          way["natural"="coastline"]
          (around:{radius_km * 1000},{center_lat},{center_lon});
        );
        out geom;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            data = response.json()
            
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    tags = element.get('tags', {})
                    
                    if 'highway' in tags and len(coords) >= 2:
                        all_roads.append({
                            'geometry': LineString(coords),
                            'highway': tags.get('highway', 'unclassified'),
                            'name': tags.get('name', ''),
                            'ref': tags.get('ref', '')
                        })
                    elif tags.get('natural') == 'coastline' and len(coords) >= 2:
                        all_coastlines.append({
                            'geometry': LineString(coords),
                            'name': tags.get('name', '')
                        })
        
        except Exception as e:
            print(f"Error downloading data: {e}")
        
        print(f"Downloaded {len(all_roads)} road segments and {len(all_coastlines)} coastline segments")
    
    # Create GeoDataFrames
    roads_gdf = gpd.GeoDataFrame(all_roads, crs='EPSG:4326') if all_roads else gpd.GeoDataFrame(columns=['geometry', 'highway', 'name', 'ref'])
    coastlines_gdf = gpd.GeoDataFrame(all_coastlines, crs='EPSG:4326') if all_coastlines else gpd.GeoDataFrame(columns=['geometry', 'name'])
    
    # Remove duplicates based on geometry
    if not roads_gdf.empty:
        roads_gdf = roads_gdf.drop_duplicates(subset=['geometry'])
        roads_gdf = roads_gdf.to_crs('EPSG:3995')
    if not coastlines_gdf.empty:
        coastlines_gdf = coastlines_gdf.drop_duplicates(subset=['geometry'])
        coastlines_gdf = coastlines_gdf.to_crs('EPSG:3995')
    
    return roads_gdf, coastlines_gdf

def plot_magnetometer_network_osm(magnetometer_coords, center_lat, center_lon, 
                                  spacing_km=100, n_rings=3,
                                  grid_center_lat=None, grid_center_lon=None):
    """
    Plot magnetometer network with OSM roads and coastlines fetched on-the-fly
    
    Parameters:
    -----------
    magnetometer_coords : list of tuples
        [(lon1, lat1, name1), (lon2, lat2, name2), ...]
    center_lat, center_lon : float
        EISCAT 3D location
    spacing_km : float
        Spacing between magnetometers in kilometers
    n_rings : int
        Number of hexagonal rings (radius = n_rings * spacing_km)
    grid_center_lat, grid_center_lon : float, optional
        Center for the magnetometer grid. If None, uses EISCAT 3D location
    """
    
    # Calculate effective radius for data fetching and plotting
    radius_km = n_rings * spacing_km
    
    # Use grid center if provided, otherwise use EISCAT location
    if grid_center_lat is None:
        grid_center_lat = center_lat
    if grid_center_lon is None:
        grid_center_lon = center_lon
    
    # Fetch road and coastline data - use grid center for fetching data
    roads_gdf, coastlines_gdf = fetch_osm_data(grid_center_lat, grid_center_lon, radius_km)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Transform coordinates to projected system
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995", always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    grid_center_x, grid_center_y = transformer.transform(grid_center_lon, grid_center_lat)
    
    # Set ocean background first
    ax.set_facecolor('#a8cdf0')  # Light blue for ocean
    
    # Create land area polygon if we have coastlines
    if not coastlines_gdf.empty:
        # Plot coastlines
        coastlines_gdf.plot(ax=ax, color='#4a5568', linewidth=2, zorder=3)
    
    # Add simplified land background as a large area
    # This is a simplified approach - Norway/Sweden/Finland are mostly land in this region
    # For more accuracy, you'd need actual land polygons
    land_rect = plt.Rectangle((grid_center_x - radius_km*1500, grid_center_y - radius_km*1500), 
                             radius_km*3000, radius_km*3000,
                             facecolor='#f0f0f0', edgecolor='none', zorder=0)
    ax.add_patch(land_rect)
    
    # Plot roads with different styles
    road_styles = {
        'motorway': {'color': '#e990a0', 'linewidth': 3, 'zorder': 4},
        'trunk': {'color': '#fbb29a', 'linewidth': 2.5, 'zorder': 3},
        'primary': {'color': '#fdd7a1', 'linewidth': 2, 'zorder': 2},
        'secondary': {'color': '#f7fabf', 'linewidth': 1.5, 'zorder': 1},
        'tertiary': {'color': '#ffffff', 'linewidth': 1, 'zorder': 0},
        'unclassified': {'color': '#dddddd', 'linewidth': 0.5, 'zorder': 0},
        'residential': {'color': '#dddddd', 'linewidth': 0.5, 'zorder': 0}
    }
    
    # Plot roads by type
    if not roads_gdf.empty:
        for road_type, style in road_styles.items():
            roads_subset = roads_gdf[roads_gdf['highway'] == road_type]
            if len(roads_subset) > 0:
                #roads_subset.plot(ax=ax, **style, alpha=0.8)
                roads_subset.plot(ax=ax, **style, alpha=1)
    
    # Plot radius circle around grid center (shows the full extent of the grid)
    # Add extra spacing to show the full outer ring
    display_radius = (n_rings + 0.5) * spacing_km * 1000
    circle = plt.Circle((grid_center_x, grid_center_y), display_radius, fill=False, 
                       edgecolor='darkred', linewidth=3, linestyle='--', alpha=0.7)
    ax.add_patch(circle)
    
    # Generate hexagonal grid for proposed magnetometers
    proposed_sites = create_hexagonal_grid(grid_center_x, grid_center_y, 
                                         spacing_km, n_rings, transformer)
    
    # Plot proposed sites
    for x, y, lon, lat in proposed_sites:
        ax.scatter(x, y, c='lightgreen', s=100, marker='o', 
                  edgecolor='darkgreen', alpha=0.8, linewidth=2, zorder=5)
    
    # Plot existing magnetometers
    for lon, lat, name in magnetometer_coords:
        x, y = transformer.transform(lon, lat)
        # Check if within plot area relative to grid center
        dist = np.sqrt((x - grid_center_x)**2 + (y - grid_center_y)**2)
        if dist <= radius_km * 1000 * 1.2:  # Show stations slightly outside radius too
            ax.scatter(x, y, c='red', s=200, marker='^', edgecolor='darkred', 
                      linewidth=2, zorder=6)
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=11, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='red', alpha=0.9))
    
    # Plot EISCAT 3D location
    ax.scatter(center_x, center_y, c='gold', s=700, marker='*', edgecolor='orange', 
              linewidth=3, zorder=7)
    ax.annotate('EISCAT 3D\nSkibotn', (center_x, center_y), xytext=(15, 15), 
               textcoords='offset points', fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                        edgecolor='orange', alpha=0.9),
               ha='center')
    
    # Add scale bar
    scalebar_length = 300000  # 300 km
    scalebar_x = ax.get_xlim()[0] + 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    scalebar_y = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    # Scale bar background
    bar_height = 15000
    rect = plt.Rectangle((scalebar_x - 10000, scalebar_y - bar_height/2), 
                        scalebar_length + 20000, bar_height * 2,
                        facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    
    # Scale bar
    ax.plot([scalebar_x, scalebar_x + scalebar_length], [scalebar_y, scalebar_y], 
           'k-', linewidth=4)
    
    # Add ticks
    for i in range(4):
        tick_x = scalebar_x + i * 100000
        ax.plot([tick_x, tick_x], [scalebar_y - 5000, scalebar_y + 5000], 'k-', linewidth=2)
        ax.text(tick_x, scalebar_y - 10000, f'{i*100}', ha='center', va='top', fontsize=10)
    
    ax.text(scalebar_x + scalebar_length/2, scalebar_y + 10000, 'km', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add north arrow
    arrow_x = ax.get_xlim()[1] - 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    arrow_y = ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    arrow_length = 100000
    
    # Arrow background
    circle_bg = plt.Circle((arrow_x, arrow_y - arrow_length/2), arrow_length * 0.7, 
                          facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(circle_bg)
    
    # Arrow
    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - arrow_length),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(arrow_x, arrow_y - arrow_length - 20000, 'N', ha='center', va='top',
           fontsize=16, fontweight='bold')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=12, label='Existing Magnetometer'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markeredgecolor='orange', markersize=18, label='EISCAT 3D'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
               markeredgecolor='darkgreen', markersize=10, label=f'Proposed Sites (n={len(proposed_sites)})'),
        Line2D([0], [0], color='darkred', linewidth=3, linestyle='--', 
               label=f'{n_rings} rings ({n_rings * spacing_km} km)'),
        Line2D([0], [0], color='#4a5568', linewidth=2, label='Coastline'),
        Line2D([0], [0], color='#e990a0', linewidth=3, label='Major Roads'),
        Line2D([0], [0], color='#fdd7a1', linewidth=2, label='Primary Roads'),
        Line2D([0], [0], color='#dddddd', linewidth=1, label='Minor Roads')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Easting (m)', fontsize=13)
    ax.set_ylabel('Northing (m)', fontsize=13)
    
    # Title
    ax.set_title('Proposed Magnetometer Network Around EISCAT 3D\n' + 
                f'{spacing_km} km Grid Spacing - {n_rings} Rings ({n_rings * spacing_km} km extent)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set axis limits with margin based on grid center
    # Use display_radius to ensure all magnetometers are visible
    margin = display_radius * 0.15
    ax.set_xlim(grid_center_x - display_radius - margin, grid_center_x + display_radius + margin)
    ax.set_ylim(grid_center_y - display_radius - margin, grid_center_y + display_radius + margin)
    
    # Add coordinate info
    info_text = f'EISCAT 3D: {center_lat:.2f}°N, {center_lon:.2f}°E\n'
    info_text += f'Grid Center: {grid_center_lat:.2f}°N, {grid_center_lon:.2f}°E\n'
    info_text += f'Projection: Arctic Polar Stereographic (EPSG:3995)'
    ax.text(0.99, 0.01, info_text, transform=ax.transAxes,
           fontsize=9, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax, proposed_sites

base = os.getcwd()
path_in = os.path.join(base, 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))

lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()[lat>0]
lat = lat[lat>0]
magnetometers = [(lo, la, 'st') for lo, la in zip(lon, lat)]

# Your magnetometer stations
#magnetometers = [
#    (23.7, 67.8, 'KIL'),  # Kilpisjärvi
#    (20.4, 69.3, 'TRO'),  # Tromsø  
#    (18.9, 69.7, 'AND'),  # Andenes
#    (16.0, 69.3, 'SOR'),  # Sørøya
#    (23.7, 70.2, 'MAS'),  # Masi
#    (27.0, 68.6, 'IVA'),  # Ivalo
#    (26.6, 67.4, 'SOD'),  # Sodankylä
#    (25.8, 69.8, 'KEV'),  # Kevo
#    (21.0, 70.2, 'ALT'),  # Alta
#    (18.6, 68.4, 'NAR'),  # Narvik
#]

# EISCAT 3D location (Skibotn)
eiscat_lat, eiscat_lon = 69.58, 20.31

# Create the plot - this will download data automatically
# Use a point further inland as the grid center to avoid magnetometers in water
grid_center_lat = 68.5  # Further south and inland
grid_center_lon = 25.5  # Further east

fig, ax, proposed_sites = plot_magnetometer_network_osm(
    magnetometers, eiscat_lat, eiscat_lon, spacing_km=100, n_rings=3,
    grid_center_lat=grid_center_lat, grid_center_lon=grid_center_lon)

# Print statistics
print(f"\nNetwork Statistics:")
print(f"- Proposed magnetometer sites: {len(proposed_sites)}")
print(f"- Coverage area: ~{np.pi * (3 * 100)**2 / 1e6:.1f} million km²")
print(f"- Existing stations shown: {len(magnetometers)}")

# Save the figure
#plt.savefig('magnetometer_network_eiscat3d.png', dpi=300, bbox_inches='tight', facecolor='white')