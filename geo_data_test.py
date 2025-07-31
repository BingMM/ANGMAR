#%% Import

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon
import gzip
import xml.etree.ElementTree as ET

#%%
# Function to parse OSM data from .osm.gz file
def parse_osm_gz(filename):
    """
    Basic OSM parser for .osm.gz files
    Returns dictionaries of roads, water features, and peaks
    """
    roads = []
    water = []
    peaks = []
    nodes = {}  # Store node coordinates
    
    with gzip.open(filename, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        
        # First pass: collect all nodes
        for node in root.findall('node'):
            node_id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            nodes[node_id] = (lon, lat)
            
            # Check if it's a peak
            for tag in node.findall('tag'):
                if tag.get('k') == 'natural' and tag.get('v') == 'peak':
                    name = ''
                    ele = ''
                    for t in node.findall('tag'):
                        if t.get('k') == 'name':
                            name = t.get('v')
                        if t.get('k') == 'ele':
                            ele = t.get('v')
                    peaks.append({
                        'geometry': Point(lon, lat),
                        'name': name,
                        'elevation': ele
                    })
        
        # Second pass: collect ways (roads and water)
        for way in root.findall('way'):
            tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            
            # Get coordinates for this way
            coords = []
            for nd in way.findall('nd'):
                ref = nd.get('ref')
                if ref in nodes:
                    coords.append(nodes[ref])
            
            if len(coords) < 2:
                continue
                
            # Check if it's a road
            if 'highway' in tags:
                roads.append({
                    'geometry': LineString(coords),
                    'highway': tags['highway'],
                    'name': tags.get('name', '')
                })
            
            # Check if it's water
            elif tags.get('natural') == 'water' or 'waterway' in tags:
                if coords[0] == coords[-1] and len(coords) > 3:  # Closed polygon
                    water.append({
                        'geometry': Polygon(coords),
                        'name': tags.get('name', '')
                    })
                else:
                    water.append({
                        'geometry': LineString(coords),
                        'name': tags.get('name', '')
                    })
    
    return roads, water, peaks

#%%

# Convert parsed data to GeoDataFrames
def create_geodataframes(roads, water, peaks):
    """Convert parsed OSM data to GeoDataFrames"""
    roads_gdf = gpd.GeoDataFrame(roads, crs='EPSG:4326')
    water_gdf = gpd.GeoDataFrame(water, crs='EPSG:4326')
    peaks_gdf = gpd.GeoDataFrame(peaks, crs='EPSG:4326')
    
    # Transform to stereographic projection for high latitudes
    roads_gdf = roads_gdf.to_crs('EPSG:3995')
    water_gdf = water_gdf.to_crs('EPSG:3995')
    peaks_gdf = peaks_gdf.to_crs('EPSG:3995')
    
    return roads_gdf, water_gdf, peaks_gdf

#%%

# Main plotting function
def plot_magnetometer_network(osm_file, magnetometer_coords, center_lat, center_lon, radius_km=1500):
    """
    Plot magnetometer network with OSM background
    
    Parameters:
    -----------
    osm_file : str
        Path to .osm.gz file
    magnetometer_coords : list of tuples
        [(lon1, lat1, name1), (lon2, lat2, name2), ...]
    center_lat, center_lon : float
        EISCAT 3D location (Skibotn)
    radius_km : float
        Radius of interest in km
    """
    
    # Parse OSM data
    print("Parsing OSM data...")
    roads, water, peaks = parse_osm_gz(osm_file)
    roads_gdf, water_gdf, peaks_gdf = create_geodataframes(roads, water, peaks)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Transform coordinates to projected system
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995", always_xy=True)
    
    # Plot OSM features
    print("Plotting map features...")
    
    # Water features
    water_gdf.plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=0.5)
    
    # Roads (style by type)
    road_styles = {
        'motorway': {'color': 'red', 'linewidth': 2},
        'trunk': {'color': 'orange', 'linewidth': 1.5},
        'primary': {'color': 'gold', 'linewidth': 1.2},
        'secondary': {'color': 'gray', 'linewidth': 1},
        'tertiary': {'color': 'lightgray', 'linewidth': 0.8}
    }
    
    for road_type, style in road_styles.items():
        roads_subset = roads_gdf[roads_gdf['highway'] == road_type]
        if len(roads_subset) > 0:
            roads_subset.plot(ax=ax, **style, alpha=0.7)
    
    # Plot all other roads in light gray
    other_roads = roads_gdf[~roads_gdf['highway'].isin(road_styles.keys())]
    if len(other_roads) > 0:
        other_roads.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # Mountains/peaks
    if len(peaks_gdf) > 0:
        peaks_gdf.plot(ax=ax, color='brown', marker='^', markersize=30, alpha=0.7)
    
    # Plot existing magnetometers
    print("Plotting magnetometers...")
    for lon, lat, name in magnetometer_coords:
        x, y = transformer.transform(lon, lat)
        ax.scatter(x, y, c='red', s=150, marker='^', edgecolor='black', 
                  linewidth=2, zorder=5)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot EISCAT 3D location
    center_x, center_y = transformer.transform(center_lon, center_lat)
    ax.scatter(center_x, center_y, c='gold', s=500, marker='*', edgecolor='black', 
              linewidth=2, zorder=5)
    ax.annotate('EISCAT 3D', (center_x, center_y), xytext=(10, 10), 
               textcoords='offset points', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Add radius circle
    circle = plt.Circle((center_x, center_y), radius_km * 1000, fill=False, 
                       edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(circle)
    
    # Generate grid points for proposed magnetometers (150 km spacing)
    spacing_m = 150000  # 150 km in meters
    proposed_sites = []
    
    # Create hexagonal grid for better coverage
    for i in range(-10, 11):
        for j in range(-10, 11):
            if i % 2 == 0:
                x = center_x + i * spacing_m * np.sqrt(3)/2
                y = center_y + j * spacing_m
            else:
                x = center_x + i * spacing_m * np.sqrt(3)/2
                y = center_y + (j + 0.5) * spacing_m
            
            # Check if within radius
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius_km * 1000:
                # Transform back to lat/lon
                lon, lat = transformer.transform(x, y, direction='INVERSE')
                proposed_sites.append((x, y, lon, lat))
                # Plot proposed location
                ax.scatter(x, y, c='lightgreen', s=80, marker='o', 
                          edgecolor='darkgreen', alpha=0.7, linewidth=1.5, zorder=3)
    
    # Add scale bar
    scalebar_length = 200000  # 200 km
    scalebar_x = ax.get_xlim()[0] + 0.1 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    scalebar_y = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([scalebar_x, scalebar_x + scalebar_length], [scalebar_y, scalebar_y], 
           'k-', linewidth=4)
    ax.text(scalebar_x + scalebar_length/2, scalebar_y - 15000, '200 km', 
           ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Add north arrow
    arrow_x = ax.get_xlim()[1] - 0.1 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    arrow_y = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 100000),
               arrowprops=dict(arrowstyle='->', lw=2), ha='center', va='bottom',
               fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='r', 
               markersize=10, label='Existing Magnetometer'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markersize=15, label='EISCAT 3D'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
               markeredgecolor='darkgreen', markersize=8, label='Proposed Sites'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', 
               label=f'{radius_km} km radius')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title(f'Magnetometer Network: EISCAT 3D Region\n{len(proposed_sites)} Proposed Sites with 150 km Spacing', 
                fontsize=16, fontweight='bold')
    
    # Set reasonable axis limits based on the radius
    margin = radius_km * 1000 * 0.1  # 10% margin
    ax.set_xlim(center_x - radius_km*1000 - margin, center_x + radius_km*1000 + margin)
    ax.set_ylim(center_y - radius_km*1000 - margin, center_y + radius_km*1000 + margin)
    
    plt.tight_layout()
    
    return fig, ax, proposed_sites

#%%
base = os.getcwd()
path_in = os.path.join(base, 'data', '20250717-14-57-supermag-stations.csv')
st_data = pd.read_csv(path_in, usecols=range(6))

lat = st_data['GEOLAT'].to_numpy()
lon = st_data['GEOLON'].to_numpy()[lat>0]
lat = lat[lat>0]
magnetometers = [(lo, la, 'st') for lo, la in zip(lon, lat)]

# Example usage
# Define your magnetometer stations from SuperMAG
#magnetometers = [
#    (23.7, 67.8, 'KIL'),  # Kilpisjärvi
#    (20.4, 69.3, 'TRO'),  # Tromsø  
#    (18.9, 69.7, 'AND'),  # Andenes
#    (16.0, 69.3, 'SOR'),  # Sørøya
#    (23.7, 70.2, 'MAS'),  # Masi
#    (27.0, 68.6, 'IVA'),  # Ivalo
#    (26.6, 67.4, 'SOD'),  # Sodankylä
#]

# EISCAT 3D location (Skibotn)
eiscat_lat, eiscat_lon = 69.58, 20.31

# Create the plot
fig, ax, proposed_sites = plot_magnetometer_network('/home/bing/Downloads/planet_18,67_30,70.osm.gz', 
                                                    magnetometers, 
                                                    eiscat_lat, eiscat_lon,
                                                    radius_km=500)

# Save the figure
# plt.savefig('magnetometer_network_scandanavia.png', dpi=300, bbox_inches='tight')

# Print some statistics
# print(f"Number of proposed magnetometer sites: {len(proposed_sites)}")
# print(f"Coverage area: ~{np.pi * 1500**2:.0f} km²")