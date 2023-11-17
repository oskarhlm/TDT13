import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from matplotlib.colors import AsinhNorm

from .metrics import haversine_distance

def plot_switzerland(test_preds, test_gold_data, data_path):
    city_gdf = gpd.read_file(f'{data_path}/swiss_cities.csv', index=False)
    city_gdf['geometry'] = [Point(lon, lat) for lon, lat in zip(city_gdf['lon'], city_gdf['lat'])]

    switzerland_polygon = gpd.read_file(f'{data_path}/switzerland.geojson')

    predicted_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in test_preds])
    predicted_points['distance'] = haversine_distance(test_preds, test_gold_data[['lat', 'lon']].to_numpy())

    fig_dim_size = 120
    font_size = fig_dim_size * 0.8
    ax = switzerland_polygon.plot(figsize=(fig_dim_size, fig_dim_size), alpha=0.1, edgecolor='k', linewidth=fig_dim_size * 0.1)

    norm = AsinhNorm(vmin=predicted_points['distance'].min(), vmax=predicted_points['distance'].max())
    sc = ax.scatter(predicted_points.geometry.x, predicted_points.geometry.y, c=predicted_points['distance'], 
                    cmap='inferno', s=fig_dim_size * 0.9, marker='o', edgecolors='k', norm=norm)

    # city_gdf.plot(ax=ax, markersize=fig_dim_size * 30, color='red', alpha=0.3, marker='o', label='Cities')
    for x, y, label in zip(city_gdf.geometry.x, city_gdf.geometry.y, city_gdf['city']):
        ax.annotate(label, xy=(x, y), xytext=(-font_size, font_size), textcoords='offset points', fontsize=font_size * 0.8,
                    bbox=dict(boxstyle='round,pad=0.2', edgecolor='blue', facecolor='lightblue'))

    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    cbar = plt.colorbar(sc, ax=ax, label='Distance to Ground Truth (km)', pad=0.01, shrink=0.4)
    cbar.ax.tick_params(labelsize=font_size)  
    cbar.ax.set_ylabel('Distance to Ground Truth (km)', fontsize=font_size)

    plt.show()


def plot_barchart(test_preds, test_gold_data):
    num_bins = 15

    distances = haversine_distance(test_preds, test_gold_data[['lat', 'lon']].to_numpy())
    
    distance_bins = np.linspace(np.min(distances), np.max(distances), num_bins + 1)
    hist, bin_edges = np.histogram(distances, bins=distance_bins)

    plt.figure(figsize=(20, 10))
    bar_width = np.diff(bin_edges)[0] * 0.8 
    plt.bar(bin_edges[:-1], hist, width=bar_width, align='edge', color='b', alpha=0.7)
    plt.xlabel('Distance to Ground Truth (km)')
    plt.ylabel('Number of Data Points')
    plt.title('Distribution of Predicted Distances to Ground Truth')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()