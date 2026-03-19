#%%
from etas.inversion import triggering_kernel, parameter_dict2array, to_days, haversine
import numpy as np
from shapely.geometry import Point, Polygon
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

def conditional_intensity_grid(forecast_time, grid_lats, grid_lons,
                                past_catalog, theta, mc):
    """
    Returns array of shape (n_grid,) — lambda at each grid point.
    grid_lats, grid_lons: 1D arrays of grid cell centers
    """
    theta_arr = parameter_dict2array(theta)
    mu = 10 ** theta['log10_mu']
    
    past = past_catalog[past_catalog['time'] < forecast_time]
    dt = to_days(forecast_time - past['time']).values  # (n_events,)
    
    # r_sq: (n_grid, n_events) via broadcasting
    r_sq = np.square(haversine(
        np.radians(past['latitude'].values)[None, :],   # (1, n_events)
        np.radians(grid_lats)[:, None],                 # (n_grid, 1)
        np.radians(past['longitude'].values)[None, :],  # (1, n_events)
        np.radians(grid_lons)[:, None],                 # (n_grid, 1)
    ))  # (n_grid, n_events)

    gij = triggering_kernel(
        [dt[None, :], r_sq, past['magnitude'].values[None, :], None],
        [theta_arr, mc]
    )  # (n_grid, n_events)

    return mu + gij.sum(axis=1)  # (n_grid,)




#%%
with open("./output_data/parameters_0.json", 'r') as f:
    inversion_config = json.load(f)
forecast_time = pd.Timestamp(inversion_config["timewindow_end"])
from numpy import array
polygon = Polygon(np.array(eval(inversion_config["shape_coords"])))
theta = inversion_config["final_parameters"]
mc = inversion_config["m_ref"]

#%%
min_lat, min_lon, max_lat, max_lon = polygon.bounds

lats = np.arange(min_lat, max_lat, 0.05)
lons = np.arange(min_lon, max_lon, 0.05)

grid_lons, grid_lats = np.meshgrid(lons, lats)

lats_flat = grid_lats.ravel()
lons_flat = grid_lons.ravel()



catalog = pd.read_csv(
                inversion_config["fn_catalog"],
                index_col=0,
                parse_dates=["time"],
                dtype={"url": str, "alert": str},
)
mask = np.array([polygon.contains(Point(lat, lon)) 
                 for lat, lon in zip(lats_flat, lons_flat)])

lambda_grid = conditional_intensity_grid(
    forecast_time, lats_flat[mask], lons_flat[mask], catalog, theta, mc
)
# %%

bounds = [-127,-113,30,45]
transform = ccrs.PlateCarree()



fig, ax = plt.subplots(subplot_kw={'projection': transform})

ax.set_extent(bounds)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE)

sc = ax.scatter(lons_flat[mask],lats_flat[mask],
                c=lambda_grid, s=4, marker='s',
                cmap='hot_r', norm=LogNorm(),
                transform=transform)

plt.colorbar(sc, label=f'ETAS_2 output')
plt.title(f'ETAS_2 Conditional Intensity')
plt.show()
# %%
