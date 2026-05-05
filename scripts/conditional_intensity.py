#%%
"""
The purpose of this file is to evaluate the conditional intensity
lambda(x,y,t | H_t) of an etas model (already inverted) on a grid
of points. The purpose being to use this as a prior.

The fundamental equation is in triggering_kernel in etas.inversion
We need to sum over all possible source events for ever target event (
or target grid point)

During inversion, this is done explicitly in 'expectation_step':

        Pij_0["tot_rates"] = (
            Pij_0["tot_rates"]
            .add((Pij_0["gij"] * Pij_0["xi_plus_1"]).groupby(level=1).sum())
            .add(target_events_0["mu"])
        )

Here, for evaluation, we neglect xi_plus_1, which compensates 
for unseen events during inversion

TODO - run a simulation, get full distribution for each grid point as well
(This will MCMC what we are analytically doing here).

"""

#%%
import numpy as np
from shapely.geometry import Point, Polygon
import json
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

# canonical implementation lives in etas.intensity so it can be imported
# by external packages (e.g. priors.EtasPriorUpdater) without executing
# this script's top-level code.
from etas.intensity import conditional_intensity_grid  # noqa: F401


def save_lambda_grid_tt3(lats, lons, lambda_grid, filepath):
    """
    Save a lambda_grid to a .tt3 file readable by SeismicPrior and bEPIC.

    Takes the masked-point output of conditional_intensity_grid() directly
    and writes it in the .tt3 format: a 6-line header followed by a
    normalized, vertically-flipped probability grid.

    Parameters
    ----------
    lats : array-like
        1-D array of grid point latitudes (masked subset), same length as lambda_grid.
    lons : array-like
        1-D array of grid point longitudes (masked subset), same length as lambda_grid.
    lambda_grid : array-like
        1-D array of conditional intensity values from conditional_intensity_grid().
    filepath : str
        Output path (e.g. 'etas_prior_20080101.tt3').
    """
    lats        = np.asarray(lats,        dtype=float)
    lons        = np.asarray(lons,        dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    # Pivot masked points onto a full rectangular grid; cells outside the
    # polygon mask are absent from lats/lons and filled with 0.
    grid = (
        pd.DataFrame({'lon': lons, 'lat': lats, 'rate': lambda_grid})
        .pivot(index='lat', columns='lon', values='rate')
        .fillna(0.0)
    )

    grid_lons = np.asarray(grid.columns, dtype=float)
    grid_lats = np.asarray(grid.index,   dtype=float)
    grid_vals = grid.values
    grid_vals = grid_vals / grid_vals.sum()

    mx     = len(grid_lons)
    my     = len(grid_lats)
    xlower = float(grid_lons.min())
    ylower = float(grid_lats.min())
    dx     = float(np.diff(grid_lons).mean())
    dy     = float(np.diff(grid_lats).mean())

    with open(filepath, 'w') as f:
        f.write(f"{mx}\n{my}\n{xlower}\n{ylower}\n{dx:.6f}\n{dy:.6f}\n")
        np.savetxt(f, np.flipud(grid_vals), fmt='%.6e')


#%%
# Open inversion output
with open("./output_data/parameters_0.json", 'r') as f:
    inversion_config = json.load(f)
# Get polygon from inversion output
from numpy import array
polygon = Polygon(np.array(eval(inversion_config["shape_coords"])))

# Get timestamp from inversion output (or specify manually)
forecast_time = pd.Timestamp("2008-01-01 00:00:00")
#forecast_time = pd.Timestamp(inversion_config["timewindow_end"])
#forecast_time = pd.Timestamp(inversion_config["testwindow_end"])

# Get set of inverted parameters (will only need a few, but this is how htey are stored)
theta = inversion_config["final_parameters"]

# get reference magnitude of completeness
mc = inversion_config["m_ref"]

#%%

# Set up grid for evaluation
min_lat, min_lon, max_lat, max_lon = polygon.bounds

lats = np.arange(min_lat, max_lat, 0.1)
lons = np.arange(min_lon, max_lon, 0.1)

grid_lons, grid_lats = np.meshgrid(lons, lats)

lats_flat = grid_lats.ravel()
lons_flat = grid_lons.ravel()

# Read in event catalog
catalog = pd.read_csv(
                inversion_config["fn_catalog"],
                index_col=0,
                parse_dates=["time"],
                dtype={"url": str, "alert": str},
)
mask = np.array([polygon.contains(Point(lat, lon)) 
                 for lat, lon in zip(lats_flat, lons_flat)])

# Evaluate on grid
lambda_grid = conditional_intensity_grid(
    forecast_time, lats_flat[mask], lons_flat[mask], catalog, theta, mc
)

#%%

import os
save_dir = "/home/a01738353/2024_NEHRP/priors/data"
filename = f"etas_{forecast_time.strftime('%Y%m%d_%H%M%S')}.tt3"
save_lambda_grid_tt3(lats_flat[mask], lons_flat[mask], lambda_grid, os.path.join(save_dir, filename))

#save_name = "etas_.tt3"

#save_lambda_grid_tt3(lats_flat, lons_flat, lambda_grid, f"{save_dir}/{save_name}")


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
plt.title(f'ETAS_2 Conditional Intensity at t={forecast_time}')
plt.show()
# %%
