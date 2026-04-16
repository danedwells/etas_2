"""
intensity.py — ETAS conditional intensity evaluation on a spatial grid.

Extracted from etas_2/conditional_intensity.py so that it can be imported
by other packages (e.g. priors.time_dependent.EtasPriorUpdater) without
executing the top-level script logic.

The core equation sums the ETAS triggering kernel contribution from every
past event to every grid point, then adds the background rate mu.
"""

import numpy as np

from etas.inversion import triggering_kernel, parameter_dict2array, to_days, haversine


def conditional_intensity_grid(forecast_time, grid_lats, grid_lons,
                                past_catalog, theta, mc):
    """
    Evaluate the ETAS conditional intensity lambda(x,y,t | H_t) on a grid.

    Parameters
    ----------
    forecast_time : pd.Timestamp
        The time at which to evaluate the intensity.
    grid_lats : np.ndarray, shape (n_grid,)
        Latitudes of grid cell centres to evaluate.
    grid_lons : np.ndarray, shape (n_grid,)
        Longitudes of grid cell centres to evaluate.
    past_catalog : pd.DataFrame
        Event catalog with columns: time (datetime), latitude, longitude,
        magnitude.  Only events with time < forecast_time are used.
    theta : dict
        Inverted ETAS parameters (as stored in parameters_0.json under
        'final_parameters').  Must contain 'log10_mu'.
    mc : float
        Reference magnitude of completeness (m_ref from inversion output).

    Returns
    -------
    lambda_vals : np.ndarray, shape (n_grid,)
        Conditional intensity at each grid point.  Units are events/day
        (inheriting from the ETAS inversion time convention).

    Notes
    -----
    xi_plus_1 (the responsibility factor that compensates for unseen events
    during inversion) is omitted here — appropriate for prospective evaluation
    where the catalog is assumed complete up to forecast_time.
    """
    theta_arr = parameter_dict2array(theta)
    mu = 10 ** theta['log10_mu']

    past = past_catalog[past_catalog['time'] < forecast_time]
    dt = to_days(forecast_time - past['time']).values   # (n_events,)

    # r_sq: (n_grid, n_events) via broadcasting
    # NOTE: the polygon and grid use (lat, lon) = (y, x) ordering
    # inherited from the inversion pipeline — haversine expects radians.
    r_sq = np.square(haversine(
        np.radians(past['latitude'].values)[None, :],    # (1, n_events)
        np.radians(grid_lats)[:, None],                  # (n_grid, 1)
        np.radians(past['longitude'].values)[None, :],   # (1, n_events)
        np.radians(grid_lons)[:, None],                  # (n_grid, 1)
    ))   # (n_grid, n_events)

    gij = triggering_kernel(
        [dt[None, :], r_sq, past['magnitude'].values[None, :], None],
        [theta_arr, mc],
    )   # (n_grid, n_events)

    return mu + gij.sum(axis=1)   # (n_grid,)
