"""
intensity.py — ETAS conditional intensity evaluation on a spatial grid.

Extracted from etas_2/conditional_intensity.py so that it can be imported
by other packages (e.g. priors.time_dependent.EtasPriorUpdater) without
executing the top-level script logic.

The core equation sums the ETAS triggering kernel contribution from every
past event to every grid point, then adds the background rate mu.
"""

import datetime

import numpy as np

from etas.inversion import to_days, haversine


def _compute_spatial_weights(grid_lats, grid_lons, lats, lons, mags, theta, mc):
    """Time-independent (space × magnitude) component of the ETAS kernel.

    Returns (n_grid, n_events) float32.  Caller is responsible for temporal
    filtering — only pass events that should contribute to the forecast.
    """
    k0    = 10 ** theta['log10_k0']
    d     = 10 ** theta['log10_d']
    a     = theta['a']
    gamma = theta['gamma']
    rho   = theta['rho']

    # NOTE: polygon/grid use (lat, lon) = (y, x) ordering inherited from the
    # inversion pipeline; haversine expects radian angles in that order.
    r_sq = np.square(haversine(
        np.radians(lats)[None, :],       # (1, n_events)
        np.radians(grid_lats)[:, None],  # (n_grid, 1)
        np.radians(lons)[None, :],       # (1, n_events)
        np.radians(grid_lons)[:, None],  # (n_grid, 1)
    )).astype(np.float32)                # (n_grid, n_events)

    aftershock_n = (k0 * np.exp(a * (mags - mc))).astype(np.float32)               # (n_events,)
    space_decay  = (1.0 / np.power(
        r_sq + d * np.exp(gamma * (mags - mc)), 1 + rho,
    )).astype(np.float32)                                                            # (n_grid, n_events)

    return aftershock_n * space_decay                                                # (n_grid, n_events)


def _compute_time_decay(event_times, forecast_time, theta):
    """Time-dependent component of the ETAS kernel.

    event_times may be a pd.Series or a numpy datetime64 array.
    Returns (n_events,) float32.
    """
    c     = 10 ** theta['log10_c']
    tau   = 10 ** theta['log10_tau']
    omega = theta['omega']

    dt = to_days(forecast_time - event_times)
    if hasattr(dt, 'values'):
        dt = dt.values
    return (np.exp(-dt / tau) / np.power(dt + c, 1 + omega)).astype(np.float32)


def conditional_intensity_grid(forecast_time, grid_lats, grid_lons,
                                past_catalog, theta, mc,
                                max_lookback_days=None):
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
    max_lookback_days : float or None
        If given, ignore events older than this many days before forecast_time.
        None (default) uses the full catalog.  730 days is a reasonable value
        for most ETAS configurations: Omori power-law decay makes events beyond
        ~2 years contribute negligibly to the conditional intensity.

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
    mu   = 10 ** theta['log10_mu']
    past = past_catalog[past_catalog['time'] < forecast_time]

    if max_lookback_days is not None:
        cutoff = forecast_time - datetime.timedelta(days=max_lookback_days)
        past   = past[past['time'] >= cutoff]

    if len(past) == 0:
        return np.full(len(grid_lats), mu)

    w  = _compute_spatial_weights(
            grid_lats, grid_lons,
            past['latitude'].values, past['longitude'].values,
            past['magnitude'].values, theta, mc)
    td = _compute_time_decay(past['time'], forecast_time, theta)

    return mu + (w * td).sum(axis=1)   # (n_grid,)
