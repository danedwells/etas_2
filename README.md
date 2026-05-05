# ETAS Codebase — Detailed Technical Reference

This document describes every Python file in the repository in enough detail to understand, maintain, and extend the codebase without needing to re-read it from scratch.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Package Structure](#package-structure)
4. [Core Library: `etas/`](#core-library-etas)
   - [`__init__.py`](#etasinitpy)
   - [`mc_b_est.py`](#etasmc_b_estpy)
   - [`inversion.py`](#etasinversionpy)
   - [`simulation.py`](#etassimulationpy)
   - [`evaluation.py`](#etasevaluationpy)
   - [`plots.py`](#etasplotspy)
   - [`download.py`](#etasdownloadpy)
   - [`oef/__init__.py` and `oef/entrypoint.py`](#etasoef)
5. [Runnable Scripts: `runnable_code/`](#runnable-scripts-runnable_code)
6. [Top-Level Scripts](#top-level-scripts)
7. [Configuration Files: `config/`](#configuration-files-config)
8. [Data Flow: End-to-End Pipeline](#data-flow-end-to-end-pipeline)
9. [Key Data Structures](#key-data-structures)
10. [Parameter Reference](#parameter-reference)
11. [Tests: `tests/`](#tests)

---

## Overview

This repository implements the **ETAS (Epidemic Type Aftershock Sequence)** model for earthquake forecasting, following the formulation of Mizrahi et al. (2021). ETAS models seismicity as a branching process: each earthquake can trigger aftershocks, which can trigger their own aftershocks, and so on. The background seismicity rate (spontaneous events not triggered by prior earthquakes) is also modelled.

The pipeline has three main stages:

1. **Inversion** — fit ETAS parameters to an observed earthquake catalog using an Expectation-Maximisation (EM) algorithm.
2. **Simulation** — use inverted parameters to stochastically simulate future earthquake catalogs (Monte Carlo forecasting).
3. **Evaluation** — score the model using log-likelihood metrics against a held-out test period.

Key academic references:
- Mizrahi et al. (2021), *The Effect of Declustering on the Size Distribution of Mainshocks*, SRL. doi:10.1785/0220200231
- Mizrahi et al. (2021), *Embracing Data Incompleteness for Better Earthquake Forecasting*, JGR. doi:10.1029/2021JB022379
- Van der Elst (2021), *b-positive*, JGR Solid Earth, Vol 126, Issue 2.

---

## Mathematical Background

### ETAS Conditional Intensity

The conditional intensity at time *t* and location *(x, y)* given history H_t is:

```
λ*(t, x, y | H_t) = μ + Σ_{i: t_i < t} κ(m_i) · g(t - t_i) · f(x - x_i, y - y_i, m_i)
```

Where:
- **μ** = background rate (events per day per km²)
- **κ(m)** = productivity: `k0 * exp(a * (m - mc))` — number of expected aftershocks from an event of magnitude m
- **g(Δt)** = Omori-Utsu temporal decay with exponential taper:
  `exp(-Δt/τ) / (Δt + c)^(1+ω)`
- **f(Δx, Δy, m)** = spatial kernel:
  `ρ * (d * exp(γ*(m-mc)))^ρ / (r² + d*exp(γ*(m-mc)))^(1+ρ)`

### Gutenberg-Richter Magnitude Distribution

Magnitudes follow an exponential (G-R) law:
`P(M > m) = exp(-β * (m - mc))`

where `β = b * ln(10)` and `b ≈ 1` typically.

### EM Algorithm for Inversion

The inversion uses an Expectation-Maximisation loop:

1. **E-step**: compute the probability matrix **P_ij** (probability that event *i* triggered event *j*) using the current parameter estimate.
2. **M-step**: update parameters by maximising the expected complete-data log-likelihood (using scipy `minimize`).
3. Repeat until parameter change `Σ|θ_new - θ_old|` < 0.001.

---

## Package Structure

```
etas_2/
├── etas/                          # Main Python package
│   ├── __init__.py                # Logger setup
│   ├── inversion.py               # EM inversion, ETASParameterCalculation class
│   ├── simulation.py              # Catalog simulation, ETASSimulation class
│   ├── evaluation.py              # Log-likelihood evaluation, ETASLikelihoodCalculation
│   ├── mc_b_est.py                # Magnitude completeness & beta estimation
│   ├── plots.py                   # Fit visualisation
│   ├── download.py                # SED catalog download
│   └── oef/
│       ├── __init__.py            # Exports entrypoints
│       ├── entrypoint.py          # HERMES OEF interface (suiETAS, europe)
│       └── data/                  # Background rate grids (CSV)
│           ├── SUIhaz2015_rates.csv
│           └── europe_rate_map.csv
├── runnable_code/                 # Entry-point scripts
│   ├── invert_etas.py             # Run inversion
│   ├── forecast_catalog_continuation.py  # Run forecast simulation
│   ├── simulate_catalog_continuation.py  # Single simulation
│   ├── simulate_catalog.py        # Fresh catalog simulation (no conditioning)
│   ├── predict_etas.py            # Evaluate log-likelihood
│   ├── visualise_fit.py           # Generate fit plots
│   ├── estimate_mc.py             # Estimate mc and beta
│   ├── reload_example.py          # Load saved inversion results
│   └── ch_forecast.py             # Swiss full pipeline
├── run_entrypoints/               # Example HERMES entrypoint scripts
│   ├── run_entrypoint_sui.py
│   └── run_entrypoint_europe.py
├── config/                        # JSON configuration files
│   ├── invert_etas_config.json
│   ├── invert_etas_config.jsonc   # With comments
│   ├── forecast_catalog_continuation_config.json
│   ├── simulate_catalog_continuation_config.json
│   ├── ch_forecast_config.json
│   └── visualisation_config.json
├── input_data/                    # Example catalogs and shape files
│   ├── example_catalog.csv
│   └── california_shape.npy
├── output_data/                   # Output directory (created at runtime)
├── tests/
│   └── test_simulation.py
├── conditional_intensity.py       # Grid evaluation of λ*(t,x,y|H_t)
├── plot_simulated_catalog_continuation.py  # Cartopy map of simulated catalog
└── test_concept.py                # Minimal ETAS concept demo
```

---

## Core Library: `etas/`

### `etas/__init__.py`

Provides `set_up_logger(level=logging.DEBUG)` — configures the root logger with a timestamped format string. Called at the top of runnable scripts.

---

### `etas/mc_b_est.py`

Magnitude of completeness (Mc) and Gutenberg-Richter beta estimation.

#### Functions

**`round_half_up(n, decimals=0)`**
A deterministic rounding function. NumPy uses banker's rounding (round half to even), which is wrong for magnitude binning. This always rounds 0.5 up.

**`estimate_beta_tinti(magnitudes, mc, weights=None, axis=None, delta_m=0)`**
Maximum likelihood estimate of β using Tinti & Mulargia (1987). For binned magnitudes (`delta_m > 0`), uses the correction formula:
`β = (1/Δm) * ln(1 + Δm / mean(m - mc))`
For continuous magnitudes: `β = 1 / mean(m - mc)`.

**`estimate_beta_positive(magnitudes, delta_m=0)`**
Estimates β from positive magnitude differences only (Van der Elst 2021). Takes consecutive differences of time-sorted magnitudes, keeps only the positive ones, then calls `estimate_beta_tinti` on those differences with `mc = delta_m`. Robust to aftershock sequences because it de-emphasises completeness issues.

**`simulate_magnitudes(n, beta, mc, m_max=None)`**
Inverse-CDF sampling of G-R magnitudes:
`m = mc - ln(1 - u * norm_factor) / β`
where `u ~ Uniform(0,1)` and `norm_factor = 1 - exp(-β*(m_max - mc))` for truncated G-R. Returns array of n magnitudes ≥ mc.

**`simulate_magnitudes_from_zone(zones, mfds)`**
Draws one magnitude per event from zone-specific discrete CDF tables. `mfds` is a DataFrame with zone indices as rows and magnitude bin columns. Uses inverse-CDF lookup per zone.

**`fitted_cdf_discrete(sample, mc, delta_m, x_max=None, beta=None)`**
Returns (x, y) arrays for the fitted G-R cumulative distribution function evaluated at discrete magnitude bins.

**`empirical_cdf(sample, weights=None)`**
Returns (x, y) arrays for the empirical CDF of a sample, with optional weighting.

**`ks_test_gr(sample, mc, delta_m, ks_ds=None, n_samples=10000, beta=None)`**
Kolmogorov-Smirnov test for whether sample follows a G-R distribution above mc. Generates 10,000 synthetic G-R samples and compares the KS statistic of each against the real data's KS statistic to produce a p-value. Returns `(ks_d, p_value, ks_ds_list)`.

**`estimate_mc(sample, mcs_test, delta_m, p_pass, stop_when_passed=True, verbose=False, beta=None, n_samples=10000)`**
Main Mc estimation function. Tests a list of candidate Mc values in order, applying `ks_test_gr` at each. Returns the first Mc where the KS p-value ≥ `p_pass`. Returns `(mcs_test, ks_ds, ps, best_mc, beta)`.

---

### `etas/inversion.py`

The largest and most important file. Implements parameter inversion via EM.

#### Module-Level Constants

Parameter search bounds used as scipy optimize bounds:
```
LOG10_MU_RANGE    = (-10, 0)       # background rate
LOG10_IOTA_RANGE  = (-10, 0)       # not typically used in standard mode
LOG10_K0_RANGE    = (-20, 10)      # productivity prefactor
A_RANGE           = (0.01, 20)     # productivity magnitude scaling
LOG10_C_RANGE     = (-8, 0)        # Omori c (days)
OMEGA_RANGE       = (-0.99, 1)     # Omori decay shape = p - 1
LOG10_TAU_RANGE   = (0.01, 12.26)  # exponential taper timescale (days)
LOG10_D_RANGE     = (-4, 3)        # spatial length scale (km)
GAMMA_RANGE       = (-1, 5.0)      # spatial magnitude scaling
RHO_RANGE         = (0.01, 5.0)    # spatial decay exponent
```

#### Helper Functions

**`coppersmith(mag, fault_type)`**
Wells & Coppersmith (1994) scaling relations. Given magnitude and fault type (1=strike-slip, 2=reverse, 3=normal, 4=oblique), returns a dict of rupture dimensions in km: `{SRL, SSRL, RW, RA, AD}`. Used to define the maximum distance threshold between potentially correlated events (events farther apart than `coppersmith_multiplier * SSRL` are assumed uncorrelated and excluded from the P_ij matrix).

**`rectangle_surface(lat1, lat2, lon1, lon2)` / `polygon_surface(polygon)`**
Compute area in km² using pyproj Albers Equal Area projection. Essential for correctly computing background rate μ per unit area.

**`in_hull(points, x)`**
Tests whether point x is inside the convex hull of `points` using linear programming (scipy.optimize.linprog). Used for 3D inversion.

**`hav(theta)` / `haversine(lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2, earth_radius=6378.1)`**
Haversine formula for great-circle distance in km. Returns scalar or array depending on inputs.

**`branching_integral(alpha_minus_beta, dm_max=None)`**
Computes `∫exp(α_eff * dm) dm` over `[0, dm_max]`, the magnitude integral of the productivity kernel. For unbounded magnitudes, requires `alpha_minus_beta < 0` and returns `-1/alpha_minus_beta`.

**`branching_ratio(theta, beta, dm_max=None)`**
Computes the branching ratio η = expected total offspring per earthquake. η < 1 is required for stationarity. Formula integrates productivity over magnitude, time, and space.

**`to_days(timediff)`**
Converts a `datetime.timedelta` or pandas Timedelta to a float (days).

**`upper_gamma_ext(a, x)`**
Extended upper incomplete gamma function Γ(a, x). Handles negative `a` via recursion:
- `a > 0`: uses `scipy.special.gammaincc`
- `a == 0`: uses `scipy.special.exp1`
- `a < 0`: recursion: `Γ(a, x) = (Γ(a+1, x) - x^a * exp(-x)) / a`

**`parameter_array2dict(theta)` / `parameter_dict2array(parameters)`**
Convert between numpy parameter arrays (length 10 or 11 if α is included) and dicts with named keys. The standard key order is: `log10_mu, log10_iota, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho`.

**`create_initial_values(ranges=RANGES)`**
Randomly sample initial parameter values from uniform distributions within bounds.

**`triggering_kernel(metrics, params)`**
Core kernel function. Given:
- `time_distance`: Δt in days
- `spatial_distance_squared`: r² in km²
- `m`: magnitude of source event
- `source_kappa`: optional pre-computed productivity (replaces k0*exp(a*(m-mc)))
- `theta`: parameter array
- `mc`: reference magnitude

Returns the unnormalised likelihood that source event triggered target event. The formula is:
`κ(m) * exp(-Δt/τ) / (Δt+c)^(1+ω) * 1 / (r² + d*exp(γ*(m-mc)))^(1+ρ)`

**`responsibility_factor(theta, beta, delta_mc)`**
Computes `ξ+1 = exp(-(a - β - γρ) * Δmc)` — the inflation factor accounting for events below completeness that were not observed but still triggered aftershocks. Used to correct for incomplete catalogs.

**`observation_factor(beta, delta_mc)`**
Computes `ζ+1 = exp(β * Δmc)` — the correction for the fact that a higher completeness threshold means fewer observed events.

**`expected_aftershocks(event, params, no_start=False, no_end=False)`**
Computes the expected number of aftershocks triggered by an event, integrated over the time window and over all space. Parameters: `[log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho]` (theta without μ and ι). Handles three cases depending on `no_start`/`no_end`.

**`neg_log_likelihood(theta, Pij, source_events, mc_min)`**
Negative log-likelihood for the full ETAS model (excluding μ and ι). Composed of:
1. **Aftershock count term**: Poisson log-likelihood for the number of aftershocks per source event.
2. **Space-time distribution term**: weighted log-likelihood of when/where aftershocks occurred, weighted by P_ij.

**`neg_log_likelihood_free_prod(...)`**
Variant for flETAS (free-productivity mode) where each event's κ is estimated individually rather than via the exponential law.

**`prod_neg_log_lik(a, args)` / `calc_a_k0_from_kappa(kappa, m_diff, weights=1)`**
In free-productivity mode: given per-event κ estimates, fit the `a` and `log10_k0` parameters by MLE (after the main EM loop).

**`read_shape_coords(shape_coords)`**
Reads spatial boundary coordinates. Accepts: `None`, a string path to `.npy` file, a string containing a Python array literal (e.g., from JSON), or an array-like. Returns a numpy array of `[[lat, lon], ...]` pairs.

**`transform_parameters(par, beta, delta_m, dm_max_orig=None)`**
Re-scales ETAS parameters when changing the reference magnitude by `delta_m`. Updates `log10_mu`, `log10_d`, and `log10_k0` accordingly, preserving the physical interpretation.

**`parameters_from_standard_formulation(par_st, par_here, ...)`**
Converts parameters from the standard temporal-only ETAS formulation (α, log10_k, p, log10_c) to the spatial formulation used here (log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho).

**`parameters_from_etes_formulation(etes_par, par, ...)`**
Similar conversion from the ETES (Extended ETAS) formulation.

#### Class: `ETASParameterCalculation`

The main inversion class. Instantiated with a `metadata` dict (or JSON config). Key lifecycle:

1. **`__init__(metadata)`** — parses all config options. Does NOT read the catalog yet. Sets up time windows, mc handling, free_background/free_productivity flags.

2. **`load_calculation(cls, metadata)`** (classmethod) — reconstructs a completed `ETASParameterCalculation` from a saved JSON output file (the parameters_N.json output). Bypasses `prepare()` and `invert()`.

3. **`prepare()`** — full preparation pipeline:
   - Filters catalog to time window and spatial region
   - Computes pairwise distances (calls `calculate_distances()`) — the most computationally expensive step; uses Coppersmith threshold to limit matrix size
   - Separates into `target_events` (in primary window) and `source_events` (any event that could trigger a target)
   - Estimates β (fixed, b-positive, or Tinti method)
   - Sets up scipy constraints if `fixed_parameters` is specified (e.g., fixing α = β)

4. **`invert()`** — EM loop:
   - Calls `expectation_step()` to compute P_ij
   - Calls `optimize_parameters()` to update θ via scipy `minimize`
   - Checks convergence: `Σ|θ_new - θ_old| < 0.001`
   - If `free_productivity`: additionally calls `calc_a_k0_from_kappa()` and `update_source_kappa()` in each iteration
   - Returns `self.theta` (dict)

5. **`store_results(data_path, store_pij=False)`** — saves:
   - `parameters_N.json`: all metadata + final parameters (N = model ID)
   - `src_N.csv`: source events with P_background
   - `ip_N.csv`: target events
   - `dist_N.csv`: distance matrix
   - `pij_N.csv`: P_ij matrix (if `store_pij=True`)

**Key internal methods:**
- `filter_catalog(catalog)`: filters by time, polygon, magnitude (handles `mc='var'`, `mc='positive'`, or fixed mc). For `mc='positive'`, sets `mc_current = m_{i-1} + delta_m` for each event.
- `calculate_distances()`: pairwise Haversine distances with Coppersmith cutoff. Returns a long-format DataFrame with columns: `source_id, target_id, time_distance, spatial_distance_squared, source_magnitude, target_magnitude, ...`
- `expectation_step(theta, mc_min)`: given current θ, computes P_ij = (triggering probability), P_background, and auxiliary quantities n_hat, i_hat.
- `optimize_parameters(theta_old)`: calls scipy minimize with L-BFGS-B, bounds=RANGES, and any nonlinear constraints.

**Config options (metadata keys):**
| Key | Type | Description |
|-----|------|-------------|
| `fn_catalog` | str | Path to CSV catalog |
| `catalog` | DataFrame | Alternative to fn_catalog |
| `auxiliary_start` | str/datetime | Start of auxiliary (burn-in) period |
| `timewindow_start` | str/datetime | Start of primary (training) period |
| `timewindow_end` | str/datetime | End of training period |
| `mc` | float or `'var'` or `'positive'` | Completeness magnitude |
| `m_ref` | float | Reference magnitude (required if mc is 'var' or 'positive') |
| `delta_m` | float | Magnitude bin size |
| `coppersmith_multiplier` | float | Distance cutoff multiplier |
| `shape_coords` | str/array | Region polygon or path to .npy file |
| `beta` | float or `'positive'` | Fixed β or estimation method |
| `theta_0` | dict | Initial parameter guess |
| `fixed_parameters` | dict | Parameters to hold fixed during inversion |
| `free_background` | bool | Enable flETAS free background (default False) |
| `free_productivity` | bool | Enable flETAS free productivity (default False) |
| `bw_sq` | float | Gaussian kernel bandwidth² for free modes |
| `three_dim` | bool | 3D inversion mode (requires x, y, z columns) |

---

### `etas/simulation.py`

Earthquake catalog simulation using inverted ETAS parameters.

#### Standalone Functions

**`bin_to_precision(x, delta_x=0.1)`**
Rounds magnitudes to the nearest bin using `round_half_up`. Ensures consistency between inverted and simulated catalogs.

**`inverse_upper_gamma_ext(a, y)`**
Inverse of Γ(a, x) w.r.t. x: finds x such that Γ(a, x) = y. For `a > 0`, uses `scipy.special.gammainccinv`. For `a ≤ 0` (which occurs in ETAS when ω = p-1 < 0), falls back to `pynverse.inversefunc` with Nelder-Mead patching for NaN results. This is used in `simulate_aftershock_time`.

**`transform_parameters(par, beta, delta_m, dm_max_orig=None)`**
Re-scales ETAS parameters to a different reference magnitude (same function as in inversion.py, re-exported here for convenience).

**`simulate_aftershock_time(log10_c, omega, log10_tau, size=1)`**
Exact inverse-CDF sampling from the tapered Omori-Utsu distribution:
`t = Γ^{-1}(-ω, (1-u)*Γ(-ω, c/τ)) * τ - c`
where u ~ Uniform(0,1). Uses `inverse_upper_gamma_ext`.

**`simulate_aftershock_time_untapered(log10_c, omega, size=1)`**
Faster inverse-CDF for pure power-law (no taper):
`t = (1-u)^{-1/ω} * c - c`
Used when `log10_tau == inf`.

**`inv_time_cdf_approx(p, c, tau, omega)`** / **`simulate_aftershock_time_approx(...)`**
Piecewise approximation of the Omori-Utsu inverse CDF, split at τ: power-law for t < τ, exponential tail for t > τ. Faster than exact inversion but approximate.

**`simulate_aftershock_place(log10_d, gamma, rho, mi, mc)`**
Samples (x, y) offsets in km from triggering events using inverse-CDF of the spatial kernel:
`r = sqrt((1-u)^{-1/ρ} * d_g - d_g)` where `d_g = d * exp(γ * (mi - mc))`
Returns east and north km offsets.

**`simulate_aftershock_radius(log10_d, gamma, rho, mi, mc)`**
Same as above but returns just the radius (not (x, y) pair).

**`simulate_background_location(latitudes, longitudes, background_probs, scale=0.1, grid=False, bsla, bslo, n=1)`**
Samples background event locations from a reference set of past events (or a grid). Uses rejection sampling weighted by `background_probs`. If `grid=True`, adds uniform jitter within a grid cell; otherwise adds Gaussian jitter with scale `scale` degrees.

**`generate_background_events(polygon, timewindow_start, timewindow_end, parameters, beta, mc, ...)`**
Generates all background events in a simulation. Steps:
1. Draws n_background ~ Poisson(μ * area * duration).
2. Assigns locations (from reference grid or uniform within bounding box, filtered to polygon).
3. Assigns times uniformly in the time window.
4. Assigns magnitudes from G-R or zone-specific MFDs.
5. Computes `expected_n_aftershocks` for each background event.
Returns a DataFrame sorted by time.

**`generate_aftershocks(sources, generation, parameters, beta, mc, timewindow_end, ...)`**
Generates all aftershocks from a set of source events in one generation. Steps:
1. Samples time deltas for all aftershocks in bulk.
2. Distributes aftershocks across sources (repeating source row n_aftershocks times).
3. Filters events outside time window.
4. Samples radii and azimuths, converts to lat/lon offsets using haversine.
5. Optionally filters to polygon.
6. Assigns magnitudes.
7. Computes `expected_n_aftershocks` for the next generation.

**`prepare_auxiliary_catalog(auxiliary_catalog, parameters, mc, delta_m=0)`**
Prepares the conditioning catalog for a catalog-continuation simulation. Assigns `xi_plus_1` (responsibility factor), computes expected aftershock counts for each event in the auxiliary catalog, and draws `n_aftershocks` from Poisson.

**`generate_catalog(polygon, timewindow_start, timewindow_end, parameters, mc, beta_main, ...)`**
Generates a fully synthetic catalog (no conditioning on past events). Starts from scratch with background events, then iterates generations until no aftershocks remain.

**`simulate_catalog_continuation(auxiliary_catalog, auxiliary_start, auxiliary_end, polygon, simulation_end, parameters, mc, beta_main, ...)`**
The main simulation workhorse for forecasting. Conditions on an observed auxiliary catalog, generates new background events in the simulation period, and runs the generational aftershock loop. Returns all simulated events (auxiliary + background + aftershocks).

#### Class: `ETASSimulation`

High-level wrapper around `simulate_catalog_continuation` for repeated simulations.

**`__init__(inversion_params, gaussian_scale=0.1, approx_times=False, m_max=None, induced_info=None)`**
Takes a completed `ETASParameterCalculation` object. Optionally supports induced seismicity via `induced_info` = `(lats, lons, term, bsla, bslo, n_induced)`.

**`prepare()`**
Sets up the simulation:
- Creates polygon from `inversion_params.shape_coords`.
- Computes `xi_plus_1` for each source event.
- Merges source events with catalog to get lat/lon/time/magnitude.
- Extracts background probabilities P_background for each target event in the training period — these probabilities are used to spatially sample future background events.

**`simulate(forecast_n_days, n_simulations, m_threshold=None, filter_polygon=True, chunksize=100, info_cols=['is_background'], i_start=0)`**
Generator that yields simulated event DataFrames in chunks of `chunksize` simulations. Each iteration:
1. Calls `simulate_catalog_continuation`.
2. Filters to forecast time window, magnitude threshold, and polygon.
3. Bins magnitudes if `delta_m > 0`.
4. Yields a DataFrame with columns: `latitude, longitude, magnitude, time, catalog_id` + `info_cols`.

**`simulate_to_csv(fn_store, forecast_n_days, n_simulations, ...)`**
Streams simulation output to a CSV file in chunks. Handles resume: if the file already exists, reads the last `catalog_id` written and continues from where it left off.

**`simulate_to_df(forecast_n_days, n_simulations, m_threshold=None)`**
Collects all simulations into a single `seismostats.ForecastCatalog` object (used by the HERMES OEF interface).

---

### `etas/evaluation.py`

Model evaluation using exact log-likelihood computation on a held-out test window.

#### Standalone Functions

**`compute_dist_squared_from_i(i, lat_rads, long_rads, earth_radius)`**
Computes squared haversine distances from event i to all prior events (0..i-1). Used in the Lambda/lambd calculations.

**`to_days(timediff)`**
Same as in inversion.py but using `np.timedelta64` instead of `datetime.timedelta`.

#### Class: `ETASLikelihoodCalculation`

Inherits from `ETASParameterCalculation`. Evaluates a fitted ETAS model on a test period using pointprocess log-likelihood theory.

**`__init__(metadata)`**
Loads final parameters from a `parameters_N.json` output file. Extracts all 10 parameters (μ, ι, k0, a, c, ω, τ, d, γ, ρ) as scalar attributes. Also requires `testwindow_end` and `area` in metadata.

**`prepare(n)`**
- Calls `filter_catalog` (inheriting the parent's logic but extending to `testwindow_end`).
- Sorts catalog and assigns sequential integer indices.
- Extracts time, magnitude, lat/lon arrays.
- Precomputes a time integral lookup table using n Monte Carlo samples from the temporal kernel (`_precompute_integral`).
- Identifies indices of events in the test window.

**`_precompute_integral(n)`**
Samples n aftershock times from the Omori-Utsu distribution, sorts them, and numerically integrates the time decay kernel cumulatively. Returns `(time_mesh, integral_values)` arrays for fast interpolation.

**`integral_time_decay(t_values)`**
Fast lookup of `∫_0^t g(s) ds` by interpolating the precomputed mesh.

**`Lambda()`**
Computes `∫_{t_{i-1}}^{t_i} λ*(s) ds` for each event i in the test window (the compensator). Uses joblib parallelism. The integral has two components:
- Background: `μ * area * (t_i - t_aux_start)`
- Triggering: sum over all prior events of κ(m_j) * spatial_integral(m_j) * ∫g(t_i - t_j) dt

**`lambd()`**
Computes `λ*(t_i, x_i, y_i | H_{t_i})` — the conditional intensity at each test event's location and time. Used for the spatial log-likelihood.

**`lambd_star()`**
Computes `λ*(t_i | H_{t_i})` = intensity integrated over all space at time t_i. Used for temporal log-likelihood.

**`evaluate()`**
Computes:
- `LL = log(λ*(t,x,y)) - Λ` (full log-likelihood per event)
- `TLL = log(λ*(t)) - Λ` (temporal log-likelihood per event)
- `SLL = LL - TLL` (spatial log-likelihood per event)
Returns mean NLL, TLL, SLL over test events as a dict.

**`evaluate_baseline_poisson_model()`**
Fits a stationary Poisson process to the training period and evaluates it on the test period. Computes NLL, TLL, SLL for the Poisson baseline as a comparison point.

**`store_results(data_path)`**
Saves `augmented_catalog.csv` (catalog with per-event LL, TLL, SLL, lambd, lambd_star, int_lambd columns) and `ll_scores.json` (ETAS vs Poisson scores).

---

### `etas/plots.py`

Visualisation of the ETAS model fit by comparing the inverted kernels against the empirical branching structure encoded in P_ij.

#### Standalone Functions

**`time_scaling_factor(c, tau, omega, t0, t1)`**
Computes the integral of the time kernel over [10^t0, 10^t1] days. Used as a normalisation constant so the kernel curve and the data histogram are on the same scale.

**`temporal_decay_plot(p_mat, tau, c, omega, label, comparison_params, file_name)`**
Creates a log-log plot of:
- Scatter: weighted histogram of P_ij time distances (empirical)
- Line: fitted tapered Omori-Utsu kernel
- Dashed vertical lines at τ and c
Optionally overlays additional parameter sets from `comparison_params` dict.

**`productivity_plot(p_mat, catalog, params, mc, delta_m, label, comparison_params, file_name)`**
Creates a log-linear plot of:
- Scatter: mean P_ij weight per source magnitude bin (empirical aftershock number)
- Line: fitted productivity law `k0 * exp(a * (m - mc))`

**`spatial_decay_plot(p_mat, magnitudes, d, gamma, rho, mc, space_unit_in_meters, label, comparison_params, file_name)`**
Creates one log-log plot per magnitude in `magnitudes` of:
- Scatter: weighted histogram of P_ij spatial distances
- Line: fitted spatial kernel
Filenames have `_mag_X.XX.pdf` appended.

#### Class: `ETASFitVisualisation`

Orchestrates all three plots from a config dict:

**`__init__(metadata)`**
Loads catalog CSV, Pij CSV, parameters dict, mc/delta_m. Converts comparison parameter sets to the same reference magnitude and space unit as the main parameters.

**`time_kernel_plot(fn_store='time_kernel_fit.pdf')`** → calls `temporal_decay_plot`
**`productivity_law_plot(fn_store='productivity_law_fit.pdf')`** → calls `productivity_plot`
**`space_kernel_plot(fn_store='space_kernel_fit.pdf')`** → calls `spatial_decay_plot`
**`all_plots()`** → calls all three

---

### `etas/download.py`

**`download_catalog_sed(starttime, endtime, minmagnitude=0.01, delta_m=0.1)`**
Downloads earthquake catalog from the Swiss Seismological Service (SED) FDSN web service at `http://eida.ethz.ch/fdsnws/event/1/query`. Parses pipe-delimited CSV response. Returns a pandas DataFrame with columns: `magnitude, latitude, longitude, time, depth`. Sorted by time.

---

### `etas/oef/`

Provides standardized HERMES model interfaces for Operational Earthquake Forecasting (OEF).

#### `etas/oef/__init__.py`
Exports `entrypoint_suiETAS` and `entrypoint_europe` for import by HERMES.

#### `etas/oef/entrypoint.py`

**`entrypoint_suiETAS(model_input: ModelInput) -> list[ForecastCatalog]`**
Full pipeline for Swiss ETAS (suiETAS):
1. Parses QuakeML seismicity data into a `seismostats.Catalog`.
2. Extracts polygon from WKT bounding polygon string.
3. Runs ETAS inversion (`ETASParameterCalculation.prepare()` + `.invert()`).
4. Loads background rate grid from `etas/oef/data/SUIhaz2015_rates.csv`.
5. Sets up `ETASSimulation` with the Swiss background grid.
6. Runs `simulate_to_df(forecast_duration_days, n_simulations)`.
7. Returns a `ForecastCatalog` with metadata (starttime, endtime, bounding_polygon, etc.).

**`entrypoint_europe(model_input: ModelInput) -> list[ForecastCatalog]`**
European ETAS — skips inversion (uses pre-inverted parameters from `model_parameters`). Loads background rates from `etas/oef/data/europe_rate_map.csv`. Otherwise same simulation logic as suiETAS.

Both entrypoints are decorated with `@validate_entrypoint(induced=False)` from the `hermes_model` package.

---

## Runnable Scripts: `runnable_code/`

These are the main scripts for running the pipeline. Each reads a JSON config file and calls the core library.

### `runnable_code/invert_etas.py`

Runs ETAS parameter inversion. Reads `./config/invert_etas_config.json`. Steps:
1. `ETASParameterCalculation(inversion_config)` — initialise
2. `.prepare()` — filter, compute distances, estimate β
3. `.invert()` — EM loop
4. `.store_results(data_path, store_pij=True)` — save outputs

Output: `output_data/parameters_0.json`, `src_0.csv`, `ip_0.csv`, `dist_0.csv`, optionally `pij_0.csv`.

### `runnable_code/forecast_catalog_continuation.py`

Runs Monte Carlo forecast using previously inverted parameters. Reads `./config/forecast_catalog_continuation_config.json`. Steps:
1. Loads `parameters_0.json` with `ETASParameterCalculation.load_calculation()`
2. `ETASSimulation(etas_invert, m_max=m_max)` + `.prepare()`
3. Iterates `simulation.simulate(forecast_duration, n_simulations)` generator, accumulates chunks
4. Saves all events to `./output_data/forecast_catalog_continuation.csv`

### `runnable_code/simulate_catalog_continuation.py`

Similar to above but uses `ETASSimulation.simulate_to_csv()` for streaming output to CSV.

### `runnable_code/simulate_catalog.py`

Generates a fully synthetic catalog (no conditioning) using `generate_catalog()` directly. Saves to CSV.

### `runnable_code/predict_etas.py`

Evaluates model performance. Reads `invert_etas_config.json` and the output `parameters_0.json`. Steps:
1. `ETASLikelihoodCalculation(inversion_output)` — load parameters
2. `.prepare(n=1000000)` — precompute time integral with 1M samples
3. `.evaluate_baseline_poisson_model()` — compute Poisson baseline scores
4. `.evaluate()` — compute ETAS NLL, TLL, SLL
5. `.store_results(data_path)` — save augmented catalog and scores

### `runnable_code/visualise_fit.py`

Generates kernel fit plots. Reads `./config/visualisation_config.json`. Calls `ETASFitVisualisation(config).all_plots()`.

### `runnable_code/estimate_mc.py`

Demonstration of Mc and β estimation. Creates a sample magnitude array, calls `estimate_mc()` with a range of candidate Mc values, prints results.

### `runnable_code/reload_example.py`

Shows how to reload a completed inversion from JSON using `ETASParameterCalculation.load_calculation()`. Useful for post-hoc analysis without re-running inversion.

### `runnable_code/ch_forecast.py`

Swiss forecast pipeline combining inversion + simulation in one script. Reads `ch_forecast_config.json`. Demonstrates using a hazard grid (SUIhaz2015_rates.csv) for background event locations.

---

## Top-Level Scripts

### `conditional_intensity.py`

Evaluates the ETAS conditional intensity λ*(x, y, t | H_t) on a 0.1° grid for visualisation as a prior distribution map.

1. Loads inverted parameters from `./output_data/parameters_0.json`.
2. Reads the earthquake catalog.
3. Sets up a lat/lon grid over the polygon bounds at 0.1° spacing.
4. Calls `conditional_intensity_grid()` which vectorises `triggering_kernel()` using broadcasting over `(n_grid, n_events)` arrays.
5. Plots using Cartopy with log-normalised colour scale.

The key function `conditional_intensity_grid(forecast_time, grid_lats, grid_lons, past_catalog, theta, mc)`:
- Computes `dt` = days since each past event
- Computes `r_sq` = squared distances from all past events to all grid points (shape `(n_grid, n_events)`)
- Evaluates `triggering_kernel` for all (grid, event) pairs
- Returns `μ + Σ_events gij` for each grid point

Note: this script ignores `xi_plus_1` (the incompleteness correction) since it's for visualisation rather than rigorous forecasting.

### `plot_simulated_catalog_continuation.py`

Cartopy-based map of simulated earthquake catalogs. Reads a CSV of simulated events. Plots each event as a scatter point with:
- Marker size ∝ magnitude
- Colour = time (using a colormap)

Overlays state/coastline boundaries. Uses `ccrs.PlateCarree()` projection.

### `test_concept.py`

A minimal stand-alone ETAS implementation for conceptual understanding. Demonstrates the basic Omori-Utsu law and background seismicity without the full machinery of the main library.

---

## Configuration Files: `config/`

### `invert_etas_config.json`

Example configuration for California data:

```json
{
  "fn_catalog": "./input_data/example_catalog.csv",
  "data_path": "./output_data/",
  "auxiliary_start": "1971-01-01 00:00:00",
  "timewindow_start": "1981-01-01 00:00:00",
  "timewindow_end": "2007-01-01 00:00:00",
  "testwindow_end": "2021-01-01 00:00:00",
  "theta_0": { ... initial parameter guess ... },
  "free_background": true,
  "bw_sq": 4,
  "mc": 3.6,
  "delta_m": 0.1,
  "coppersmith_multiplier": 100,
  "shape_coords": "./input_data/california_shape.npy",
  "id": "0"
}
```

The auxiliary period (1971–1981) provides burn-in: events in this window act as sources but not targets, allowing their aftershocks to "warm up" the model before the training period.

### `forecast_catalog_continuation_config.json`

```json
{
  "forecast_duration": 365,
  "fn_store_simulation": "./output_data/forecast_catalog_continuation.csv",
  "fn_inversion_output": "./output_data/parameters_0.json",
  "n_simulations": 100
}
```

### Other configs

- `invert_etas_config.jsonc` — same as above but with inline comments explaining each field.
- `simulate_catalog_continuation_config.json` — parameters for single simulation.
- `ch_forecast_config.json` — Swiss full-pipeline configuration.
- `visualisation_config.json` — paths to catalog/Pij CSVs for plot generation.

---

## Data Flow: End-to-End Pipeline

```
Input catalog (CSV: id, latitude, longitude, time, magnitude)
         │
         ▼
ETASParameterCalculation.__init__(config)
  │  parse config, set time windows and mc
  │
  ▼
.prepare()
  │  filter_catalog() → time/space/magnitude filter
  │  calculate_distances() → pairwise P_ij initialisation (Coppersmith cutoff)
  │  prepare_source_events() / prepare_target_events()
  │  estimate β (Tinti or b-positive)
  │
  ▼
.invert()           ← EM loop
  │  expectation_step(θ)  → P_ij, P_background
  │  optimize_parameters() → new θ
  │  check convergence
  │  (repeat)
  │
  ▼
.store_results()
  │  parameters_N.json (all metadata + final θ)
  │  src_N.csv, ip_N.csv, dist_N.csv, pij_N.csv
  │
  ▼
ETASParameterCalculation.load_calculation(parameters_N.json)
  │
  ▼
ETASSimulation(inversion_params)
  │
  ▼
.prepare()
  │  extract source events + background probabilities
  │
  ▼
.simulate(forecast_n_days, n_simulations)   ← generator
  │  for each simulation:
  │    simulate_catalog_continuation(...)
  │      │  generate_background_events()   → Poisson(μ*A*T) background events
  │      │  prepare_auxiliary_catalog()    → condition on past events
  │      │  generational loop:
  │      │    generate_aftershocks()
  │      │      │  simulate_aftershock_time()   (Omori-Utsu CDF inversion)
  │      │      │  simulate_aftershock_radius() (power-law CDF inversion)
  │      │      └  simulate_magnitudes()        (G-R CDF inversion)
  │      └  filter to polygon and time window
  │  yield chunk of simulations (DataFrame)
  │
  ▼
CSV / ForecastCatalog
```

---

## Key Data Structures

### Catalog DataFrame

Standard columns (required for inversion):
| Column | Type | Description |
|--------|------|-------------|
| `id` | str/int | Unique event identifier (index) |
| `latitude` | float | Degrees N |
| `longitude` | float | Degrees E |
| `time` | datetime | UTC event time |
| `magnitude` | float | Moment/local magnitude |
| `mc_current` | float | Completeness magnitude at event (required if `mc='var'`) |

After `filter_catalog`:
- `mc_current` column is added (constant or variable)

After `expectation_step`:
- `P_background` — probability event is background (spontaneous)
- `zeta_plus_1` — observation factor for completeness correction
- `xi_plus_1` — responsibility factor

### Distance/P_ij DataFrame

Long-format DataFrame indexed by `(source_id, target_id)`:
| Column | Description |
|--------|-------------|
| `time_distance` | Δt in days (positive, source before target) |
| `spatial_distance_squared` | r² in km² |
| `source_magnitude` | m of triggering event |
| `target_time` | datetime of triggered event |
| `Pij` | probability source triggered target |
| `zeta_plus_1` | observation correction factor |
| `xi_plus_1` | responsibility correction factor |

### Parameters Dict (θ)

10 named parameters:
| Key | Physical meaning | Typical range |
|-----|-----------------|---------------|
| `log10_mu` | log₁₀(background rate, events/day/km²) | -8 to -3 |
| `log10_iota` | (auxiliary, rarely used) | — |
| `log10_k0` | log₁₀(productivity prefactor) | -5 to 0 |
| `a` | productivity magnitude scaling | 0.5 to 3 |
| `log10_c` | log₁₀(Omori c, days) | -5 to -1 |
| `omega` | Omori decay shape = p - 1 | -0.5 to 0.5 |
| `log10_tau` | log₁₀(taper timescale, days) | 1 to 8 |
| `log10_d` | log₁₀(spatial scale, km) | -2 to 1 |
| `gamma` | spatial magnitude scaling | 0.5 to 2 |
| `rho` | spatial decay exponent | 0.1 to 2 |

Derived: `alpha = a - rho * gamma` is the net magnitude scaling of the total productivity integrated over space.

---

## Parameter Reference

### Relationship between formulations

The productivity in this code uses:
`κ(m) = k0 * exp(a * (m - mc))`

The standard ETAS formulation uses:
`κ(m) = K * 10^{α(m - mc)}`

Conversion: `k0 = K * π/ρ * d^{-ρ}`, `a = α * ln(10)`.

### Temporal kernel normalisation

The time kernel `g(t) = exp(-t/τ) / (t + c)^{1+ω}` is not normalised by default. The normalisation constant is `exp(c/τ) * τ^{-ω} * Γ(-ω, c/τ)` where Γ is the upper incomplete gamma function.

---

## Tests

### `tests/test_simulation.py`

Tests for the simulation module's sampling functions. Covers:
- `inv_time_cdf_approx` — checks the approximate CDF inverse gives physically reasonable values and matches the exact sampler distribution.
- `simulate_aftershock_time` — verifies sampled times are non-negative.
- `simulate_aftershock_place` / `simulate_aftershock_radius` — checks sampled locations/radii are real and non-negative.
- `inverse_upper_gamma_ext` — checks the inverse correctly inverts `upper_gamma_ext` for both positive and negative `a`.
- Property-based tests using hypothesis (if installed) to check statistical properties of magnitude simulation.

Run with: `pytest tests/`
