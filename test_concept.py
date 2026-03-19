#%%
import numpy as np
import matplotlib.pyplot as plt

def run_etas(mu, A, alpha, c, p, mc, T, seed=42):
    """
    Minimal ETAS simulation over time horizon T (days).
    Background seismicity is Poisson(mu*T). Each earthquake
    triggers aftershocks according to the Omori-Utsu law.
    """
    np.random.seed(seed)

    # --- background events (Poisson process) ---
    n_background = np.random.poisson(mu * T)
    times = list(np.random.uniform(0, T, n_background))
    mags  = list(mc - np.log(np.random.uniform(size=n_background)) / 2.3)

    # process queue — each event may trigger children
    i = 0
    while i < len(times):
        t_i, m_i = times[i], mags[i]

        # expected number of aftershocks from this event
        n_aftershocks = np.random.poisson(A * np.exp(alpha * (m_i - mc)))

        for _ in range(n_aftershocks):
            # Omori-Utsu: sample delay time from power-law decay
            u = np.random.uniform()
            dt = c * (np.power(u, -1/(p-1)) - 1)   # inverse CDF of Omori
            t_child = t_i + dt

            if t_child < T:
                times.append(t_child)
                mags.append(mc - np.log(np.random.uniform()) / 2.3)

        i += 1

    return np.array(times), np.array(mags)

#%%
# --- run it ---
times, mags = run_etas(
    mu=0.1,    # background rate (events/day)
    A=0.8,     # aftershock productivity
    alpha=1.5, # magnitude scaling
    c=0.1,    # Omori c (days)
    p=1.1,     # Omori p > 1
    mc=3.0,    # magnitude of completeness
    T=365,     # 1 year
)

order = np.argsort(times)
times, mags = times[order], mags[order]
print(f"Total events: {len(times)}")


# %%
