#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#%%
 
# ── Data ──────────────────────────────────────────────────────────────────────
#file_path = "./output_data/simulated_catalog_continuation.csv"
file_path = "./output_data/forecast_catalog_continuation.csv"
 
df = pd.read_csv(file_path, header=0, index_col=None, delimiter=",",parse_dates=['time'])
 
# ── Encoding ──────────────────────────────────────────────────────────────────
# Size: scale magnitude to marker area
size = ((df["magnitude"] - df["magnitude"].min() + 0.1) * 10) ** 1.3
 
# Color: map time to numeric for colormap
t_num = df["time"].astype(np.int64)
norm = mcolors.Normalize(vmin=t_num.min(), vmax=t_num.max())
cmap = plt.cm.plasma
 
# ── Map extent with padding ───────────────────────────────────────────────────
pad = 2.0
extent = [
    df["longitude"].min() - pad,
    df["longitude"].max() + pad,
    df["latitude"].min() - pad,
    df["latitude"].max() + pad,
]

#%%
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(9, 7),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
ax.set_extent(extent, crs=ccrs.PlateCarree())
 
# Base map features
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
 
# Scatter
sc = ax.scatter(
    df["longitude"],
    df["latitude"],
    s=size,
    c=t_num,
    cmap=cmap,
    norm=norm,
    edgecolors="black",
    linewidths=0.6,
    transform=ccrs.PlateCarree(),
    zorder=3,
)
 
# ── Colourbar (time) ──────────────────────────────────────────────────────────
cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Time", fontsize=10)
tick_vals = np.linspace(t_num.min(), t_num.max(), 5)
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(
    pd.to_datetime(tick_vals).strftime("%Y-%m-%d"), fontsize=7
)
 
# ── Size legend (magnitude) ───────────────────────────────────────────────────
mag_min = np.floor(df["magnitude"].min())
mag_max = np.ceil(df["magnitude"].max())
mag_levels = np.arange(mag_min, mag_max + 1)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D(
        [], [],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=np.sqrt(((m - df["magnitude"].min() + 1) * 60) ** 1.3 / np.pi) * 0.6,
        label=f"M {m:.0f}",
    )
    for m in mag_levels
]
ax.legend(handles=legend_handles, title="Magnitude", loc="lower right", fontsize=8)
 
ax.set_title("Earthquakes from ETAS forecast (100 simulations)", fontsize=12)
plt.tight_layout()
#plt.savefig("/mnt/user-data/outputs/earthquakes_map.png", dpi=150, bbox_inches="tight")
plt.show()
#print("Saved: earthquakes_map.png")
 
# %%
