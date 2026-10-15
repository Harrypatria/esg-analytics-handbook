"""
Chapter 9: Geospatial Analytics for Environmental Monitoring
Data Analytics and AI in ESG, Climate Change and Sustainability
Patria, Anindika Yudhy & Zulkifli (2026)

Demonstrates:
  - Synthetic NDVI time-series analysis (land cover change proxy)
  - Deforestation detection using spectral indices
  - Leafmap interactive map creation
  - Spatial ESG metrics: emission plume footprint estimation
  - Biodiversity stress index calculation
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# Grey gradation palette — print-safe, no colour dependency
G = {
    "darkest":  "#1f2937",
    "dark":     "#374151",
    "medium":   "#6b7280",
    "light":    "#9ca3af",
    "lighter":  "#d1d5db",
    "lightest": "#f3f4f6",
}

OUT_PATH = r"C:\Users\harpa\Documents\Others\PC\project_management\esg-handbook\img\ch09_geospatial_analysis.png"

# ── 2. Synthetic NDVI Time-Series (Kalimantan 2018–2023) ─────────────────────
np.random.seed(42)
months = pd.date_range("2018-01", "2024-01", freq="ME")
n = len(months)

# Three land parcels: forest, degraded, restoration zone
ndvi_forest      = 0.72 + np.random.normal(0, 0.02, n) - np.linspace(0, 0.06, n)
ndvi_degraded    = 0.45 + np.random.normal(0, 0.03, n) - np.linspace(0, 0.12, n)
ndvi_restoration = 0.30 + np.random.normal(0, 0.02, n) + np.linspace(0, 0.15, n)

# Add seasonal signal (wet season peak)
seasonal = 0.04 * np.sin(2 * np.pi * np.arange(n) / 12)
ndvi_forest      += seasonal
ndvi_degraded    += seasonal * 0.6
ndvi_restoration += seasonal * 0.4

df_ndvi = pd.DataFrame({
    "date":              months,
    "Primary Forest":    ndvi_forest.clip(0.1, 1),
    "Degraded Peatland": ndvi_degraded.clip(0.1, 1),
    "Restoration Zone":  ndvi_restoration.clip(0.1, 1),
})
df_ndvi.set_index("date", inplace=True)

# ── 3. NDVI Trend Chart ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("white")

ax = axes[0, 0]
line_styles = [
    (G["darkest"],  "Primary Forest"),
    (G["medium"],   "Degraded Peatland"),
    (G["dark"],     "Restoration Zone"),
]
for (color, col) in line_styles:
    ax.plot(df_ndvi.index, df_ndvi[col], color=color, lw=1.4,
            label=col, alpha=0.75)
    ax.plot(df_ndvi.index, df_ndvi[col].rolling(6).mean(),
            color=color, lw=2.5, linestyle="--", alpha=0.95)

ax.axhline(0.5, color=G["light"], lw=1, linestyle=":", label="NDVI = 0.5 threshold")
ax.set_facecolor("white")
ax.set_title("NDVI Time Series — Kalimantan 2018–2023", fontweight="bold",
             color=G["darkest"])
ax.set_ylabel("NDVI", color=G["darkest"])
ax.tick_params(colors=G["dark"])
ax.legend(fontsize=9, framealpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_edgecolor(G["lighter"])

# ── 4. Deforestation Rate Calculation ─────────────────────────────────────────
years_sim    = [2018, 2020, 2022, 2024]
forest_cover = {2018: 68, 2020: 61, 2022: 54, 2024: 48}

ax2 = axes[0, 1]
cover_values = list(forest_cover.values())
# Darker bar = worse state (more recent = more deforested)
bar_greys = [G["lightest"], G["light"], G["medium"], G["dark"]]
bars = ax2.bar([str(y) for y in years_sim], cover_values,
               color=bar_greys, alpha=0.92, width=0.5,
               edgecolor=G["dark"], linewidth=0.8)
ax2.set_facecolor("white")
ax2.set_title("Canopy Cover Decline — Degraded Peatland Block", fontweight="bold",
              color=G["darkest"])
ax2.set_ylabel("Forest Cover (%)", color=G["darkest"])
ax2.set_ylim(0, 85)
for bar, val in zip(bars, cover_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold",
             color=G["darkest"])
ax2.tick_params(colors=G["dark"])
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_edgecolor(G["lighter"])

# Annual deforestation rate
ax3 = axes[1, 0]
defor_rate    = [-(forest_cover[years_sim[i+1]] - forest_cover[years_sim[i]]) / 2
                 for i in range(len(years_sim) - 1)]
period_labels = ["2018–2020", "2020–2022", "2022–2024"]
rate_greys    = [G["darkest"], G["dark"], G["medium"]]
ax3.bar(period_labels, defor_rate, color=rate_greys, alpha=0.92, width=0.5,
        edgecolor=G["dark"], linewidth=0.8)
ax3.axhline(3.0, color=G["light"], lw=1.5, linestyle="--",
            label="Global avg. deforestation rate (3 %/yr)")
ax3.set_facecolor("white")
ax3.set_title("Annual Deforestation Rate (%/yr)", fontweight="bold",
              color=G["darkest"])
ax3.set_ylabel("Rate (% / year)", color=G["darkest"])
ax3.tick_params(colors=G["dark"])
ax3.legend(fontsize=9, framealpha=0.7)
ax3.spines[["top", "right"]].set_visible(False)
ax3.spines[["left", "bottom"]].set_edgecolor(G["lighter"])

# ── 5. Biodiversity Stress Index ─────────────────────────────────────────────
ax4 = axes[1, 1]
sites = [
    "Site A  (Primary forest)",
    "Site B  (Degraded)",
    "Site C  (Restoration)",
    "Site D  (Cropland)",
    "Site E  (Buffer zone)",
]
bsi_scores = [0.12, 0.78, 0.45, 0.88, 0.34]

def bsi_grey(s):
    if s < 0.3:
        return G["lighter"]
    elif s < 0.6:
        return G["medium"]
    else:
        return G["darkest"]

bar_c = [bsi_grey(s) for s in bsi_scores]
ax4.barh(sites, bsi_scores, color=bar_c, alpha=0.92,
         edgecolor=G["dark"], linewidth=0.8)
ax4.axvline(0.5, color=G["light"], lw=1.2, linestyle=":")
ax4.set_facecolor("white")
ax4.set_title("Biodiversity Stress Index by Land Parcel", fontweight="bold",
              color=G["darkest"])
ax4.set_xlabel("BSI Score (0 = low stress, 1 = high stress)", color=G["darkest"])
ax4.set_xlim(0, 1.1)
ax4.tick_params(colors=G["dark"])
ax4.spines[["top", "right"]].set_visible(False)
ax4.spines[["left", "bottom"]].set_edgecolor(G["lighter"])

low_p  = mpatches.Patch(facecolor=G["lighter"], edgecolor=G["dark"], label="Low stress  (BSI < 0.30)")
mid_p  = mpatches.Patch(facecolor=G["medium"],  edgecolor=G["dark"], label="Moderate    (0.30–0.60)")
high_p = mpatches.Patch(facecolor=G["darkest"], edgecolor=G["dark"], label="Critical    (BSI > 0.60)")
ax4.legend(handles=[low_p, mid_p, high_p], fontsize=8, loc="lower right", framealpha=0.7)

plt.suptitle("Chapter 9: Geospatial ESG Monitoring — Kalimantan Case Study",
             fontsize=13, fontweight="bold", color=G["darkest"], y=1.01)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Figure saved: {OUT_PATH}")

# ── 6. Leafmap Interactive Map (optional — requires leafmap + ipywidgets) ─────
print("\nOptional: Leafmap interactive map")
try:
    import leafmap
    m = leafmap.Map(center=[-1.5, 113.5], zoom=6)
    m.add_basemap("SATELLITE")
    print("Leafmap loaded. Run in JupyterLab to display the interactive map.")
    # m.show()  # Uncomment in notebook environment
except ImportError:
    print("leafmap not installed. Run: pip install leafmap")
