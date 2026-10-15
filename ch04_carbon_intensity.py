"""
Chapter 4: Carbon Intensity Accounting and Net Zero Pathway Analysis
Data Analytics and AI in ESG, Climate Change and Sustainability
Patria, Anindika Yudhy & Zulkifli (2026)

Demonstrates:
  - Loading UCI Air Quality dataset via ucimlrepo
  - Feature engineering for carbon intensity modelling
  - CRISP-DM Phase 3 data preparation
  - Scope 1/2/3 emissions inventory (PT SIN illustrative case)
  - Net zero pathway projection
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from ucimlrepo import fetch_ucirepo

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#1e3a5f", "#2563eb", "#10b981", "#f97316", "#7c3aed"]

# ── 2. Load UCI Air Quality Dataset (id=360) ─────────────────────────────────
print("Loading UCI Air Quality dataset...")
air_quality = fetch_ucirepo(id=360)
X = air_quality.data.features
y = air_quality.data.targets
df = pd.concat([X, y], axis=1)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 3. Data Cleaning (CRISP-DM Phase 3) ──────────────────────────────────────
df.columns = df.columns.str.strip()
df.replace(-200, np.nan, inplace=True)
df.replace(-9999, np.nan, inplace=True)
df.dropna(thresh=int(df.shape[1] * 0.6), inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"\nAfter cleaning: {df.shape[0]} rows, {df.sum().isna().sum()} missing values")

# ── 4. Feature Engineering for Carbon Intensity ───────────────────────────────
# Carbon intensity proxy: CO concentration normalised to NOx ratio
if "CO(GT)" in df.columns and "NOx(GT)" in df.columns:
    df["CO_NOx_ratio"] = df["CO(GT)"] / (df["NOx(GT)"].replace(0, np.nan) + 1)
    df["CO_rolling_3h"] = df["CO(GT)"].rolling(window=3, min_periods=1).mean()
    df["CO_rolling_24h"] = df["CO(GT)"].rolling(window=24, min_periods=1).mean()

# ── 5. Regression: Predict CO concentration ──────────────────────────────────
feature_cols = [c for c in df.columns if c not in ["CO(GT)", "Date", "Time"]
                and df[c].dtype in [np.float64, np.int64]]
target_col = "CO(GT)"

if target_col in df.columns and len(feature_cols) >= 3:
    X_feat = df[feature_cols].fillna(0)
    y_target = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for name, model in [("OLS Linear", LinearRegression()), ("Ridge", Ridge(alpha=1.0))]:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        print(f"{name}: MAE={mean_absolute_error(y_test, y_pred):.3f}, R2={r2_score(y_test, y_pred):.3f}")

# ── 6. GHG Inventory: PT SIN Illustrative Case ───────────────────────────────
years = list(range(2020, 2036))
scope1 = [42000 * (1 - 0.042) ** (y - 2020) for y in years]
scope2 = [18000 * (1 - 0.065) ** (y - 2020) for y in years]
scope3 = [95000 * (1 - 0.035) ** (y - 2020) for y in years]
total = [s1 + s2 + s3 for s1, s2, s3 in zip(scope1, scope2, scope3)]
sbti_benchmark = [155000 * (1 - 0.042) ** (y - 2020) for y in years]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked area chart
ax = axes[0]
ax.stackplot(years, scope1, scope2, scope3,
             labels=["Scope 1", "Scope 2", "Scope 3"],
             colors=[COLORS[0], COLORS[2], COLORS[3]], alpha=0.75)
ax.set_title("PT SIN GHG Inventory 2020–2035", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("tCO2e")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.legend(loc="upper right", fontsize=9)

# Net zero pathway
ax2 = axes[1]
ax2.plot(years, [t / 1000 for t in total], color=COLORS[0], lw=2, label="PT SIN total")
ax2.plot(years, [s / 1000 for s in sbti_benchmark], color=COLORS[3], lw=2,
         linestyle="--", label="SBTi 1.5°C benchmark")
ax2.fill_between(years,
                 [t / 1000 for t in total],
                 [s / 1000 for s in sbti_benchmark],
                 alpha=0.12, color=COLORS[3], label="Gap to benchmark")
ax2.set_title("Net Zero Pathway: PT SIN vs. SBTi Benchmark", fontweight="bold")
ax2.set_xlabel("Year")
ax2.set_ylabel("ktCO2e")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig4_ghg_inventory.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved: fig4_ghg_inventory.png")
