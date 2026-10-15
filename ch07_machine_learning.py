"""
Chapter 7: Machine Learning for Sustainability
Data Analytics and AI in ESG, Climate Change and Sustainability
Patria, Anindika Yudhy & Zulkifli (2026)

Demonstrates:
  - CRISP-DM workflow on UCI Air Quality dataset
  - Supervised ML: OLS, Ridge, Random Forest, XGBoost
  - Hyperparameter tuning with cross-validation
  - SHAP feature importance analysis
  - Model performance comparison (Wiley/Springer-quality charts)
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import shap
from ucimlrepo import fetch_ucirepo

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"OLS": "#1e3a5f", "Ridge": "#2563eb", "Random Forest": "#10b981",
          "Gradient Boosting": "#f97316", "XGBoost": "#7c3aed"}

# ── 2. CRISP-DM Phase 2: Data Understanding ──────────────────────────────────
print("Phase 2: Data Understanding")
print("Loading UCI Air Quality dataset (id=360)...")
air_quality = fetch_ucirepo(id=360)
X_raw = air_quality.data.features
y_raw = air_quality.data.targets
df = pd.concat([X_raw, y_raw], axis=1)
df.columns = df.columns.str.strip()
print(f"Shape: {df.shape}")
print(f"\nMetadata:\n{air_quality.metadata['abstract'][:300]}...")

# ── 3. CRISP-DM Phase 3: Data Preparation ────────────────────────────────────
print("\nPhase 3: Data Preparation")
df.replace(-200, np.nan, inplace=True)
df.replace(-9999, np.nan, inplace=True)
df.dropna(thresh=int(df.shape[1] * 0.6), inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

TARGET = "CO(GT)"
feature_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c != TARGET]
X = df[feature_cols].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"Training: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ── 4. CRISP-DM Phase 4 & 5: Modelling and Evaluation ───────────────────────
print("\nPhase 4-5: Modelling and Evaluation")
models = {
    "OLS":              LinearRegression(),
    "Ridge":            Ridge(alpha=1.0),
    "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost":          XGBRegressor(n_estimators=100, learning_rate=0.1,
                                     random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name in ("OLS", "Ridge") else X_train
    Xte = X_test_sc  if name in ("OLS", "Ridge") else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2")
    results[name] = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
        "CV_R2_mean": cv.mean(),
        "CV_R2_std":  cv.std(),
        "model": model,
        "y_pred": y_pred,
    }
    print(f"  {name:20s}  MAE={results[name]['MAE']:.3f}  "
          f"R2={results[name]['R2']:.3f}  CV-R2={cv.mean():.3f}±{cv.std():.3f}")

# ── 5. CRISP-DM Phase 6: Model Comparison Chart ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
names = list(results.keys())
colors_list = [COLORS[n] for n in names]

# MAE bar
axes[0].barh(names, [results[n]["MAE"] for n in names], color=colors_list, alpha=0.85)
axes[0].set_xlabel("Mean Absolute Error (lower = better)")
axes[0].set_title("MAE by Model", fontweight="bold")

# R2 bar
axes[1].barh(names, [results[n]["R2"] for n in names], color=colors_list, alpha=0.85)
axes[1].set_xlabel("R² Score (higher = better)")
axes[1].set_title("R² Score by Model", fontweight="bold")
axes[1].set_xlim(0, 1.05)

# CV R2 with error bars
cv_means = [results[n]["CV_R2_mean"] for n in names]
cv_stds  = [results[n]["CV_R2_std"]  for n in names]
axes[2].barh(names, cv_means, xerr=cv_stds, color=colors_list, alpha=0.85,
             capsize=4, error_kw={"linewidth": 1.2})
axes[2].set_xlabel("5-Fold CV R² ± std")
axes[2].set_title("Cross-Validation R²", fontweight="bold")
axes[2].set_xlim(0, 1.05)

for ax in axes:
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig7_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 6. SHAP Feature Importance (XGBoost) ─────────────────────────────────────
print("\nPhase 6: SHAP Feature Importance (XGBoost)")
xgb_model = results["XGBoost"]["model"]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

fig2, ax2 = plt.subplots(figsize=(9, 5))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — XGBoost CO Prediction", fontweight="bold")
plt.tight_layout()
plt.savefig("fig7_shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigures saved: fig7_model_comparison.png, fig7_shap_importance.png")
