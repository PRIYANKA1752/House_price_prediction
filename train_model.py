"""
Train Real Estate House Price Prediction Models
Trains multiple ML models, evaluates them, and saves the best one.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "real_estate.csv")
MODEL_DIR   = "models"
STATIC_DIR  = "static"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Load & Clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [
    "No", "transaction_date", "house_age",
    "distance_to_mrt", "convenience_stores",
    "latitude", "longitude", "price_per_unit_area"
]
df.drop(columns=["No"], inplace=True)

print(f"Dataset shape: {df.shape}")
print(df.describe())

# ── Feature Engineering ───────────────────────────────────────────────────────
df["distance_to_mrt_log"] = np.log1p(df["distance_to_mrt"])
df["transaction_year"]    = df["transaction_date"].astype(int)
df["transaction_month"]   = ((df["transaction_date"] % 1) * 12).round().astype(int).clip(1, 12)

FEATURES = [
    "transaction_date", "house_age",
    "distance_to_mrt_log", "convenience_stores",
    "latitude", "longitude",
    "transaction_year", "transaction_month"
]
TARGET = "price_per_unit_area"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define Models ─────────────────────────────────────────────────────────────
models = {
    "Linear Regression":        LinearRegression(),
    "Ridge Regression":         Ridge(alpha=1.0),
    "Random Forest":            RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name in ("Linear Regression", "Ridge Regression") else X_train
    Xte = X_test_sc  if name in ("Linear Regression", "Ridge Regression") else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "preds": preds}
    print(f"{name:28s} | MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# ── Pick Best Model ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["R2"])
best_info = results[best_name]
print(f"\nBest model: {best_name}  (R²={best_info['R2']:.4f})")

best_model  = best_info["model"]
best_Xtr    = X_train_sc if best_name in ("Linear Regression", "Ridge Regression") else X_train
best_Xte    = X_test_sc  if best_name in ("Linear Regression", "Ridge Regression") else X_test

# ── Save Artefacts ────────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES,   os.path.join(MODEL_DIR, "features.pkl"))
joblib.dump(best_name,  os.path.join(MODEL_DIR, "best_model_name.pkl"))
joblib.dump(results,    os.path.join(MODEL_DIR, "all_results.pkl"))
print("Model artefacts saved to models/")

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ["MAE", "RMSE", "R2"]
labels  = list(results.keys())
colors  = ["#3b82d4", "#7c5cd8", "#22c55e", "#f59e0b"]
for i, metric in enumerate(metrics):
    vals = [results[n][metric] for n in labels]
    x = range(len(labels))
    axes[i].bar(x, vals, color=colors)
    axes[i].set_title(metric, fontsize=13)
    axes[i].set_xticks(list(x))
    axes[i].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[i].set_ylabel(metric)
plt.suptitle("Model Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=120)
plt.close()

# 2. Actual vs Predicted
preds_best = best_info["preds"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, preds_best, alpha=0.6, color="#3b82d4", edgecolors="white", linewidths=0.4)
mn, mx = min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual Price (per unit area)")
ax.set_ylabel("Predicted Price (per unit area)")
ax.set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close()

# 3. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

# 4. Feature importances (if tree-based)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="#3b82d4")
    ax.set_title(f"Feature Importances — {best_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importance.png"), dpi=120)
    plt.close()

print("Charts saved to static/")
print("\nTraining complete.")
