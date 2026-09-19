"""
Real Estate House Price Prediction — Streamlit Frontend
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82d4 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1e3a5f, #3b82d4);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
DATA_PATH  = os.path.join("data", "real_estate.csv")
STATIC_DIR = "static"

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model      = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features   = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    all_res    = joblib.load(os.path.join(MODEL_DIR, "all_results.pkl"))
    return model, scaler, features, model_name, all_res

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [
        "No", "transaction_date", "house_age",
        "distance_to_mrt", "convenience_stores",
        "latitude", "longitude", "price_per_unit_area"
    ]
    df.drop(columns=["No"], inplace=True)
    df["distance_to_mrt_log"] = np.log1p(df["distance_to_mrt"])
    df["transaction_year"]    = df["transaction_date"].astype(int)
    df["transaction_month"]   = ((df["transaction_date"] % 1) * 12).round().astype(int).clip(1, 12)
    return df

# ── Check models exist ─────────────────────────────────────────────────────────
model_ready = all(
    os.path.exists(os.path.join(MODEL_DIR, f))
    for f in ["best_model.pkl", "scaler.pkl", "features.pkl",
              "best_model_name.pkl", "all_results.pkl"]
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.2rem;">🏠 Real Estate House Price Predictor</h1>
    <p style="margin:0.5rem 0 0; opacity:0.85; font-size:1.05rem;">
        Taiwan Real Estate Dataset — predict price per unit area using
        historical property & location data
    </p>
</div>
""", unsafe_allow_html=True)

# ── Train button (shown when models not yet trained) ──────────────────────────
if not model_ready:
    st.warning("⚠️ Models not trained yet. Click the button below to train all models.")
    if st.button("🚀 Train Models Now", type="primary"):
        with st.spinner("Training models — this takes ~30 seconds..."):
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "train_model.py"],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("✅ Training complete! Reload the page.")
            st.code(result.stdout, language="text")
        else:
            st.error("Training failed.")
            st.code(result.stderr, language="text")
    st.stop()

# ── Load data & models ─────────────────────────────────────────────────────────
model, scaler, features, model_name, all_results = load_artefacts()
df = load_data()

# ── Sidebar — prediction inputs ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Property Details")
    st.caption("Adjust sliders to describe the property you want to value.")

    transaction_date = st.slider(
        "📅 Transaction Date (Year.Month)",
        min_value=2012.0, max_value=2014.0, value=2013.5, step=0.083,
        help="Fractional year (e.g. 2013.5 ≈ July 2013)"
    )
    house_age = st.slider(
        "🏗️ House Age (years)",
        min_value=0.0, max_value=45.0, value=15.0, step=0.1
    )
    distance_to_mrt = st.slider(
        "🚇 Distance to Nearest MRT (m)",
        min_value=23.0, max_value=6500.0, value=500.0, step=10.0
    )
    convenience_stores = st.slider(
        "🏪 Convenience Stores Nearby",
        min_value=0, max_value=10, value=5
    )
    latitude = st.number_input(
        "🌐 Latitude",
        min_value=24.9, max_value=25.1, value=24.97, step=0.0001,
        format="%.5f"
    )
    longitude = st.number_input(
        "🌐 Longitude",
        min_value=121.4, max_value=121.7, value=121.54, step=0.0001,
        format="%.5f"
    )

    st.markdown("---")
    predict_btn = st.button("🏷️ Predict Price", type="primary", use_container_width=True)

# ── Build input frame ──────────────────────────────────────────────────────────
distance_to_mrt_log = np.log1p(distance_to_mrt)
transaction_year    = int(transaction_date)
transaction_month   = int(round((transaction_date % 1) * 12))
transaction_month   = max(1, min(12, transaction_month))

input_dict = {
    "transaction_date":     transaction_date,
    "house_age":            house_age,
    "distance_to_mrt_log":  distance_to_mrt_log,
    "convenience_stores":   convenience_stores,
    "latitude":             latitude,
    "longitude":            longitude,
    "transaction_year":     transaction_year,
    "transaction_month":    transaction_month,
}
input_df = pd.DataFrame([input_dict])[features]

# Scale only if linear model
use_scaler = model_name in ("Linear Regression", "Ridge Regression")
input_ready = scaler.transform(input_df) if use_scaler else input_df

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏷️ Prediction", "📊 Model Performance", "📈 Data Insights", "📋 Dataset"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📌 Property Summary")
        summary = pd.DataFrame({
            "Feature": [
                "Transaction Date", "House Age", "Distance to MRT",
                "Convenience Stores", "Latitude", "Longitude"
            ],
            "Value": [
                f"{transaction_date:.3f}",
                f"{house_age} yrs",
                f"{distance_to_mrt:.0f} m",
                str(convenience_stores),
                f"{latitude:.5f}",
                f"{longitude:.5f}"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.info(
            f"**Model in use:** {model_name}  \n"
            f"**Scaled input:** {'Yes' if use_scaler else 'No (tree-based)'}"
        )

    with col_result:
        st.subheader("💰 Predicted Price")
        if predict_btn:
            prediction = model.predict(input_ready)[0]
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:1rem; opacity:0.8; margin-bottom:0.5rem;">
                    Estimated Price per Unit Area
                </div>
                <div style="font-size:3rem; font-weight:800; letter-spacing:1px;">
                    {prediction:,.1f}
                </div>
                <div style="font-size:0.9rem; opacity:0.75; margin-top:0.4rem;">
                    10,000 NTD / Ping (Taiwan unit)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Comparable Properties")
            dist_thresh = 300
            comps = df[
                (df["distance_to_mrt"].between(distance_to_mrt - dist_thresh, distance_to_mrt + dist_thresh)) &
                (df["house_age"].between(max(0, house_age - 5), house_age + 5))
            ].copy()
            if len(comps) > 0:
                st.dataframe(
                    comps[["house_age", "distance_to_mrt", "convenience_stores",
                            "latitude", "longitude", "price_per_unit_area"]]
                    .head(8).reset_index(drop=True),
                    use_container_width=True
                )
                avg_comp = comps["price_per_unit_area"].mean()
                delta = prediction - avg_comp
                st.metric(
                    label="Avg price of comparable properties",
                    value=f"{avg_comp:.1f}",
                    delta=f"{delta:+.1f} vs prediction"
                )
            else:
                st.caption("No close comparables found in the dataset.")
        else:
            st.info("👈 Set property details in the sidebar and click **Predict Price**.")

    # Residual distribution for context
    st.markdown("---")
    st.subheader("📉 Prediction Residuals Distribution (Test Set)")
    if os.path.exists(os.path.join(STATIC_DIR, "actual_vs_predicted.png")):
        img = Image.open(os.path.join(STATIC_DIR, "actual_vs_predicted.png"))
        st.image(img, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏆 Model Comparison")

    perf_df = pd.DataFrame([
        {
            "Model": n,
            "MAE": round(v["MAE"], 3),
            "RMSE": round(v["RMSE"], 3),
            "R² Score": round(v["R2"], 4),
            "Best": "✅" if n == model_name else ""
        }
        for n, v in all_results.items()
    ]).sort_values("R² Score", ascending=False).reset_index(drop=True)

    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Bar charts side by side
    if os.path.exists(os.path.join(STATIC_DIR, "model_comparison.png")):
        img = Image.open(os.path.join(STATIC_DIR, "model_comparison.png"))
        st.image(img, use_container_width=True, caption="MAE / RMSE / R² across all models")

    # Feature importance
    feat_img = os.path.join(STATIC_DIR, "feature_importance.png")
    if os.path.exists(feat_img):
        st.subheader("📌 Feature Importances")
        img = Image.open(feat_img)
        st.image(img, use_container_width=True)
    else:
        # Show coefficients for linear models
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.subheader("📌 Model Coefficients")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#3b82d4" if c > 0 else "#ef4444" for c in coef_df["Coefficient"]]
            ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
            ax.set_title("Standardised Coefficients", fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Dataset Exploration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",      len(df))
    c2.metric("Avg Price",          f"{df['price_per_unit_area'].mean():.1f}")
    c3.metric("Min Price",          f"{df['price_per_unit_area'].min():.1f}")
    c4.metric("Max Price",          f"{df['price_per_unit_area'].max():.1f}")

    st.markdown("---")

    # Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df["price_per_unit_area"], bins=30, color="#3b82d4", edgecolor="white")
    axes[0].set_title("Price Distribution", fontweight="bold")
    axes[0].set_xlabel("Price per Unit Area")
    axes[0].set_ylabel("Count")

    axes[1].scatter(df["distance_to_mrt"], df["price_per_unit_area"],
                    alpha=0.5, color="#7c5cd8", edgecolors="white", linewidths=0.3)
    axes[1].set_title("Distance to MRT vs Price", fontweight="bold")
    axes[1].set_xlabel("Distance to MRT (m)")
    axes[1].set_ylabel("Price per Unit Area")
    plt.tight_layout()
    st.pyplot(fig)

    # Age vs price
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    axes2[0].scatter(df["house_age"], df["price_per_unit_area"],
                     alpha=0.5, color="#22c55e", edgecolors="white", linewidths=0.3)
    axes2[0].set_title("House Age vs Price", fontweight="bold")
    axes2[0].set_xlabel("House Age (years)")
    axes2[0].set_ylabel("Price per Unit Area")

    axes2[1].scatter(df["convenience_stores"], df["price_per_unit_area"],
                     alpha=0.5, color="#f59e0b", edgecolors="white", linewidths=0.3)
    axes2[1].set_title("Convenience Stores vs Price", fontweight="bold")
    axes2[1].set_xlabel("Number of Convenience Stores")
    axes2[1].set_ylabel("Price per Unit Area")
    plt.tight_layout()
    st.pyplot(fig2)

    # Heatmap
    st.subheader("🌡️ Correlation Heatmap")
    if os.path.exists(os.path.join(STATIC_DIR, "correlation_heatmap.png")):
        img = Image.open(os.path.join(STATIC_DIR, "correlation_heatmap.png"))
        st.image(img, use_container_width=True)

    # Geographic scatter
    st.subheader("🗺️ Geographic Price Map")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sc = ax3.scatter(
        df["longitude"], df["latitude"],
        c=df["price_per_unit_area"], cmap="RdYlGn",
        alpha=0.7, s=60, edgecolors="white", linewidths=0.3
    )
    plt.colorbar(sc, ax=ax3, label="Price per Unit Area")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Property Locations — Coloured by Price", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dataset
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Full Dataset")

    search_col = st.columns([1, 1, 1])
    min_p = float(df["price_per_unit_area"].min())
    max_p = float(df["price_per_unit_area"].max())
    price_range = st.slider(
        "Filter by price range",
        min_value=min_p, max_value=max_p,
        value=(min_p, max_p), step=0.5
    )
    filtered = df[df["price_per_unit_area"].between(*price_range)]
    st.caption(f"Showing {len(filtered)} / {len(df)} records")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="filtered_real_estate.csv",
        mime="text/csv"
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#57606a; font-size:0.8rem;'>"
    "Real Estate Price Predictor · Taiwan Dataset · Built with Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
