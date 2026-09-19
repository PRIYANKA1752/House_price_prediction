# 🏠 Real Estate House Price Predictor

A machine-learning web application that predicts residential property prices (per unit area)
using historical transaction data and location-based features from the **Taiwan Real Estate Valuation Dataset**.

---

## 📌 Features

| Feature | Description |
|---|---|
| Transaction Date | Year.Month of the sale |
| House Age | Age of the property in years |
| Distance to MRT | Distance (meters) to the nearest MRT station |
| Convenience Stores | Number of nearby convenience stores |
| Latitude / Longitude | GPS coordinates of the property |

**Target:** `price_per_unit_area` (10,000 NTD / Ping)

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline, scaled input |
| Ridge Regression | Regularised linear model |
| Random Forest | Ensemble of decision trees ✅ typically best |
| Gradient Boosting | Sequential boosting ensemble |

The best-performing model (highest R²) is automatically saved and used for predictions.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd real_estate_project
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train_model.py
```

This produces:
- `models/best_model.pkl` — serialised best model
- `models/scaler.pkl` — fitted StandardScaler
- `models/features.pkl` — feature list
- `static/*.png` — performance charts

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
real_estate_project/
├── data/
│   └── real_estate.csv          # Dataset
├── models/                      # Saved model artefacts (after training)
├── static/                      # Generated charts (after training)
├── train_model.py               # ML training pipeline
├── app.py                       # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## 📊 App Tabs

| Tab | Contents |
|---|---|
| 🏷️ Prediction | Sidebar inputs → live price prediction + comparable properties |
| 📊 Model Performance | Metrics table, model comparison chart, feature importances |
| 📈 Data Insights | Price distribution, scatter plots, correlation heatmap, geo map |
| 📋 Dataset | Filterable full dataset + CSV download |

---

## 📈 Dataset

**Source:** UCI Machine Learning Repository — Real Estate Valuation Data Set  
**Records:** 414 transactions (Sindian District, New Taipei City, Taiwan, 2012–2013)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — model training & evaluation
- **Streamlit** — interactive web frontend
- **pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualisations
- **joblib** — model serialisation
