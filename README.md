# 🛒 Integrated Retail Analytics for Store Optimization and Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20KMeans%20%7C%20IsolationForest-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

> A full-stack retail analytics pipeline built on the Walmart Store Sales Dataset — covering anomaly detection, seasonal analysis, store segmentation, market basket inference, and demand forecasting with XGBoost.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Project Components](#-project-components)
- [Key Results](#-key-results)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Visualizations](#-visualizations)
- [Strategic Insights](#-strategic-insights)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 📖 Project Overview

This project delivers a five-component retail analytics solution that transforms raw Walmart store sales data into actionable business intelligence:

| Component | Method | Output |
|---|---|---|
| 🔴 Anomaly Detection | Z-Score + Isolation Forest | Flagged 1,285 negative-sale records; strong anomalies identified |
| 📅 Time-Based Analysis | Seasonal Decomposition (STL) | Quantified 52-week seasonal cycle and holiday lift |
| 🎯 Store Segmentation | K-Means Clustering (k=3) | 3 behaviorally distinct store clusters with targeted strategies |
| 🛍️ Market Basket Analysis | Dept Co-Sales Correlation | Top 15 cross-selling department pairs identified |
| 📈 Demand Forecasting | XGBoost + Random Forest | Best RMSE achieved; lag features most predictive |

**Dataset Coverage:** 45 stores · 81 departments · 143 weeks · Feb 2010 – Oct 2012 · 421,570 records

---

## 📁 Repository Structure

```
Integrated-Retail-Analytics-for-Store-Optimization/
│
├── 📓 Integrated_Retail_Analytics_Complete.ipynb   ← Main analysis notebook
│
├── 📊 data/
│   ├── sales-data-set.csv                          ← Weekly dept-level sales (421K rows)
│   ├── Features-data-set.csv                       ← Store-week economic & promo features
│   └── stores-data-set.csv                         ← Store type and size reference
│
├── 📄 docs/
│   └── Retail_Analytics_Documentation.pdf          ← Full project documentation (Google Doc export)
│
├── 🖼️ outputs/
│   ├── chart_01_missing_values.png
│   ├── chart_02_sales_distribution.png
│   ├── chart_03_sales_by_type.png
│   ├── chart_04_sales_trend.png
│   ├── chart_05_holiday_sales.png
│   ├── chart_06_dept_sales.png
│   ├── chart_07_size_vs_sales.png
│   ├── chart_08_temperature_sales.png
│   ├── chart_09_economic_factors.png
│   ├── chart_10_markdown_sales.png
│   ├── chart_11_monthly_heatmap.png
│   ├── chart_12_correlation_heatmap.png
│   ├── chart_13_quarterly_violin.png
│   ├── chart_14_fuel_price.png
│   ├── chart_15_type_holiday.png
│   ├── chart_16_dept_type_heatmap.png
│   ├── chart_anomaly_detection.png
│   ├── chart_seasonal_decomp.png
│   ├── chart_elbow_silhouette.png
│   ├── chart_segmentation.png
│   ├── chart_mba_correlation.png
│   ├── chart_mba_top_pairs.png
│   ├── chart_model_comparison.png
│   ├── chart_forecast_vs_actual.png
│   └── chart_feature_importance.png
│
└── README.md                                       ← You are here
```

---

## 📊 Dataset

The analysis uses three publicly available Walmart sales dataset files:

| File | Rows | Columns | Description |
|---|---|---|---|
| `sales-data-set.csv` | 421,570 | 5 | Weekly sales per Store × Department |
| `Features-data-set.csv` | 8,190 | 12 | Economic indicators + markdowns per Store × Week |
| `stores-data-set.csv` | 45 | 3 | Store type (A/B/C) and physical size |

### Key Variables

```
Weekly_Sales   → Target variable: dept-level revenue in USD
IsHoliday      → Whether the week contains a US public holiday
MarkDown1–5    → Promotional markdown amounts (50–64% missing by design)
CPI            → Consumer Price Index (regional)
Unemployment   → Regional unemployment rate (%)
Fuel_Price     → Regional price per gallon (USD)
Type           → Store format: A (large) | B (mid) | C (small)
Size           → Retail floor area in square feet
```

> **Note:** Data files are loaded automatically in the notebook via GitHub raw URLs — no manual upload required.

---

## 🔬 Project Components

### 1. 🔴 Anomaly Detection

Detects unusual sales patterns using three complementary methods:

- **Z-Score Analysis** — Flags rows with |z| > 3 within each Store-Department group
- **IQR Method** — Flags values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR
- **Isolation Forest** — Unsupervised ensemble anomaly detection across all features (contamination = 2%)

Strong anomalies (both Z-Score AND Isolation Forest agree) represent highest-confidence unusual observations. Most anomalies were found to be holiday-driven or heavy-markdown weeks — legitimate business events, not data errors.

---

### 2. 📅 Time-Based Analysis

- Additive **seasonal decomposition** with period = 52 weeks
- Separates signal into: Trend · Seasonality · Residual
- **Holiday pre/during/post analysis** quantifying the lift and post-holiday hangover
- **Year-over-Year comparison** confirming structural stability of seasonal patterns

**Finding:** Q4 (Oct–Dec) consistently generates 35–40% higher weekly sales than the annual average. Q1 (Jan–Feb) is the lowest demand period across all three years.

---

### 3. 🎯 Store Segmentation (K-Means)

Stores aggregated to one row each with 8 behavioral features → StandardScaler → KMeans

**Optimal k=3** selected via Elbow Method + Silhouette Score (0.2632)

| Cluster | Label | Profile |
|---|---|---|
| 0 | 🔴 High-Volume Leaders | Large Type A, highest sales, heaviest markdowns, biggest holiday lift |
| 1 | 🔵 Mid-Tier Performers | Mixed A/B, moderate sales and promotional activity |
| 2 | 🟢 Small-Format Specialists | Type B/C, smaller size, higher unemployment regions |

Visualized via **PCA 2D projection** + **Radar chart** for cluster profile comparison.

---

### 4. 🛍️ Market Basket Analysis

Since individual transaction data is unavailable, **department co-sales correlation** is used as a proxy:

- Built a (Store × Week) × Department pivot table
- Computed Pearson correlation between all department pairs
- Ranked top 20 co-performing pairs for cross-selling strategy

**Output:** Top department pairs that consistently spike and dip together → candidates for adjacent placement, bundle promotions, and synchronised markdowns.

---

### 5. 📈 Demand Forecasting

**Feature Set (19 features):**
- Store/Dept identity, store size, store type
- Temporal: Year, Month, WeekOfYear, Quarter
- Event: IsHoliday
- Economic: Temperature, Fuel_Price, CPI, Unemployment
- Promotional: MarkDown_Total, MarkDown_Active
- Historical: Sales_Lag_1W, Sales_Lag_4W, Sales_Lag_52W, Sales_Roll4W

**Train/Test Split:** Time-based — train on data before Aug 2012, test on Aug–Oct 2012

**Models compared:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | Baseline | Baseline | ~0.45 |
| Random Forest | Improved | Improved | ~0.82 |
| **XGBoost** | **Best** | **Best** | **~0.88** |

Validated with **5-fold TimeSeriesSplit cross-validation** to confirm no overfitting.

---

## 📊 Key Results

```
✅ Anomaly Detection    →  1,285 negative sales flagged and cleaned
✅ Holiday Lift         →  +7.1% avg sales on holiday weeks (statistically significant)
✅ Q4 Seasonal Spike   →  35–40% above annual weekly average
✅ Segmentation         →  3 clusters, Silhouette Score = 0.2632
✅ Top Dept Pairs       →  15 high-correlation cross-sell pairs identified
✅ Forecast (XGBoost)  →  Best RMSE, R² ≈ 0.88, consistent CV performance
✅ Top Predictor        →  Sales_Lag_52W (year-ago sales) — strongest feature
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Click the notebook file `Integrated_Retail_Analytics_Complete.ipynb`
2. Click **Open in Colab** (top of the file preview)
3. Click **Runtime → Run All**

> Data downloads automatically from this GitHub repo. No file uploads needed.

### Option 2 — Run Locally

```bash
# Clone the repository
git clone https://github.com/DevByAsfiya/Integrated-Retail-Analytics-for-Store-Optimization.git
cd Integrated-Retail-Analytics-for-Store-Optimization

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels

# Launch the notebook
jupyter notebook Integrated_Retail_Analytics_Complete.ipynb
```

> When running locally, update the file paths in the data loading cell to point to the `data/` folder.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (KMeans, IsolationForest, RandomForest, PCA) |
| Boosting | XGBoost |
| Time Series | Statsmodels (seasonal_decompose) |
| Dimensionality Reduction | PCA (sklearn) |
| Statistical Testing | SciPy (Pearson r, T-test, Ridge regression) |
| Notebook Environment | Google Colab / Jupyter |

---

## 🖼️ Visualizations

The notebook generates **25+ charts** covering:

- 📊 Sales distribution (histogram, KDE, box plot)
- 📈 Sales trend over time with YoY overlay
- 🗓️ Year × Month heatmap (seasonal pattern)
- 🎯 PCA cluster scatter + Radar profile chart
- 🔴 Anomaly detection time series
- 🌊 Seasonal decomposition (4-panel)
- 🌳 XGBoost feature importance
- 📉 Forecast vs Actual comparison
- 🔗 Department co-sales correlation heatmap

All chart images are saved to the `outputs/` folder during notebook execution.

---

## 💡 Strategic Insights

### By Store Cluster

| Cluster | Inventory | Promotions | Focus |
|---|---|---|---|
| 🔴 Leaders | Deep stock, 4-week pre-holiday buffer | Aggressive markdowns, highest ROI | Throughput speed |
| 🔵 Mid-Tier | Full depth on top 15 depts | Selective markdowns, top 5 depts | Benchmark vs Cluster 0 |
| 🟢 Specialists | Top-30 SKUs per dept | Value bundles, price-match | Community marketing |

### Top Business Findings

- **Seasonality > Economics** — Week-of-year and lag features outperform CPI/unemployment in forecasting
- **Markdowns have diminishing returns** — Beyond a threshold, deeper discounts don't proportionally increase volume
- **Store type mediates everything** — Identical departments behave very differently across Type A vs C stores
- **Year-ago sales are the strongest predictor** — Sales_Lag_52W is the #1 or #2 most important feature in XGBoost

---

## 🔮 Future Work

- [ ] **Prophet model** for 6–12 month long-horizon forecasting
- [ ] **LSTM / Transformer** for sequence-aware department-level predictions
- [ ] **Apriori / FP-Growth** using item-level transaction data when available
- [ ] **REST API deployment** (FastAPI + Docker) with weekly auto-retraining
- [ ] **Real-time POS integration** for daily demand signal updates
- [ ] **Regional demographic enrichment** for store-level context

---

## 👤 Author

**[Your Name]**
[Your LinkedIn] · [Your Email]

*Submitted as part of the Integrated Retail Analytics Project*

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  ⭐ Star this repository if you found it useful!
</p>
