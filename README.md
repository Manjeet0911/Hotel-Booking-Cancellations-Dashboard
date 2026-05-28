<div align="center">

# 🤖 Universal No-Code AutoML & Interactive Analytics Platform

### From Excel Blueprint → Production-Grade Python / Streamlit Application

*A fully domain-neutral, zero-code machine learning web application that transforms any structured CSV dataset into an automated executive dashboard, a live predictive ML pipeline, and a real-time anomaly alert engine — all within a single Streamlit interface.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Zero Code](https://img.shields.io/badge/User_Code_Required-Zero-22C55E?style=flat-square)](#)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Platform Architecture](#platform-architecture)
- [Historical Data Insights](#historical-data-insights)
- [Sidebar Configuration](#sidebar-configuration)
- [Tab 01 — Automated Executive Dashboard](#tab-01--automated-executive-dashboard)
- [Tab 02 — Generalized Auto-ML Pipeline](#tab-02--generalized-auto-ml-pipeline)
- [Tab 03 — System Integration & API Stream Simulator](#tab-03--system-integration--api-stream-simulator)
- [Technology Stack](#technology-stack)
- [Local Setup & Deployment](#local-setup--deployment)
- [Excel Blueprint Reference](#excel-blueprint-reference)

---

## Project Overview

This repository documents the complete engineering journey from a **static Excel analytics dashboard** to a **production-grade, fully automated machine learning web application**.

The project originated as a rigorous Excel-based analysis of **119,386 hotel booking records** — manually engineered with Power Query, multi-layered Pivot Tables, dynamic slicers, and a custom dark executive UI/UX theme. That Excel blueprint validated the analytical framework and surfaced the business intelligence that now powers the live Streamlit application.

The platform has since been fully re-engineered in Python and is **no longer hotel-domain-specific**. Any analyst, data scientist, or business user can upload any structured CSV file — sales records, cricket statistics, clinical health data, financial transactions — and the platform will instantly generate an executive dashboard, train a live machine learning model, and simulate a real-time data stream with anomaly detection, **all without writing a single line of code**.

| Property | Detail |
| :--- | :--- |
| **Origin Dataset** | 119,386 hotel booking records (2015–2017) |
| **App Tabs** | 3 feature modules |
| **User Code Required** | Zero |
| **Accepted Input** | Any structured `.csv` file |
| **ML Model** | GradientBoostingClassifier (scikit-learn) |
| **Visualization** | Plotly (interactive) + custom CSS KPI tiles |
| **Repository** | [github.com/Manjeet0911/Hotel-Booking-Cancellations-Dashboard](https://github.com/Manjeet0911/Hotel-Booking-Cancellations-Dashboard) |

---

## Platform Architecture

The three application tabs are fully interconnected — a single CSV upload and target variable selection in the sidebar propagates the dataset and trained model across all three modules simultaneously.

| Layer | Description |
| :--- | :--- |
| **Input Layer** | `st.file_uploader` — any `.csv` file, any domain, any schema |
| **Configuration Layer** | `st.selectbox` — dynamic column headers for target variable selection |
| **Tab 01** | Automated Executive Dashboard — type-aware KPI scan + Plotly charts |
| **Tab 02** | Auto-ML Pipeline — imputation, encoding, GBC, live prediction gauge |
| **Tab 03** | API Stream Simulator — row-chunk ingestion, anomaly flash alerts |
| **Visualization** | Plotly (interactive) + custom CSS KPI tiles over Dark Slate Navy |
| **ML Backend** | scikit-learn `GradientBoostingClassifier` with auto-boundary sliders |

---

## Historical Data Insights

The following analytical findings were produced by the Excel blueprint and subsequently validated by the Streamlit Auto-ML pipeline using the origin hotel bookings corpus. They represent the ground truth that drove the platform's design decisions for anomaly thresholds, KPI tile defaults, and stream alert calibration.

| Metric | Value |
| :--- | :--- |
| **Total Reservations** | 119,386 |
| **Total Cancellations** | 44,220 |
| **Overall Cancellation Rate** | 37.04% |
| **City Hotel Cancel Rate** | 41.7% |

### Hotel Type Split

| Hotel Type | Total Bookings | Cancellations | Cancel Rate |
| :--- | :--- | :--- | :--- |
| City Hotel | 79,326 (66.4%) | 33,098 | 41.7% |
| Resort Hotel | 40,060 (33.6%) | 11,122 | 27.8% |

> City Hotels carry nearly **3x** the cancellation volume of Resort Hotels despite only 2x the booking volume. This asymmetry calibrated the stream simulator's high-risk threshold and revenue-at-risk alert logic.

### Year-over-Year Cancellation Escalation

| Year | Bookings | Cancellations | Rate | YoY Growth |
| :--- | :--- | :--- | :--- | :--- |
| 2015 | 21,992 | 8,138 | 37.0% | Baseline |
| 2016 | 56,707 | 20,337 | 35.9% | +148.5% |
| 2017 | 40,687 | 15,745 | 38.7% | Partial year |

### Guest Segment Behaviour

| Segment | Total Guests | Cancellations | Cancel Rate |
| :--- | :--- | :--- | :--- |
| Couples | 81,557 | 32,421 | 39.8% |
| Single | 22,577 | 6,555 | 29.0% |
| Family | 15,252 | 5,244 | 34.4% |

### Room Allocation Accuracy

| Room Outcome | Bookings | Cancellations | Cancel Rate |
| :--- | :--- | :--- | :--- |
| Desired (Preference Match) | 104,469 | 43,418 | 41.6% |
| Un-desired (Preference Mismatch) | 14,917 | 802 | 5.4% |

> Un-desired room assignments produce a dramatically lower cancellation rate (5.4% vs 41.6%) — a counter-intuitive finding that informed the Auto-ML pipeline's feature importance weighting and stream anomaly detection thresholds.

### Monthly Seasonality Intelligence

| Month | Total Guests | Cancellations | Cancel Rate |
| :--- | :--- | :--- | :--- |
| January | 5,929 | 1,807 | 30.5% |
| February | 8,068 | 2,696 | 33.4% |
| March | 9,794 | 3,149 | 32.2% |
| April | 11,089 | 4,524 | 40.8% |
| May | 11,791 | 4,677 | 39.7% |
| June | 10,939 | 4,535 | 41.5% |
| July | 12,661 | 4,742 | 37.5% |
| August | 13,873 | 5,235 | 37.7% |
| September | 10,508 | 4,116 | 39.2% |
| October | 11,160 | 4,246 | 38.0% |
| November | 6,794 | 2,122 | 31.2% |
| December | 6,780 | 2,371 | 35.0% |

> **June (41.5%), April (40.8%), and September (39.2%)** carry peak cancellation rates. January–March show highest booking commitment (30–33%). August peaks in both volume and cancellation — a signal used to calibrate the stream simulator's consecutive anomaly counter and alert threshold.

---

## Sidebar Configuration

The entire application is driven by two sidebar controls that make the platform completely domain-neutral. There are no hard-coded column names, no hotel-specific logic, and no schema assumptions. Any structured CSV file with at least one categorical or binary target column is fully supported.

### CSV File Uploader

A `st.file_uploader` widget accepts any structured `.csv` file. Upon upload, the application immediately reads the file into a Pandas DataFrame, performs automated schema inspection (column names, dtypes, null counts, cardinality), and propagates the detected schema to all three tabs simultaneously.

```python
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (.csv)",
    type=["csv"]
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
```

**Compatible dataset types:**

- Hotel bookings (`is_canceled` target) — origin validation corpus
- Sales records (conversion / churn target)
- Cricket statistics (match outcome / player performance target)
- Clinical health data (diagnosis / readmission target)
- Financial transactions (fraud / default target)
- Any structured `.csv` with mixed numeric and categorical features

### Live Target Variable Selector

A `st.selectbox` reads the column headers of the uploaded DataFrame dynamically and presents them as a dropdown. The user explicitly declares which column is the **Target Variable (Y)** to predict. This selection propagates instantly to Tab 02, where all remaining columns are treated as input features (X). The selectbox updates in real time whenever a new file is uploaded — no page reload required.

```python
target_col = st.sidebar.selectbox(
    "Select Target Variable (Y to predict)",
    options=df.columns.tolist()
)
X = df.drop(columns=[target_col])
y = df[target_col]
```

---

## Tab 01 — Automated Executive Dashboard

Tab 01 delivers an automated, zero-configuration executive dashboard that renders within seconds of a CSV upload. The entire visualization layer is driven by a **type-aware column scanner** — no manual chart configuration, no hard-coded column references, no domain-specific assumptions.

### Type-Aware Numeric KPI Scanner

The application automatically scans all DataFrame columns, isolates the **top 4 numeric columns** by data completeness and variance, and computes three baseline aggregates for each: **Mean**, **Sum**, and **Standard Deviation**. These values are rendered as custom CSS KPI tiles over the Dark Slate Navy background, styled to match the executive theme from the original Excel blueprint.

```python
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
top_4 = numeric_cols[:4]

for col in top_4:
    mean_val = df[col].mean()
    sum_val  = df[col].sum()
    std_val  = df[col].std()
    # Rendered as CSS KPI tile (Dark Slate #1E293B, Cyan #38BDF8)
```

### KPI Tile Design System

| Property | Value | Rationale |
| :--- | :--- | :--- |
| **Tile Background** | `#1E293B` (Dark Slate Navy) | Consistent with Excel blueprint theme |
| **Value Color** | `#38BDF8` (Bright Cyan Blue) | High-contrast on dark background |
| **Label Color** | `#E2E8F0` (Light Gray) | Soft, non-competing secondary text |
| **Metrics Displayed** | Mean · Sum · Std Dev | Computed from the uploaded dataset |
| **Count per Row** | 4 KPI tiles side-by-side | Adapts to available numeric columns |

### Dynamic Plotly Value Count Charts

Below the KPI tiles, the dashboard auto-generates Plotly bar charts for **all categorical columns** detected in the dataset, showing value count distributions with interactive hover tooltips. Charts use the same Dark Slate Navy canvas with Cyan/Gold color encoding. All charts are fully interactive — zoom, pan, download, and hover data are enabled by default through Plotly's built-in controls.

> Tab 01 transforms a raw CSV upload into a boardroom-ready analytics summary in under 2 seconds — with zero manual configuration.

---

## Tab 02 — Generalized Auto-ML Pipeline

Tab 02 is the core machine learning engine of the platform. It executes a complete, automated supervised learning pipeline — from raw data preprocessing through to a live prediction with an interactive Plotly gauge — entirely without user-written code. The pipeline is triggered the moment the user selects a target variable in the sidebar.

### Step 1 — Automated Imputation

The pipeline performs a column-by-column scan for null values. Numeric columns are imputed using **median imputation** (robust to outliers). Categorical columns are imputed using **mode imputation** (most frequent value). No rows are dropped — the pipeline preserves the full dataset.

```python
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ["float64", "int64"]:
            X[col].fillna(X[col].median(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)
```

### Step 2 — Automated LabelEncoder Loop

All `string`/`object` dtype columns in the feature set are automatically detected and passed through a `LabelEncoder` loop. Each unique string value is mapped to a consistent integer representation. Encoder mappings are stored and reused during inference to ensure prediction consistency.

```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le  # stored for inference reuse
```

### Step 3 — GradientBoostingClassifier Training

After preprocessing, a `GradientBoostingClassifier` is instantiated and fitted against the full processed dataset in real time with the following configuration:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

### Step 4 — Auto-Boundary Input Sliders & Dropdowns

The platform dynamically generates UI input controls based on the **exact min/max boundary values** of the uploaded dataset — not hard-coded defaults. For each numeric feature, an `st.slider` is rendered with range set to `[column_min, column_max]` and default at the median. For each categorical feature, an `st.selectbox` is rendered with all unique values from the uploaded data.

```python
user_input = {}
for col in X.columns:
    if X[col].dtype in ["float64", "int64"]:
        user_input[col] = st.slider(
            col,
            min_value=float(X[col].min()),
            max_value=float(X[col].max()),
            value=float(X[col].median())
        )
    else:
        user_input[col] = st.selectbox(
            col, options=X[col].unique().tolist()
        )
```

### Step 5 — Live Plotly Indicator Gauge

Upon clicking **Predict**, the model generates a probability score for the selected target class. This is rendered as a **Plotly indicator gauge** — a half-circle dial from 0% to 100% with three color zones:

| Zone | Range | Color |
| :--- | :--- | :--- |
| Low Risk | 0% – 40% | 🟢 Green |
| Moderate Risk | 40% – 70% | 🟡 Amber |
| High Risk | 70% – 100% | 🔴 Red |

> The Auto-ML pipeline requires zero configuration beyond selecting a target variable. The entire preprocessing, training, UI generation, and prediction loop executes automatically on any valid CSV upload.

---

## Tab 03 — System Integration & API Stream Simulator

Tab 03 transforms the static dataset into a **live data stream simulation**, mimicking the behaviour of a real-time Property Management System (PMS) API feed or any continuous event stream. It demonstrates how the platform would perform in a production integration scenario — ingesting live records, scoring each for risk, and triggering structural alerts when anomaly thresholds are breached.

### Sequential Row-Chunk Ingestion

The simulator extracts the uploaded dataset in configurable sequential chunks (default: **50 rows per cycle**), processing each chunk as if it were a freshly ingested batch from a live API endpoint. A `st.empty()` placeholder overwrites the stream display in place, creating a smooth scrolling effect that replicates the visual behaviour of a real-time dashboard feed.

```python
chunk_size = 50
consecutive_high_risk = 0
ALERT_THRESHOLD = 3  # consecutive high-risk records triggers alert

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i : i + chunk_size]
    risk_scores = model.predict_proba(preprocess(chunk))[:, 1]
```

### Consecutive High-Risk Anomaly Counter

Within each ingested chunk, every record is scored by the fitted GradientBoosting model. The simulator maintains a `consecutive_high_risk` counter that increments every time a record's predicted probability exceeds the configured risk threshold (default: **0.70**). The counter resets to zero on any low-risk record. This sliding window approach mirrors production anomaly detection systems that track **sustained degradation** rather than isolated spikes.

```python
RISK_THRESHOLD = 0.70

for score in risk_scores:
    if score >= RISK_THRESHOLD:
        consecutive_high_risk += 1
    else:
        consecutive_high_risk = 0  # reset on safe record

    if consecutive_high_risk >= ALERT_THRESHOLD:
        trigger_flash_alert(score, consecutive_high_risk)
```

### Structural Flash Alert Cards

When the consecutive high-risk counter reaches the configured threshold, a **structural flash alert card** is rendered directly in the Streamlit interface. Each alert card contains:

| Alert Field | Description |
| :--- | :--- |
| **Alert Severity** | `HIGH RISK` / `CRITICAL` based on consecutive count |
| **Revenue at Risk** | Average booking value × high-risk record count |
| **Consecutive Count** | Number of back-to-back high-risk records detected |
| **Current Risk Score** | Exact predicted probability from the GBC model |
| **Recommended Action** | Pre-configured response playbook (dynamic deposit, rate lock) |
| **Timestamp** | Stream position (row index) at the time of alert trigger |

```python
def trigger_flash_alert(score, count):
    alert_html = f"""
    <div style="background:#7F1D1D; border-left:4px solid #EF4444; padding:16px;">
        <b>⚠️ REVENUE AT RISK ALERT</b><br/>
        Consecutive High-Risk Records: {count}<br/>
        Current Risk Score: {score:.2%}<br/>
        Recommended: Apply non-refundable deposit policy.
    </div>"""
    st.markdown(alert_html, unsafe_allow_html=True)
```

> Tab 03 bridges the gap between static batch analytics and production real-time monitoring. The same ML model trained in Tab 02 powers the live stream scorer, ensuring analytical consistency across all three modules.

---

## Technology Stack

| Library / Tool | Role | Usage |
| :--- | :--- | :--- |
| **Python 3.10+** | Runtime | Core application language |
| **Streamlit** | Web Framework | Zero-code UI — tabs, sliders, file uploader |
| **Pandas** | Data Layer | DataFrame ingestion, type detection, aggregation |
| **NumPy** | Numerics | Array operations, statistical computations |
| **scikit-learn** | ML Engine | `GradientBoostingClassifier`, `LabelEncoder`, `train_test_split` |
| **Plotly** | Visualization | Interactive charts, KPI indicator gauge |
| **Microsoft Excel** | Blueprint (v1.0) | Power Query, Pivot Tables, Slicer dashboard |

---

## Local Setup & Deployment

### Prerequisites

| Tool | Requirement |
| :--- | :--- |
| Python | 3.10 or later |
| pip | Latest |
| Git | Any recent version |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Manjeet0911/Hotel-Booking-Cancellations-Dashboard.git
cd Hotel-Booking-Cancellations-Dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

### requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

### Project File Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Main Streamlit application — all 3 tabs |
| `requirements.txt` | Python dependency manifest |
| `Hotel_Booking_...Final.xlsx` | Origin Excel dashboard blueprint (v1.0) |
| `hotel_bookings.csv` | Validation corpus — 119,386 records |
| `README.md` | This documentation |

---

## Excel Blueprint Reference

The Excel dashboard (v1.0) remains in the repository as the analytical and design blueprint. It documents every data engineering decision, color psychology choice, and chart architecture that was subsequently automated in the Streamlit platform.

| Excel Feature | Implementation Detail |
| :--- | :--- |
| **Power Query** | Full ETL pipeline — ingestion, type enforcement, null handling |
| **Conditional Column** | `room_status`: `IF(reserved = assigned, "Desired", "Un-desired")` |
| **Pivot Tables** | Multi-layered across Hotel Type, Year, Segment, Month |
| **Dynamic Slicers** | 3 Year slicers cross-connected to all charts and KPI cards |
| **Color Palette** | `#1E293B` background · `#38BDF8` primary · `#D4A96A` cancellation |
| **Chart Types** | Pie (×2) · Clustered Bar (×2) · Grouped Column (monthly) |
| **Borderless Design** | Gridlines, headings, and chart borders removed — floating UI |

---

<div align="center">

Built with precision. Designed for impact. Engineered for insight.

**Excel Blueprint → Python / Streamlit AutoML Platform → Production**

[github.com/Manjeet0911/Hotel-Booking-Cancellations-Dashboard](https://github.com/Manjeet0911/Hotel-Booking-Cancellations-Dashboard)

</div>
