"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        UNIVERSAL NO-CODE MACHINE LEARNING & ANALYTICS PLATFORM              ║
║        Works with ANY user-uploaded CSV file, fully dynamic.                ║
║                                                                              ║
║  Run:  pip install streamlit plotly scikit-learn pandas numpy                ║
║        streamlit run app.py                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":      "#1E293B",
    "surface": "#0F172A",
    "surf2":   "#1E2D3D",
    "surf3":   "#243447",
    "primary": "#38BDF8",
    "sec":     "#0EA5E9",
    "accent":  "#D4A96A",
    "success": "#34D399",
    "danger":  "#F87171",
    "warn":    "#FBBF24",
    "text":    "#F1F5F9",
    "muted":   "#94A3B8",
    "border":  "#334155",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="'IBM Plex Sans', sans-serif", size=12),
    xaxis=dict(
        gridcolor=C["border"],
        linecolor=C["border"],
        zeroline=False,
        showgrid=True,
        tickfont=dict(color=C["muted"]),
    ),
    yaxis=dict(
        gridcolor=C["border"],
        linecolor=C["border"],
        zeroline=False,
        showgrid=True,
        tickfont=dict(color=C["muted"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C["border"],
        font=dict(color=C["muted"]),
    ),
    hoverlabel=dict(
        bgcolor=C["surface"],
        font_color=C["text"],
        bordercolor=C["border"],
    ),
    margin=dict(l=40, r=20, t=48, b=40),
    colorway=[
        C["primary"], C["accent"], C["sec"],
        C["success"], C["danger"], C["warn"],
        "#A78BFA", "#FB923C", "#2DD4BF",
    ],
)


def themed(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    """Apply the platform dark theme to any Plotly figure."""
    fig.update_layout(
        **PLOTLY_BASE,
        title_text=title,
        title_font=dict(color=C["text"], size=14),
        height=height,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
  
  html, body, [class*="css"] {{
      font-family: 'IBM Plex Sans', sans-serif;
      background-color: {C['bg']};
      color: {C['text']};
  }}
  .stApp {{ background-color: {C['bg']}; }}
  .block-container {{ padding-top: 3rem !important; padding-bottom: 2rem; max-width: 1600px; }}
header[data-testid="stHeader"] button p {{ color: #FFFFFF !important; }}
footer {{ visibility: hidden; }}

  section[data-testid="stSidebar"] {{
      background-color: {C['surface']};
      border-right: 1px solid {C['border']};
      
  }}
  section[data-testid="stSidebar"] .stMarkdown p {{ color: {C['muted']}; }}

  .stTabs [data-baseweb="tab-list"] {{
      gap: 3px;
      background: {C['surface']};
      border-radius: 10px;
      padding: 4px;
      border: 1px solid {C['border']};
      margin-bottom: 0.5rem;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 8px;
      color: {C['muted']};
      font-weight: 500;
      font-size: 0.88rem;
      padding: 8px 22px;
      border: none;
      transition: all 0.15s ease;
  }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg, {C['surf2']}, {C['surf3']});
      color: {C['primary']} !important;
      border-bottom: 2px solid {C['primary']};
  }}
  .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.25rem; }}

  .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-bottom: 1.5rem;
  }}
  .kpi-card {{
      background: linear-gradient(145deg, {C['surface']} 0%, {C['surf2']} 100%);
      border: 1px solid {C['border']};
      border-radius: 14px;
      padding: 1.2rem 1.4rem;
      position: relative;
      overflow: hidden;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  .kpi-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  }}
  .kpi-card::after {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      border-radius: 14px 14px 0 0;
  }}
  .kpi-card.c0::after {{ background: linear-gradient(90deg, {C['primary']}, {C['sec']}); }}
  .kpi-card.c1::after {{ background: linear-gradient(90deg, {C['accent']}, #F59E0B); }}
  .kpi-card.c2::after {{ background: linear-gradient(90deg, {C['success']}, #059669); }}
  .kpi-card.c3::after {{ background: linear-gradient(90deg, #A78BFA, #7C3AED); }}
  .kpi-icon  {{ font-size: 1.6rem; margin-bottom: 8px; opacity: 0.85; }}
  .kpi-label {{ color: {C['muted']}; font-size: 0.72rem; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 1.9rem; font-weight: 700; line-height: 1; color: {C['text']}; }}
  .kpi-sub   {{ font-size: 0.75rem; color: {C['muted']}; margin-top: 5px; }}

  .sec-hdr {{
      color: {C['text']};
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      border-left: 3px solid {C['primary']};
      padding-left: 10px;
      margin: 1.2rem 0 0.8rem;
  }}
  .sec-hdr.warn  {{ border-color: {C['accent']}; }}
  .sec-hdr.green {{ border-color: {C['success']}; }}
  .sec-hdr.red   {{ border-color: {C['danger']}; }}

  .metric-band {{
      display: flex;
      gap: 10px;
      padding: 0.9rem 1.2rem;
      background: rgba(56,189,248,0.07);
      border: 1px solid rgba(56,189,248,0.22);
      border-radius: 10px;
      flex-wrap: wrap;
      margin-bottom: 1rem;
      align-items: center;
  }}
  .metric-pill {{
      padding: 4px 14px;
      background: rgba(56,189,248,0.13);
      border-radius: 20px;
      font-size: 0.82rem;
      color: {C['primary']};
      font-weight: 600;
  }}
  .metric-pill span {{ color: {C['text']}; }}

  .pred-high {{
      background: linear-gradient(135deg, rgba(248,113,113,0.18), rgba(212,169,106,0.08));
      border: 1px solid {C['danger']};
      border-radius: 14px;
      padding: 1.6rem 2rem;
      text-align: center;
  }}
  .pred-low {{
      background: linear-gradient(135deg, rgba(52,211,153,0.14), rgba(56,189,248,0.07));
      border: 1px solid {C['success']};
      border-radius: 14px;
      padding: 1.6rem 2rem;
      text-align: center;
  }}
  .pred-title {{ font-size: 1.2rem; font-weight: 700; margin-bottom: 6px; }}
  .pred-pct   {{ font-size: 2.6rem; font-weight: 800; line-height: 1; }}
  .pred-note  {{ font-size: 0.8rem; color: {C['muted']}; margin-top: 8px; }}

  .alert-box {{
      background: rgba(248,113,113,0.10);
      border: 1px solid {C['danger']};
      border-left: 5px solid {C['danger']};
      border-radius: 10px;
      padding: 0.9rem 1.3rem;
      margin-bottom: 1.2rem;
  }}
  .alert-box.warn {{
      background: rgba(251,191,36,0.08);
      border-color: {C['warn']};
      border-left-color: {C['warn']};
  }}
  .alert-box.info {{
      background: rgba(56,189,248,0.07);
      border-color: rgba(56,189,248,0.35);
      border-left-color: {C['primary']};
      color: {C['primary']};
  }}

  .live-badge {{
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: {C['success']};
      margin-right: 7px;
      animation: blink 1.4s infinite;
  }}
  @keyframes blink {{
      0%, 100% {{ opacity: 1; }}
      50%       {{ opacity: 0.25; }}
  }}

  .empty-state {{
      height: 260px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px dashed {C['border']};
      border-radius: 14px;
      color: {C['muted']};
      text-align: center;
  }}

  .stButton > button {{
      background: linear-gradient(135deg, {C['primary']}, {C['sec']});
      color: {C['surface']};
      border: none;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.88rem;
      padding: 0.55rem 1.2rem;
      transition: opacity 0.15s;
  }}
  .stButton > button:hover {{ opacity: 0.88; }}
  .stSelectbox label, .stSlider label, .stNumberInput label,
  .stFileUploader label {{ color: {C['muted']}; font-size: 0.83rem; font-weight: 500; }}
  div[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
  hr {{ border-color: {C['border']}; margin: 1.2rem 0; }}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT SYNTHETIC DATASET  (fallback when no CSV is uploaded)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def make_default_dataset(n: int = 4_000, seed: int = 0) -> pd.DataFrame:
    """
    Generic synthetic dataset with a realistic mix of numeric and categorical
    columns plus a binary target. Intentionally domain-neutral.
    """
    rng = np.random.default_rng(seed)

    regions    = rng.choice(["North", "South", "East", "West", "Central"], size=n,
                            p=[0.20, 0.18, 0.22, 0.25, 0.15])
    channels   = rng.choice(["Online", "Partner", "Direct", "Referral"],   size=n,
                            p=[0.35, 0.25, 0.28, 0.12])
    categories = rng.choice(["Premium", "Standard", "Economy"],            size=n,
                            p=[0.20, 0.55, 0.25])
    segments   = rng.choice(["Enterprise", "SMB", "Startup", "Consumer"],  size=n,
                            p=[0.15, 0.30, 0.25, 0.30])
    status     = rng.choice(["Active", "Inactive"],                        size=n,
                            p=[0.70, 0.30])

    value    = np.clip(rng.normal(520,  180, n), 50,   2_000).round(2)
    duration = np.clip(rng.normal(14.5, 7,   n), 1,    60).round(1)
    score    = np.clip(rng.normal(68,   18,  n), 0,    100).round(1)
    quantity = rng.integers(1, 50, size=n).astype(float)
    discount = np.clip(rng.exponential(0.12, n), 0, 0.60).round(3)

    log_odds = (
        -1.2
        + 0.003  * (value - 520)
        - 0.015  * score
        + 0.05   * duration
        + np.where(channels   == "Online",   0.5, 0.0)
        + np.where(categories == "Economy",  0.7, 0.0)
        + np.where(segments   == "Startup",  0.4, 0.0)
        + rng.normal(0, 0.7, n)
    )
    prob   = 1 / (1 + np.exp(-log_odds))
    target = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "region":   regions,
        "channel":  channels,
        "category": categories,
        "segment":  segments,
        "status":   status,
        "value":    value,
        "duration": duration,
        "score":    score,
        "quantity": quantity,
        "discount": discount,
        "target":   target,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV, trying common encodings in order."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    raise ValueError("Could not decode the CSV with utf-8, latin-1, or cp1252.")


def classify_columns(df: pd.DataFrame, target_col: str):
    """
    Returns:
        num_cols  – numeric columns excluding target
        cat_cols  – object / category / bool columns excluding target
        all_feat  – num_cols + cat_cols (the full feature set)
    """
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c != target_col
    ]
    return num_cols, cat_cols, num_cols + cat_cols


def smart_format(val: float) -> str:
    """Return a compact, human-readable representation of a number."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:.1f}K"
    if abs(val) == int(abs(val)):
        return f"{int(val):,}"
    return f"{val:,.2f}"


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOMATED ML PIPELINE  (cached per unique data + target combination)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_pipeline(df_hash: int, df: pd.DataFrame, target_col: str):
    """
    Fully automated pipeline:
      1.  Separate features and target.
      2.  Impute missing values — median for numeric, mode for categorical.
      3.  LabelEncode every categorical feature column automatically.
      4.  Binarise the target (keeps top-2 classes when multi-class).
      5.  Train GradientBoostingClassifier.
      6.  Return all artefacts required for prediction and visualisation.

    Returns a 8-tuple:
      (clf, feat_encoders, target_le, metrics, importances, feat_cols, col_meta, work_df)
      All items are None-filled if training is not possible.
    """
    _empty = (None,) * 8

    # 1. Classify columns and clean data
    num_cols, cat_cols, feat_cols = classify_columns(df, target_col)
    if not feat_cols:
        return _empty

    work = df[feat_cols + [target_col]].copy()

    # Impute missing values
    for col in num_cols:
        work[col] = work[col].fillna(work[col].median() if not pd.isna(work[col].median()) else 0)
    for col in cat_cols:
        work[col] = work[col].fillna(work[col].mode()[0] if not work[col].mode().empty else "Unknown")

    X = work[feat_cols].copy()
    y = work[target_col].copy()

    # 2. Encode categorical features
    feat_encoders = {}
    for col in cat_cols:
        from sklearn.preprocessing import LabelEncoder
        le_f = LabelEncoder()
        X[col] = le_f.fit_transform(X[col].astype(str))
        feat_encoders[col] = le_f

    # 3. 🔥 Auto-Detect Task: Regression vs Classification
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    is_regression = y.dtype in [np.float64, np.float32] or y.nunique() > (len(y) * 0.4)
    target_le = None

    if is_regression:
        y = y.fillna(y.mean() if not pd.isna(y.mean()) else 0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        clf = GradientBoostingRegressor(n_estimators=50, random_state=42, min_samples_split=2)

        clf.fit(X_tr, y_tr)

        model_metrics = {
            "Train Score": float(clf.score(X_tr, y_tr)),
            "Test Score": float(clf.score(X_te, y_te))
        }
    else:
        y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=50, random_state=42, min_samples_split=2)

        clf.fit(X_tr, y_tr)

        y_pred_te = clf.predict(X_te)

        model_metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred_te)),
            "precision": float(precision_score(y_te, y_pred_te, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_te, y_pred_te, average='weighted')),
            "f1": float(f1_score(y_te, y_pred_te, average='weighted')),
            "roc_auc": 0.0,
            "Train Score": float(clf.score(X_tr, y_tr)),
            "Test Score": float(clf.score(X_te, y_te))
        }
    importances = list(clf.feature_importances_) if hasattr(clf, "feature_importances_") else [0] * len(feat_cols)

    # Per-column metadata for dynamic predictor form (Dashboard feature retained)
    col_meta = {}
    for col in num_cols:
        if col in work.columns:
            col_meta[col] = {
                "type": "numeric",
                "min": float(work[col].min()),
                "max": float(work[col].max()),
                "mean": float(work[col].mean()),
            }
    for col in cat_cols:
        col_meta[col] = {
            "type": "categorical",
            "classes": list(feat_encoders[col].classes_),
        }

    work_df = work.copy()

    return clf, feat_encoders, target_le, model_metrics, importances, feat_cols, col_meta, work_df


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:1.2rem 0 1.5rem;">
            <div style="font-size:2.2rem;">⚡</div>
            <div style="font-size:1rem; font-weight:700; color:{C['text']};
                        letter-spacing:0.01em;">AutoML Platform</div>
            <div style="font-size:0.7rem; color:{C['muted']}; letter-spacing:0.12em;
                        text-transform:uppercase; margin-top:2px;">
                No-Code Intelligence Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CSV uploader ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{C["primary"]}; font-size:0.78rem; font-weight:600;'
        f' text-transform:uppercase; letter-spacing:0.09em; margin-bottom:6px;">'
        f'📂 Data Source</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help=(
            "Upload any CSV. The platform will auto-detect column types, "
            "train a GradientBoosting model, and generate all visuals dynamically."
        ),
    )

    using_default = False
    if uploaded_file is not None:
        try:
            raw_df = load_csv(uploaded_file)
            st.success(
                f"✅  **{uploaded_file.name}**\n\n"
                f"{len(raw_df):,} rows × {len(raw_df.columns)} columns"
            )
        except Exception as exc:
            st.error(f"❌  Could not read file: {exc}")
            raw_df        = make_default_dataset()
            using_default = True
    else:
        raw_df        = make_default_dataset()
        using_default = True

    if using_default:
        st.markdown(
            f'<div class="alert-box info" style="font-size:0.79rem;">'
            f'ℹ️ Demo dataset active (4,000 rows). '
            f'Upload any CSV above to replace it.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Target variable selector ──────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{C["primary"]}; font-size:0.78rem; font-weight:600;'
        f' text-transform:uppercase; letter-spacing:0.09em; margin-bottom:6px;">'
        f'🎯 ML Target Variable</div>',
        unsafe_allow_html=True,
    )
    all_cols = raw_df.columns.tolist()

    # Heuristic default: prefer columns named target/label/class/churn etc.
    heuristic_candidates = [
        c for c in all_cols
        if any(
            kw in c.lower()
            for kw in ("target", "label", "class", "churn", "cancel",
                       "outcome", "flag", "result", "default", "fraud")
        )
    ]
    default_idx = (
        all_cols.index(heuristic_candidates[0])
        if heuristic_candidates
        else len(all_cols) - 1
    )

    target_col = st.selectbox(
        "Select target column (Y)",
        options=all_cols,
        index=default_idx,
        help="The column the ML model will learn to predict.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Simulator settings ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:{C["primary"]}; font-size:0.78rem; font-weight:600;'
        f' text-transform:uppercase; letter-spacing:0.09em; margin-bottom:6px;">'
        f'🔁 Simulator Settings</div>',
        unsafe_allow_html=True,
    )
    sim_chunk      = st.slider("Chunk Size (rows)", 5, 100, 20)
    sim_interval   = st.slider("Refresh Interval (sec)", 1, 10, 3)
    risk_threshold = st.slider("High-Risk Threshold (%)", 10, 90, 50)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        f"**Rows**: {len(raw_df):,}  \n"
        f"**Columns**: {len(raw_df.columns)}  \n"
        f"**Target**: `{target_col}`  \n"
        f"**Platform**: AutoML v2.0"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN MODEL  (re-runs only when data or target column changes)
# ══════════════════════════════════════════════════════════════════════════════

# Use a lightweight hash so the cache key survives serialisation
df_hash = hash(raw_df.to_json())

with st.spinner("🔧  Auto-engineering features and training model …"):
    (
        clf,
        feat_encoders,
        target_le,
        model_metrics,
        importances,
        feat_cols,
        col_meta,
        work_df,
    ) = build_pipeline(df_hash, raw_df, target_col)

model_ready = clf is not None

# Column classification (used throughout all tabs)
num_cols, cat_cols, _ = classify_columns(raw_df, target_col)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL HEADER
# ══════════════════════════════════════════════════════════════════════════════

status_badge = (
    f'<span style="background:rgba(52,211,153,0.18); color:{C["success"]};'
    f' border:1px solid {C["success"]}; border-radius:20px; padding:3px 12px;'
    f' font-size:0.72rem; font-weight:700; letter-spacing:0.05em;">MODEL READY</span>'
    if model_ready else
    f'<span style="background:rgba(248,113,113,0.15); color:{C["danger"]};'
    f' border:1px solid {C["danger"]}; border-radius:20px; padding:3px 12px;'
    f' font-size:0.72rem; font-weight:700; letter-spacing:0.05em;">NEEDS MORE DATA</span>'
)

st.markdown(
    f"""
    <div style="display:flex; align-items:baseline; gap:14px;
                margin-bottom:0.2rem; flex-wrap:wrap;">
        <span style="font-size:1.7rem; font-weight:800; color:{C['text']};">
            Universal AutoML Platform
        </span>
        {status_badge}
    </div>
    <div style="color:{C['muted']}; font-size:0.86rem; margin-bottom:1.4rem;">
        Automated Analytics &nbsp;·&nbsp; Dynamic ML Pipeline &nbsp;·&nbsp;
        Real-Time Ingestion Simulator &nbsp;·&nbsp;
        <span style="color:{C['accent']};">
            {len(raw_df):,} rows &nbsp;·&nbsp; {len(raw_df.columns)} columns
            &nbsp;·&nbsp; Target:
            <code style="background:rgba(56,189,248,0.12); color:{C['primary']};
                         padding:1px 6px; border-radius:4px;">{target_col}</code>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊  Executive Dashboard",
    "🤖  ML Prediction Pipeline",
    "🔁  Live Ingestion Simulator",
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — AUTOMATED EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Dataset summary strip ─────────────────────────────────────────────────
    n_rows, n_cols   = raw_df.shape
    n_numeric_all    = len(raw_df.select_dtypes(include=np.number).columns)
    n_categorical_all = len(raw_df.select_dtypes(include=["object", "category"]).columns)
    missing_pct      = raw_df.isnull().mean().mean() * 100

    st.markdown(
        f"""
        <div class="metric-band">
            <div class="metric-pill">Rows <span>{n_rows:,}</span></div>
            <div class="metric-pill">Columns <span>{n_cols}</span></div>
            <div class="metric-pill">Numeric <span>{n_numeric_all}</span></div>
            <div class="metric-pill">Categorical <span>{n_categorical_all}</span></div>
            <div class="metric-pill">Missing Values <span>{missing_pct:.1f}%</span></div>
            <div class="metric-pill">Target <span>{target_col}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards – top 4 numeric columns ────────────────────────────────────
    all_numeric_cols = [c for c in raw_df.select_dtypes(include=np.number).columns]
    kpi_num_cols = all_numeric_cols[:4]
    icons = ["📈", "📉", "🔢", "💡"]
    card_classes = ["c0", "c1", "c2", "c3"]

    if kpi_num_cols:
        kpi_cols = st.columns(4)
        for i, col in enumerate(kpi_num_cols):
            with kpi_cols[i]:
                series = raw_df[col].dropna()
                mean_v = series.mean()
                sum_v = series.sum()
                std_v = series.std()

                card_html = f"""
                    <div class="kpi-card {card_classes[i]}">
                        <div class="kpi-icon">{icons[i]}</div>
                        <div class="kpi-label">{col}</div>
                        <div class="kpi-value">{smart_format(mean_v)}</div>
                        <div class="kpi-sub">
                            Mean &nbsp;·&nbsp; Sum {smart_format(sum_v)}
                            &nbsp;·&nbsp; σ {smart_format(std_v)}
                        </div>
                    </div>
                    """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("No numeric columns detected for KPI cards.")

    # ── Categorical distributions ─────────────────────────────────────────────
    dash_cat_cols = [
        c for c in raw_df.select_dtypes(include=["object", "category", "bool"]).columns
        if c != target_col
    ]

    if dash_cat_cols:
        st.markdown(
            '<div class="sec-hdr">Categorical Column Distributions</div>',
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(dash_cat_cols), 2):
            pair = dash_cat_cols[row_start: row_start + 2]
            cols = st.columns(len(pair))
            for ci, col in enumerate(pair):
                with cols[ci]:
                    vc = raw_df[col].value_counts().head(12).reset_index()
                    vc.columns = ["category", "count"]
                    fig = go.Figure(go.Bar(
                        x=vc["category"],
                        y=vc["count"],
                        marker=dict(
                            color=vc["count"],
                            colorscale=[[0, C["surf3"]], [1, C["primary"]]],
                            line_width=0,
                        ),
                        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
                    ))
                    themed(fig, f"{col} — Value Counts", height=290)
                    fig.update_xaxes(tickangle=-30 if len(vc) > 6 else 0)
                    st.plotly_chart(fig, use_container_width=True)

    # ── Numeric distributions – histograms ───────────────────────────────────
    dash_num_cols = [c for c in num_cols][:6]

    if dash_num_cols:
        st.markdown(
            '<div class="sec-hdr">Numeric Feature Distributions</div>',
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(dash_num_cols), 3):
            trio = dash_num_cols[row_start: row_start + 3]
            cols = st.columns(len(trio))
            for ci, col in enumerate(trio):
                with cols[ci]:
                    series = raw_df[col].dropna()
                    fig    = go.Figure(go.Histogram(
                        x=series,
                        nbinsx=35,
                        marker_color=C["secondary"] if "secondary" in C else C["sec"],
                        marker_line_width=0,
                        opacity=0.85,
                        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
                    ))
                    fig.add_vline(
                        x=float(series.mean()),
                        line_dash="dash",
                        line_color=C["accent"],
                        annotation_text=f"μ={series.mean():.2g}",
                        annotation_font_color=C["accent"],
                        annotation_position="top right",
                    )
                    themed(fig, col, height=250)
                    st.plotly_chart(fig, use_container_width=True)

    # ── Target variable class balance ─────────────────────────────────────────
    st.markdown(
        '<div class="sec-hdr">Target Variable — Class Balance</div>',
        unsafe_allow_html=True,
    )
    bal_c1, bal_c2 = st.columns([1, 2])

    with bal_c1:
        vc_target = raw_df[target_col].value_counts().reset_index()
        vc_target.columns = ["class", "count"]
        fig_donut = go.Figure(go.Pie(
            labels=vc_target["class"].astype(str),
            values=vc_target["count"],
            hole=0.58,
            marker_colors=[
                C["primary"], C["accent"], C["success"], C["danger"], C["warn"]
            ],
            textinfo="label+percent",
            textfont=dict(color=C["text"], size=11),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        ))
        themed(fig_donut, f"Target: {target_col}", height=300)
        fig_donut.update_layout(showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with bal_c2:
        if dash_cat_cols:
            pivot_col = dash_cat_cols[0]
            cross     = pd.crosstab(raw_df[pivot_col], raw_df[target_col])
            cross.index   = cross.index.astype(str).str[:18]
            cross.columns = cross.columns.astype(str)

            fig_stack = go.Figure()
            palette   = [C["primary"], C["accent"], C["success"], C["danger"], C["warn"]]
            for i, cls in enumerate(cross.columns):
                fig_stack.add_trace(go.Bar(
                    name=f"Class: {cls}",
                    x=cross.index,
                    y=cross[cls],
                    marker_color=palette[i % len(palette)],
                    marker_line_width=0,
                ))
            themed(fig_stack, f"{target_col} by {pivot_col}", height=300)
            fig_stack.update_layout(barmode="stack")
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.info("No categorical columns available for cross-tabulation.")

    # ── Pearson correlation heatmap ───────────────────────────────────────────
    corr_candidates = [c for c in raw_df.select_dtypes(include=np.number).columns][:12]
    if len(corr_candidates) >= 2:
        st.markdown(
            '<div class="sec-hdr">Numeric Correlation Matrix</div>',
            unsafe_allow_html=True,
        )
        corr_matrix = raw_df[corr_candidates].corr()
        fig_corr    = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[
                [0,   C["danger"]],
                [0.5, C["surface"]],
                [1,   C["primary"]],
            ],
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=9, color=C["text"]),
            showscale=True,
            colorbar=dict(tickfont=dict(color=C["muted"])),
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        ))
        themed(
            fig_corr,
            "Pearson Correlation Matrix",
            height=max(280, len(corr_candidates) * 38),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Raw data preview ──────────────────────────────────────────────────────
    with st.expander("🗂  Raw Data Preview (first 100 rows)"):
        st.dataframe(raw_df.head(100), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — GENERALISED ML PREDICTION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

with tab2:

    if not model_ready:
        st.markdown(
            f"""
            <div class="alert-box">
                <strong style="color:{C['danger']};">⚠ Model could not be trained.</strong><br>
                Possible causes: only one class in target column, no usable feature columns,
                or dataset is too small after filtering. Try a different target column or
                upload a richer dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Model performance banner ──────────────────────────────────────────────
    target_classes = target_le.classes_ if target_le is not None else ["0", "1"]

    if 'accuracy' in model_metrics:
        st.markdown(f"""
        <div class="metric-band">
            <div style="font-size:0.82rem; color:{C['muted']}; margin-right:4px;">
                GradientBoostingClassifier &nbsp;&middot;&nbsp; Classes:&nbsp;
                <code style="color:{C['primary']};">{list(target_classes)}</code>
            </div>
            <div class="metric-pill">
                Accuracy <span>{model_metrics['accuracy']:.1%}</span>
            </div>
            <div class="metric-pill">
                Precision <span>{model_metrics['precision']:.1%}</span>
            </div>
            <div class="metric-pill">
                Recall <span>{model_metrics['recall']:.1%}</span>
            </div>
            <div class="metric-pill">
                F1 <span>{model_metrics['f1']:.1%}</span>
            </div>
            <div class="metric-pill">
                AUC-ROC <span>{model_metrics['roc_auc']:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-band">
            <div style="font-size:0.82rem; color:{C['muted']}; margin-right:4px;">
                GradientBoostingRegressor &nbsp;&middot;&nbsp; Continuous Target
            </div>
            <div class="metric-pill">
                Train Score (R²) <span>{model_metrics['Train Score']:.2f}</span>
            </div>
            <div class="metric-pill">
                Test Score (R²) <span>{model_metrics['Test Score']:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    ml_left, ml_right = st.columns([2, 3])

    # ── Dynamic predictor form ────────────────────────────────────────────────
    with ml_left:
        st.markdown(
            '<div class="sec-hdr">⚙ Live Predictor — Auto-Generated Input Form</div>',
            unsafe_allow_html=True,
        )

        if not feat_cols:
            st.warning("No feature columns are available.")
        else:
            input_values: dict = {}
            # Cap form to 12 features to maintain usability
            visible_feats = feat_cols[:12]
            overflow      = len(feat_cols) - len(visible_feats)

            for feat in visible_feats:
                meta = col_meta.get(feat, {})

                if meta.get("type") == "numeric":
                    lo      = meta["min"]
                    hi      = meta["max"]
                    default = float(np.clip(meta["mean"], lo, hi))
                    step    = max(round((hi - lo) / 100, 4), 0.0001)
                    val = st.slider(
                        feat,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=round(default, 4),
                        step=step,
                        format="%.4g",
                    )
                    input_values[feat] = val

                elif meta.get("type") == "categorical":
                    classes = meta["classes"]
                    val = st.selectbox(feat, options=classes)
                    input_values[feat] = val

                else:
                    val = st.text_input(feat, value="")
                    input_values[feat] = val

            if overflow > 0:
                st.caption(
                    f"ℹ +{overflow} additional features are used by the model; "
                    f"their values are fixed at the training-set mean / mode."
                )

            predict_btn = st.button(
                "🔮  Predict Now", use_container_width=True, type="primary"
            )

    # ── Prediction result + feature importance ────────────────────────────────
    with ml_right:

        if predict_btn and feat_cols:

            # Build a complete feature row; unseen features use mean/first-class
            row: dict = {}
            for feat in feat_cols:
                meta    = col_meta.get(feat, {})
                raw_val = input_values.get(feat)

                if raw_val is None:
                    # Feature not in visible form → impute from metadata
                    if meta.get("type") == "numeric":
                        raw_val = meta.get("mean", 0.0)
                    else:
                        classes = meta.get("classes", [""])
                        raw_val = classes[0] if classes else ""

                if meta.get("type") == "categorical":
                    le = feat_encoders.get(feat)
                    if le is not None:
                        sv = str(raw_val)
                        if sv in le.classes_:
                            row[feat] = float(le.transform([sv])[0])
                        else:
                            row[feat] = 0.0
                    else:
                        row[feat] = 0.0
                else:
                    try:
                        row[feat] = float(raw_val)
                    except (ValueError, TypeError):
                        row[feat] = 0.0

            X_input = pd.DataFrame([row])[feat_cols]

            if isinstance(clf, GradientBoostingRegressor):
                pred_val = clf.predict(X_input)[0]
                outcome = f"📊 Predicted Value — {pred_val:.2f}"
                advice = f"The model predicts the continuous outcome to be approximately {pred_val:.2f} units."
                result_cls = "pred-low"
                prob_display = f"{pred_val:.1f}"
                sub_note = "Predicted Regression Value"
                color = C["success"]
            else:
                prob = clf.predict_proba(X_input)[0, 1]
                label = clf.predict(X_input)[0]

                if target_le is not None:
                    label_str = str(target_le.inverse_transform([label])[0])
                    pos_class = str(target_le.classes_[-1])
                else:
                    label_str = str(label)
                    pos_class = "1"

                is_high = prob >= (risk_threshold / 100)
                color = C["danger"] if is_high else C["success"]
                result_cls = "pred-high" if is_high else "pred-low"
                outcome = f"⚠️ HIGH RISK — {label_str}" if is_high else f"✅ LOW RISK — {label_str}"
                advice = "Predicted positive / high-risk class. Consider targeted intervention or review." if is_high else "Predicted negative / low-risk class. No immediate action required."
                prob_display = f"{prob:.1%}"
                sub_note = f"Probability of positive class ({pos_class})"

            st.markdown(
                f"""
                            <div class="{result_cls}">
                                <div class="pred-title" style="color:{color};">{outcome}</div>
                                <div class="pred-pct"   style="color:{color};">{prob_display}</div>
                                <div class="pred-note">{sub_note}</div>
                                <div style="margin-top:10px; font-size:0.82rem; color:{C['muted']};">{advice}</div>
                            </div>
                            """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            if type(clf).__name__ != "GradientBoostingRegressor":
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={
                        "reference": risk_threshold,
                        "valueformat": ".1f",
                        "suffix": "%",
                        "increasing": {"color": C["danger"]},
                        "decreasing": {"color": C["success"]},
                    },
                    number={"suffix": "%", "font": {"size": 38, "color": C["text"]}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "ticksuffix": "%",
                            "tickfont": {"color": C["muted"]},
                        },
                        "bar": {"color": color},
                        "bgcolor": C["surface"],
                        "bordercolor": C["border"],
                        "steps": [
                            {
                                "range": [0, risk_threshold],
                                "color": "rgba(52,211,153,0.10)",
                            },
                            {
                                "range": [risk_threshold, 100],
                                "color": "rgba(248,113,113,0.12)",
                            },
                        ],
                        "threshold": {
                            "line": {"color": C["accent"], "width": 2},
                            "thickness": 0.75,
                            "value": risk_threshold,
                        },
                    },
                ))
                themed(fig_gauge, "Positive-Class Probability", height=270)
                st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown(
                f"""
                <div class="empty-state">
                    <div>
                        <div style="font-size:2.8rem; margin-bottom:10px;">🎯</div>
                        <div style="font-weight:600;">Configure inputs on the left</div>
                        <div style="font-size:0.84rem; margin-top:4px; color:{C['muted']};">
                            then click
                            <strong style="color:{C['primary']};">🔮 Predict Now</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Feature importance bar chart (always visible) ─────────────────────
        st.markdown(
            '<div class="sec-hdr green" style="margin-top:1.6rem;">'
            'Feature Importance (Gini Impurity)</div>',
            unsafe_allow_html=True,
        )
        if isinstance(importances, list):
            imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
        else:
            imp_df = importances.reset_index()
            imp_df.columns = ["feature", "importance"]

        fig_imp = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale=[[0, C["sec"]], [1, C["primary"]]],
                line_width=0,
            ),
            text=[f"{v:.4f}" for v in imp_df["importance"]],
            textposition="outside",
            textfont=dict(color=C["muted"], size=10),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        themed(fig_imp, "", height=max(260, len(imp_df) * 28))
        fig_imp.update_xaxes(showgrid=False)
        st.plotly_chart(fig_imp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LIVE INGESTION SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

with tab3:

    st.markdown(
        f"""
        <div class="alert-box info" style="font-size:0.83rem;">
            <strong>How it works:</strong>&nbsp; The simulator sequentially pulls chunks of rows
            from the loaded dataset, scores each chunk through the trained model in real time,
            appends results to a live scrolling table, and plots the risk rate per batch.
            Use the sidebar controls to tune chunk size, refresh speed, and the alert threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session state initialisation ──────────────────────────────────────────
    if "sim_running"   not in st.session_state:
        st.session_state.sim_running   = False
    if "sim_cursor"    not in st.session_state:
        st.session_state.sim_cursor    = 0
    if "sim_results"   not in st.session_state:
        st.session_state.sim_results   = pd.DataFrame()
    if "sim_batch_log" not in st.session_state:
        st.session_state.sim_batch_log = []

    # ── Control buttons ───────────────────────────────────────────────────────
    ctl1, ctl2, ctl3, _ = st.columns([1, 1, 1, 4])
    with ctl1:
        if st.button("▶  Start", use_container_width=True, type="primary"):
            st.session_state.sim_running = True
    with ctl2:
        if st.button("⏹  Stop", use_container_width=True):
            st.session_state.sim_running = False
    with ctl3:
        if st.button("🗑  Reset", use_container_width=True):
            st.session_state.sim_running   = False
            st.session_state.sim_cursor    = 0
            st.session_state.sim_results   = pd.DataFrame()
            st.session_state.sim_batch_log = []

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chunk scoring function ────────────────────────────────────────────────
    def score_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run one dataframe chunk through the trained pipeline.
        Appends two columns to the returned copy:
          _risk_prob  – float probability of the positive class (0–100)
          _risk_label – str  "HIGH" | "LOW"
        """
        if not model_ready or not feat_cols:
            out                = chunk_df.copy()
            out["_risk_prob"]  = np.nan
            out["_risk_label"] = "N/A"
            return out

        work = chunk_df.copy()

        # Numeric imputation
        for col in num_cols:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
                col_median = work[col].median()
                work[col]  = work[col].fillna(0.0 if pd.isna(col_median) else col_median)

        # Categorical imputation + encoding
        for col in cat_cols:
            if col in work.columns:
                mode_vals  = work[col].mode()
                fill_val   = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else ""
                work[col]  = work[col].fillna(fill_val).astype(str)
                le         = feat_encoders.get(col)
                if le is not None:
                    known      = set(le.classes_)
                    work[col]  = work[col].apply(
                        lambda v: float(le.transform([v])[0]) if v in known else 0.0
                    )
                else:
                    work[col] = 0.0

        # Ensure all feature columns exist; fill absent ones with 0
        for fc in feat_cols:
            if fc not in work.columns:
                work[fc] = 0.0

        X = work[feat_cols].values.astype(float)

        try:
            probs = clf.predict_proba(X)[:, 1]
        except Exception:
            probs = np.full(len(X), np.nan)

        threshold = risk_threshold / 100
        out                = chunk_df.copy()
        out["_risk_prob"]  = (probs * 100).round(1)
        out["_risk_label"] = np.where(probs >= threshold, "HIGH", "LOW")
        return out

    # ── Ingest one chunk if simulator is running ──────────────────────────────
    if st.session_state.sim_running and model_ready:
        cursor  = st.session_state.sim_cursor
        total   = len(raw_df)
        end_idx = min(cursor + sim_chunk, total)

        # Wrap around to the beginning when the dataset is exhausted
        if cursor >= total:
            st.session_state.sim_cursor = 0
            cursor  = 0
            end_idx = min(sim_chunk, total)

        chunk                = raw_df.iloc[cursor:end_idx].copy()
        chunk["_batch_ts"]   = datetime.now().strftime("%H:%M:%S")
        scored               = score_chunk(chunk)

        # Keep the live table bounded to the last 500 rows
        st.session_state.sim_results = pd.concat(
            [st.session_state.sim_results, scored],
            ignore_index=True,
        ).tail(500)

        high_count = (scored["_risk_label"] == "HIGH").sum()
        high_rate  = (high_count / len(scored) * 100) if len(scored) > 0 else 0.0

        st.session_state.sim_batch_log.append({
            "batch":      len(st.session_state.sim_batch_log) + 1,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "rows":       len(scored),
            "high_count": int(high_count),
            "high_rate":  round(high_rate, 1),
            "alert":      high_rate >= risk_threshold,
        })

        st.session_state.sim_cursor = end_idx if end_idx < total else 0

    # ── Per-batch alert banner ────────────────────────────────────────────────
    if st.session_state.sim_batch_log:
        last = st.session_state.sim_batch_log[-1]
        if last["alert"]:
            st.markdown(
                f"""
                <div class="alert-box">
                    <strong style="color:{C['danger']};">
                        🚨 ALERT — Batch #{last['batch']}
                    </strong><br>
                    High-risk rate
                    <strong style="color:{C['danger']};">{last['high_rate']:.1f}%</strong>
                    exceeds the configured threshold of
                    <strong>{risk_threshold}%</strong>.
                    &nbsp; {last['high_count']} of {last['rows']} records flagged.
                    &nbsp; Timestamp: {last['timestamp']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Live aggregate KPIs ───────────────────────────────────────────────────
    if not st.session_state.sim_results.empty:
        results        = st.session_state.sim_results
        total_scored   = len(results)
        high_total     = (results["_risk_label"] == "HIGH").sum()
        high_rate_agg  = (high_total / total_scored * 100) if total_scored > 0 else 0.0
        batches_done   = len(st.session_state.sim_batch_log)

        sm1, sm2, sm3, sm4 = st.columns(4)
        live_kpis = [
            (sm1, "Records Scored",    f"{total_scored:,}",       C["primary"]),
            (sm2, "High-Risk Records", f"{high_total:,}",         C["danger"]),
            (sm3, "Overall Risk Rate", f"{high_rate_agg:.1f}%",   C["accent"]),
            (sm4, "Batches Processed", f"{batches_done}",         C["success"]),
        ]
        for col_obj, label, value, color in live_kpis:
            with col_obj:
                st.markdown(
                    f"""
                    <div style="background:{C['surface']}; border:1px solid {C['border']};
                                border-radius:10px; padding:1rem 1.2rem; text-align:center;">
                        <div style="font-size:0.72rem; color:{C['muted']};
                                    text-transform:uppercase; letter-spacing:0.08em;
                                    margin-bottom:4px;">{label}</div>
                        <div style="font-size:1.7rem; font-weight:700;
                                    color:{color};">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ────────────────────────────────────────────────────────
        sim_c1, sim_c2 = st.columns([3, 2])

        with sim_c1:
            log_df   = pd.DataFrame(st.session_state.sim_batch_log)
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=log_df["batch"].astype(str),
                y=log_df["high_rate"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.07)",
                line=dict(color=C["primary"], width=2.5),
                marker=dict(
                    color=[C["danger"] if a else C["success"] for a in log_df["alert"]],
                    size=9,
                    line=dict(color=C["text"], width=1),
                ),
                name="High-Risk %",
                hovertemplate="Batch %{x}: %{y:.1f}%<extra></extra>",
            ))
            fig_live.add_hline(
                y=risk_threshold,
                line_dash="dash",
                line_color=C["accent"],
                annotation_text=f"Alert threshold {risk_threshold}%",
                annotation_font_color=C["accent"],
            )
            themed(fig_live, "High-Risk Rate per Batch (%)", height=300)
            fig_live.update_xaxes(title_text="Batch #")
            fig_live.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_live, use_container_width=True)

        with sim_c2:
            lbl_counts = results["_risk_label"].value_counts().reset_index()
            lbl_counts.columns = ["label", "count"]
            fig_donut2 = go.Figure(go.Pie(
                labels=lbl_counts["label"],
                values=lbl_counts["count"],
                hole=0.55,
                marker_colors=[
                    C["danger"] if lbl == "HIGH" else C["success"]
                    for lbl in lbl_counts["label"]
                ],
                textinfo="label+percent",
                textfont=dict(color=C["text"], size=11),
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            ))
            themed(fig_donut2, "Risk Label Split", height=300)
            fig_donut2.update_layout(showlegend=False)
            st.plotly_chart(fig_donut2, use_container_width=True)

        # ── Live scrolling scored table ───────────────────────────────────────
        st.markdown(
            f'<div class="sec-hdr" style="margin-top:0.4rem;">'
            f'<span class="live-badge"></span>'
            f'Live Scored Records — last 100</div>',
            unsafe_allow_html=True,
        )

        display_df = results.tail(100).copy()

        # Bring prediction columns to the front for readability
        front  = ["_batch_ts", "_risk_label", "_risk_prob"]
        others = [c for c in display_df.columns if c not in front]
        display_df = display_df[front + others].rename(columns={
            "_batch_ts":    "Time",
            "_risk_label":  "Risk Label",
            "_risk_prob":   "Risk Prob %",
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk Prob %": st.column_config.ProgressColumn(
                    "Risk Prob %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

    else:
        st.markdown(
            f"""
            <div class="empty-state">
                <div>
                    <div style="font-size:3rem; margin-bottom:10px;">🔁</div>
                    <div style="font-size:1.05rem; font-weight:600;">
                        Simulator Inactive
                    </div>
                    <div style="font-size:0.86rem; margin-top:5px;">
                        Press
                        <strong style="color:{C['primary']};">▶ Start</strong>
                        to begin live batch ingestion
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Auto-rerun loop when simulator is active ──────────────────────────────
    if st.session_state.sim_running:
        time.sleep(sim_interval)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

perf_line = "Universal ML Engine Active"
if 'model_ready' in globals() and model_ready:
    if 'model_metrics' in globals() and model_metrics is not None:
        if 'accuracy' in model_metrics:
            perf_line = f"GradientBoostingClassifier &nbsp;·&nbsp; AUC-ROC: {model_metrics.get('roc_auc', 0.0):.3f} &nbsp;·&nbsp; Accuracy: {model_metrics.get('accuracy', 0.0):.3f}"
        elif 'Test Score' in model_metrics:
            perf_line = f"GradientBoostingRegressor &nbsp;·&nbsp; Train R²: {model_metrics.get('Train Score', 0.0):.3f} &nbsp;·&nbsp; Test R²: {model_metrics.get('Test Score', 0.0):.3f}"

st.markdown(
    f"""
    <div style="text-align:center; padding:1.8rem 0 1rem;
                border-top:1px solid {C['border']}; margin-top:2rem;">
        <span style="color:{C['muted']}; font-size:0.78rem; letter-spacing:0.02em;">
            <strong>Universal AutoML Platform</strong> &nbsp;·&nbsp; 
            Engineered & Developed by <strong style="color:{C['primary']};">Manjeet Kumar</strong> &nbsp;·&nbsp; 
            {perf_line} &nbsp;·&nbsp; 
            Built with Streamlit · Plotly · scikit-learn
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)