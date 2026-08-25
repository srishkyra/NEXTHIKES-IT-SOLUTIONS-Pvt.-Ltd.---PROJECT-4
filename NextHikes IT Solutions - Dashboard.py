"""
Mobile Phone Price Prediction — Interactive Dashboard
NextHikes IT Solutions
"""

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + PALETTE
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile Price Prediction Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDIGO = "#211A4D"
TEAL = "#527783"
BERRY = "#AE2C59"
GOLD = "#F5C18A"
CORAL = "#E97F70"
CREAM = "#F7E5D2"
CATEGORY_PALETTE = [BERRY, TEAL, GOLD, INDIGO, CORAL, "#9C9BC3", "#91AFC8", "#6B4A6F"]

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F8F9FA; }}
    h1, h2, h3 {{ color: {INDIGO} !important; }}
    div[data-testid="stMetricValue"] {{ color: {INDIGO}; font-weight: 700; }}
    div[data-testid="stMetricLabel"] {{ color: #666666; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #F0F0F0; border-radius: 6px 6px 0 0; padding: 8px 14px; font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {INDIGO}; color: white; }}
    section[data-testid="stSidebar"] {{ background-color: #F8F9FA; }}
    </style>
    """,
    unsafe_allow_html=True,
)

LAYOUT_KWARGS = dict(
    paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
    font=dict(color=INDIGO, family="sans-serif"),
    title_font=dict(size=16, color=INDIGO),
    colorway=CATEGORY_PALETTE,
    margin=dict(t=60, b=40, l=40, r=20),
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "Processed_Flipdata.xlsx"
FEATURE_IMPORTANCE_PATH = APP_DIR / "feature_importance.json"
RANDOM_STATE = 42

# Every hardcoded metric, feature-importance value, and business-question figure in this
# dashboard was traced to a dataset of this size (see report/notebook). The model below
# retrains live on whatever DATA_PATH contains, but the hardcoded numbers do NOT — if the
# loaded row count drifts from this, those numbers are stale until re-verified.
EXPECTED_DATASET_ROWS = 531

# Raw columns the pipeline depends on (checked before any transformation runs,
# so a schema drift in the source file fails loudly and specifically instead
# of raising a confusing KeyError deep inside load_and_clean_data).
REQUIRED_RAW_COLUMNS = [
    "Model", "Colour", "Memory", "RAM", "Battery_",
    "Rear Camera", "Front Camera", "Mobile Height", "Processor_",
]

# ──────────────────────────────────────────────────────────────────────────
# VERIFIED MODEL RESULTS — single source of truth for every metric shown
# anywhere in the dashboard. Values traced to the project report except
# where flagged verified=False.
# ──────────────────────────────────────────────────────────────────────────
MODEL_RESULTS = {
    "Naive (mean)":              {"test_r2": -0.0003, "test_mae": 7040, "oof_r2": -0.0280, "verified": True},
    "Linear Regression":         {"test_r2": 0.5113,  "test_mae": 3499, "oof_r2": 0.3743,  "verified": True},
    "Ridge":                     {"test_r2": 0.6238,  "test_mae": 3001, "oof_r2": 0.5115,  "verified": True},
    "Lasso":                     {"test_r2": 0.6875,  "test_mae": 2743, "oof_r2": 0.4986,  "verified": True},
    "XGBoost (tuned)":           {"test_r2": 0.6440,  "test_mae": 3337, "oof_r2": 0.6123,  "verified": True},
    "Gradient Boosting (tuned)": {"test_r2": 0.7225,  "test_mae": 2968, "oof_r2": 0.5947,  "verified": True},
    "Random Forest (tuned)":     {"test_r2": 0.7507,  "test_mae": 3206, "oof_r2": 0.5898,  "verified": True},
}
FINAL_MODEL = "Random Forest (tuned)"
# Separate, full-dataset (531-row) grouped-CV stability check for the final model.
# Deliberately NOT the same figure as MODEL_RESULTS[...]["oof_r2"], which is a
# 428-row training-subset pooled-OOF figure — the two use different samples and
# different aggregation and should never be merged into one number.
FULL_DATA_CV = {"mean_r2": 0.688, "lo": 0.582, "hi": 0.769}

# Repeated train/test-split stability check for the final model (notebook Section 9C:
# 5 GroupShuffleSplit train/test splits at different random seeds, same model config).
# Verified against the notebook's Section 9C output. Shown as a global error/variance
# reference, not a per-prediction interval — seed=42 is the single reported split (9A);
# the repeated-split figures describe the spread across five different splits.
REPEATED_SPLIT_MAE = {
    "held_out": 3206,               # seed=42 (9A canonical result), reused in 9C
    "repeated_mean": 3676,          # mean MAE across 5 seeds [1, 2, 3, 42, 99]
    "repeated_mae_std": 448,        # std of MAE across those 5 seeds
    "repeated_r2_mean": 0.5836,     # mean test R² across the same 5 seeds
    "repeated_r2_std": 0.1183,      # std of test R² across the same 5 seeds
    "verified_against_report": True,
}

MODELS_EVALUATED_TOTAL = 9  # report states nine algorithms were compared; only the
                             # principal candidates below are surfaced as a leaderboard

# Centralized fallback for grouped permutation importance, used whenever
# feature_importance.json isn't present next to app.py — kept here (not just
# inline in a function) so the dashboard never hard-depends on the external
# file to show its verified reference numbers.
FEATURE_IMPORTANCE_FALLBACK = {
    "Front_Camera_MP": 0.4495, "Battery_Level": 0.1796, "Memory": 0.1440,
    "Brand": 0.0196, "RAM": 0.0094, "Processor_Brand": 0.0055,
    "Base_Colour": 0.0032, "is_flagged_dq": 0.0023, "Is_5G": 0.0021,
    "Rear_Camera_MP": -0.0170, "Screen_Size_cm": -0.0221,
}

# ──────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (mirrors the verified notebook pipeline — Sections 1–3)
# ──────────────────────────────────────────────────────────────────────────
COLOUR_KEYWORDS = [
    ("Black", r"black|midnight|obsidian|night|dark matter|shadow"),
    ("White", r"white|chalk|snow|starlight|ice|frost(?!ed blue)"),
    ("Blue", r"blue|sea|ocean|sky|aqua(?!marine)"),
    ("Green", r"green|lime|olive|rainforest|jade|vert"),
    ("Red", r"red|maroon|crimson"),
    ("Gray", r"gray|grey|charcoal|graphite|slate"),
    ("Silver", r"silver"),
    ("Gold", r"gold"),
    ("Purple", r"purple|violet|lavender"),
    ("Pink", r"pink|rose|magenta"),
    ("Orange", r"orange|amber|peach"),
    ("Yellow", r"yellow|lemon"),
    ("Brown", r"brown|beige|tan"),
    ("Cyan", r"cyan|teal|aquamarine|turquoise"),
    ("Copper", r"copper|bronze"),
]

CHIP_KEYWORDS = [
    ("Qualcomm", r"qualcomm|snapdragon|sm6|sm7|sm8"),
    ("MediaTek", r"mediatek|meditek|mtk|helio|dimensity|mt6|^g\d{2,3}$"),
    ("Samsung Exynos", r"exynos|s5e"),
    ("Unisoc", r"unisoc|spreadtrum|sc98|sc65|tiger"),
    ("Google", r"tensor"),
    ("Apple", r"bionic|apple"),
]

FEATURES_NUM = ["Memory", "RAM", "Battery_Level", "Rear_Camera_MP", "Front_Camera_MP", "Screen_Size_cm"]
FEATURES_CAT = ["Brand", "Processor_Brand", "Base_Colour", "Is_5G", "is_flagged_dq"]


def _base_colour(name: str) -> str:
    n = str(name).lower()
    for label, pat in COLOUR_KEYWORDS:
        if re.search(pat, n):
            return label
    return "Other"


def _proc_brand(text: str) -> str:
    t = str(text).strip().lower()
    for label, pat in CHIP_KEYWORDS:
        if re.search(pat, t):
            return label
    return "Unknown"


@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean_data(path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Safe handling of either raw column name — the source file has used both
    # "Prize" (typo) and "Price" across different export runs.
    if "Prize" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"Prize": "Price"})

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if "Price" not in df.columns:
        missing.append("Price (or Prize)")
    if missing:
        raise ValueError(
            "Source file is missing required column(s): " + ", ".join(missing)
        )

    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    # Flag the 15-listing data-quality cluster before transforming it
    cluster_mask = (df["Mobile Height"] >= 4.5) & (df["Mobile Height"] <= 7.11)
    df["is_flagged_dq"] = cluster_mask
    df = df.rename(columns={"Mobile Height": "Screen_Size_cm"})
    df.loc[cluster_mask, "Screen_Size_cm"] *= 2.54
    moto_mask = (df["Model"] == "MOTOROLA G62 5G") & (df["Screen_Size_cm"] > 20)
    df.loc[moto_mask, "Screen_Size_cm"] = 16.64

    # Camera columns: extract the first numeric MP value, tolerating blanks, decimals,
    # "N/A", stray whitespace, and multi-camera strings like "50MP + 8MP" (first value wins,
    # matching the source data's convention of listing the primary sensor first).
    for col in ["Rear Camera", "Front Camera"]:
        extracted = df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df[col] = pd.to_numeric(extracted, errors="coerce").replace(0, np.nan)
    df = df.rename(columns={
        "Rear Camera": "Rear_Camera_MP", "Front Camera": "Front_Camera_MP",
        "Battery_": "Battery_Level", "Processor_": "Processor_Type",
    })

    # Brand extraction (needed before camera imputation, which is brand-median based)
    brand_raw = df["Model"].str.split().str[0].str.upper()
    df["Brand"] = brand_raw.replace({"MICROMAX1": "MICROMAX", "I": "IKALL"})

    for col in ["Rear_Camera_MP", "Front_Camera_MP"]:
        df[col] = df[col].fillna(df.groupby("Brand")[col].transform("median"))

    df["Is_5G"] = df["Model"].str.contains("5G", case=False, na=False)

    df["Base_Colour"] = df["Colour"].apply(_base_colour)
    small_buckets = df["Base_Colour"].value_counts()
    df["Base_Colour"] = df["Base_Colour"].replace(
        {c: "Other" for c in small_buckets[small_buckets < 10].index}
    )

    df["Processor_Brand"] = df["Processor_Type"].apply(_proc_brand)
    apple_fallback = (df["Processor_Brand"] == "Unknown") & (df["Brand"] == "APPLE")
    df.loc[apple_fallback, "Processor_Brand"] = "Apple"

    df["Total_Camera_MP"] = df["Rear_Camera_MP"] + df["Front_Camera_MP"]

    tier_bounds = df["Price"].quantile([0.25, 0.5, 0.75]).values

    def assign_tier(p):
        if p <= tier_bounds[0]:
            return "Budget"
        elif p <= tier_bounds[1]:
            return "Mid"
        elif p <= tier_bounds[2]:
            return "Premium"
        return "Flagship"

    df["Price_Tier"] = df["Price"].apply(assign_tier)
    return df


@st.cache_resource(show_spinner="Training pricing model...")
def train_model(df: pd.DataFrame):
    X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT, drop_first=True).astype(float)
    y = df["Price"]
    model = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def build_prediction_row(inputs: dict, feature_cols: list) -> pd.DataFrame:
    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    row["Memory"] = inputs["memory"]
    row["RAM"] = inputs["ram"]
    row["Battery_Level"] = inputs["battery"]
    row["Rear_Camera_MP"] = inputs["rear_mp"]
    row["Front_Camera_MP"] = inputs["front_mp"]
    row["Screen_Size_cm"] = inputs["screen"]
    brand_col = f"Brand_{inputs['brand']}"
    proc_col = f"Processor_Brand_{inputs['processor']}"
    colour_col = f"Base_Colour_{inputs['colour']}"
    if brand_col in row.columns:
        row[brand_col] = 1
    if proc_col in row.columns:
        row[proc_col] = 1
    if colour_col in row.columns:
        row[colour_col] = 1
    if inputs["is_5g"] and "Is_5G_True" in row.columns:
        row["Is_5G_True"] = 1
    return row


def load_feature_importance() -> pd.Series:
    """Loads grouped permutation importance from feature_importance.json (notebook
    Section 10C export) when present, so a fresher export is always picked up first.
    Falls back to FEATURE_IMPORTANCE_FALLBACK — the app's own bundled, verified
    values — so the dashboard never depends on the external file being present."""
    try:
        with open(FEATURE_IMPORTANCE_PATH) as f:
            data = json.load(f)
        return pd.Series(data["values"]).sort_values()
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        st.caption(
            f"ℹ️ {FEATURE_IMPORTANCE_PATH.name} not found next to app.py — using the "
            "dashboard's bundled reference values instead."
        )
        return pd.Series(FEATURE_IMPORTANCE_FALLBACK).sort_values()


def typical_error_range(model, X_row):
    """Per-prediction spread across the forest's individual trees.
    Labeled 'disagreement range', not a formal prediction interval (informal
    uncertainty signal from tree-to-tree variance, not conformal prediction).
    Also returns spread as a % of the mean tree prediction, used to help
    flag low-confidence predictions in the Live Price Predictor tab."""
    tree_preds = np.array([t.predict(X_row.values) for t in model.estimators_]).flatten()
    lo, hi = np.percentile(tree_preds, 5), np.percentile(tree_preds, 95)
    mean_pred = tree_preds.mean()
    spread_pct = (tree_preds.std() / abs(mean_pred) * 100) if mean_pred else 0.0
    return lo, hi, spread_pct


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA + MODEL — every stage guarded so a bad file or environment issue
# surfaces a specific, actionable message instead of crashing the app.
# ──────────────────────────────────────────────────────────────────────────
try:
    df = load_and_clean_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find **{DATA_PATH.name}**. Place `Processed_Flipdata.xlsx` in the same "
        "folder as this script and rerun `streamlit run app.py`."
    )
    st.stop()
except ImportError:
    st.error(
        "⚠️ Missing Excel engine — reading `.xlsx` files requires the `openpyxl` package. "
        "Install it with `pip install openpyxl` and rerun `streamlit run app.py`."
    )
    st.stop()
except ValueError as e:
    st.error(f"⚠️ Data schema problem: {e}")
    st.stop()

if len(df) != EXPECTED_DATASET_ROWS:
    st.warning(
        f"⚠️ Loaded dataset has {len(df):,} rows, but every verified metric, feature-importance "
        f"value, and business-question figure in this dashboard was traced to a "
        f"{EXPECTED_DATASET_ROWS}-row dataset. The model above has retrained on the data actually "
        "loaded, but the hardcoded reference numbers throughout the dashboard have NOT been "
        "recomputed — treat them as stale until re-verified against this new data."
    )

try:
    model, feature_cols = train_model(df)
except Exception as e:
    st.error(f"⚠️ Model training failed: {e}")
    st.stop()

FILTER_NOTE = (
    "Filters below apply to the exploratory tabs (Overview through Outliers & Data Quality). "
    "Model Performance, Feature Importance, and the Live Price Predictor always reflect the "
    "model trained on the full dataset, so those figures don't change with filter selection."
)

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS (apply to exploratory tabs) + RESET / DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📱 Mobile Price Prediction")
st.sidebar.markdown("**NextHikes IT Solutions**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

FILTER_KEYS = ["f_brand", "f_price", "f_memory", "f_ram", "f_proc", "f_5g", "f_tier", "f_dq"]


def _reset_all_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


brand_options = sorted(df["Brand"].unique())
memory_options = sorted(df["Memory"].unique())
ram_options = sorted(df["RAM"].unique())
proc_options = sorted(df["Processor_Brand"].unique())
tier_options = ["Budget", "Mid", "Premium", "Flagship"]
price_min, price_max = int(df["Price"].min()), int(df["Price"].max())

sel_brand = st.sidebar.multiselect("Brand", brand_options, default=[], key="f_brand")
sel_price = st.sidebar.slider("Price Range (₹)", price_min, price_max, (price_min, price_max), key="f_price")
sel_memory = st.sidebar.multiselect("Memory (GB)", memory_options, default=[], key="f_memory")
sel_ram = st.sidebar.multiselect("RAM (GB)", ram_options, default=[], key="f_ram")
sel_proc = st.sidebar.multiselect("Processor Brand", proc_options, default=[], key="f_proc")
sel_5g = st.sidebar.selectbox("5G", ["All", "5G Only", "Non-5G Only"], key="f_5g")
sel_tier = st.sidebar.multiselect("Price Tier", tier_options, default=[], key="f_tier")
sel_dq = st.sidebar.selectbox(
    "Data-Quality Cluster", ["All", "Flagged Only", "Exclude Flagged"], key="f_dq",
    help="The flagged cluster (~3% of listings) is the hardest-to-predict segment — "
         "see Key Insights & Limitations.",
)
st.sidebar.button("↺ Reset All Filters", on_click=_reset_all_filters, width="stretch", key="btn_reset_filters")

filtered = df.copy()
if sel_brand:
    filtered = filtered[filtered["Brand"].isin(sel_brand)]
filtered = filtered[filtered["Price"].between(*sel_price)]
if sel_memory:
    filtered = filtered[filtered["Memory"].isin(sel_memory)]
if sel_ram:
    filtered = filtered[filtered["RAM"].isin(sel_ram)]
if sel_proc:
    filtered = filtered[filtered["Processor_Brand"].isin(sel_proc)]
if sel_5g == "5G Only":
    filtered = filtered[filtered["Is_5G"]]
elif sel_5g == "Non-5G Only":
    filtered = filtered[~filtered["Is_5G"]]
if sel_tier:
    filtered = filtered[filtered["Price_Tier"].isin(sel_tier)]
if sel_dq == "Flagged Only":
    filtered = filtered[filtered["is_flagged_dq"]]
elif sel_dq == "Exclude Flagged":
    filtered = filtered[~filtered["is_flagged_dq"]]

st.sidebar.download_button(
    "⬇️ Download Filtered Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_mobile_listings.csv",
    mime="text/csv",
    width="stretch",
    key="btn_download_filtered",
)
st.sidebar.caption(f"{len(filtered):,} of {len(df):,} listings match current filters")
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} listings · {df['Model'].nunique()} unique models")
st.sidebar.caption(f"Final model: {FINAL_MODEL} · Test R²={MODEL_RESULTS[FINAL_MODEL]['test_r2']:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# MAIN — TITLE + TABS
# ──────────────────────────────────────────────────────────────────────────
st.title("📱 Mobile Phone Price Prediction Dashboard")
st.caption("NextHikes IT Solutions — interactive analysis and pricing-sanity tool")

(
    tab_overview, tab_uni, tab_bi, tab_multi, tab_outliers,
    tab_perf, tab_importance, tab_predictor, tab_questions, tab_insights,
) = st.tabs([
    "🏠 Overview", "📈 Univariate Analysis", "🔗 Bivariate & Correlation",
    "🧬 Multivariate Analysis", "🧹 Outliers & Data Quality",
    "🎯 Model Performance", "🌟 Feature Importance",
    "💰 Live Price Predictor", "❓ Business Questions", "💡 Key Insights & Limitations",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.caption(FILTER_NOTE)
    st.markdown(
        "Predictive analysis of mobile phone pricing from device specifications. "
        "Price here is market-driven, not formula-derived — the tabs on this page surface "
        "which specs are associated with price and let you price a hypothetical device live."
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Listings (filtered)", f"{len(filtered):,}")
    c2.metric("Unique Models", f"{filtered['Model'].nunique():,}")
    c3.metric("Average Price", f"₹{filtered['Price'].mean():,.0f}" if len(filtered) else "—")
    c4.metric("Held-out Test R²", f"{MODEL_RESULTS[FINAL_MODEL]['test_r2']:.3f}")
    c5.metric("Grouped-CV Mean R²", f"{FULL_DATA_CV['mean_r2']:.3f}")
    st.caption(
        f"Grouped-CV range ({FINAL_MODEL}, full 531-row dataset): "
        f"{FULL_DATA_CV['lo']:.3f}–{FULL_DATA_CV['hi']:.3f}  |  "
        f"Held-out MAE: ₹{MODEL_RESULTS[FINAL_MODEL]['test_mae']:,}  |  "
        f"Naive MAE: ₹{MODEL_RESULTS['Naive (mean)']['test_mae']:,}"
    )

    if filtered.empty:
        st.warning("No listings match the current filters. Adjust filters in the sidebar or click Reset All Filters.")
    else:
        st.markdown("### Price Distribution")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(
                filtered, x="Price", nbins=40, color_discrete_sequence=[TEAL],
                labels={"Price": "Price (₹)"}, marginal="box", title="Price Distribution",
            )
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False, height=380,
                               xaxis_title="Price (₹)", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch", key="ov_price_hist")
        with col2:
            tier_counts = filtered["Price_Tier"].value_counts().reindex(tier_options).fillna(0)
            fig2 = px.pie(
                values=tier_counts.values, names=tier_counts.index,
                color_discrete_sequence=[TEAL, GOLD, CORAL, BERRY], hole=0.45, title="Listings by Price Tier",
            )
            fig2.update_layout(**LAYOUT_KWARGS, height=380, showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig2, width="stretch", key="ov_tier_pie")

    st.markdown("### The Central Finding")
    st.info(
        "Two pricing regimes are visible in this market. Below the flagship tier, price is "
        "associated with specifications — storage, RAM, and camera specs explain most of the "
        "variation, and brand carries little independent pricing signal once specs are known. "
        "At the flagship tier, a premium is associated with Apple and Google specifically (small "
        "samples in this dataset, n=2–9 — directional, not a precise estimate) that no spec "
        "combination fully explains — the model's largest errors concentrate exactly there."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB — UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_uni:
    st.subheader("📈 Univariate Analysis")
    st.caption(FILTER_NOTE)
    if filtered.empty:
        st.warning("No listings match the current filters.")
    else:
        num_choice = st.selectbox("Numeric feature", ["Price"] + FEATURES_NUM, index=0, key="uni_num_choice")
        fig = px.histogram(
            filtered, x=num_choice, nbins=35, color_discrete_sequence=[TEAL],
            marginal="box", title=f"Distribution of {num_choice}",
        )
        fig.update_layout(**LAYOUT_KWARGS, height=420, showlegend=False)
        st.plotly_chart(fig, width="stretch", key="uni_hist")

        cat_choice = st.selectbox(
            "Categorical feature", ["Brand", "Processor_Brand", "Base_Colour", "Is_5G", "Price_Tier"],
            index=0, key="uni_cat_choice",
        )
        counts = filtered[cat_choice].value_counts().head(15)
        fig2 = px.bar(
            x=counts.values, y=counts.index.astype(str), orientation="h",
            color_discrete_sequence=[BERRY], labels={"x": "Count", "y": ""},
            title=f"{cat_choice} — Listing Counts",
        )
        fig2.update_layout(**LAYOUT_KWARGS, height=420)
        st.plotly_chart(fig2, width="stretch", key="uni_bar")

# ══════════════════════════════════════════════════════════════════════════
# TAB — BIVARIATE & CORRELATION
# ══════════════════════════════════════════════════════════════════════════
with tab_bi:
    st.subheader("🔗 Bivariate & Correlation")
    st.caption(FILTER_NOTE)
    if filtered.empty:
        st.warning("No listings match the current filters.")
    else:
        st.markdown("### Correlation with Price")
        corr_data = filtered[FEATURES_NUM + ["Price"]].corr(numeric_only=True)["Price"].drop("Price").sort_values()
        fig = px.bar(
            x=corr_data.values, y=corr_data.index, orientation="h",
            color=corr_data.values, color_continuous_scale=[BERRY, "white", TEAL],
            labels={"x": "Pearson r with Price", "y": ""}, title="Correlation with Price",
        )
        fig.update_layout(**LAYOUT_KWARGS, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", key="bi_corr_bar")
        st.caption(
            "Memory and RAM are typically the strongest linear predictors. Front-camera MP shows "
            "a genuinely non-linear relationship — its monotonic correlation is notably stronger "
            "than the linear figure shown here."
        )

        st.markdown("### Correlation Heatmap")
        heatmap_data = filtered[FEATURES_NUM + ["Price"]].corr(numeric_only=True)
        fig2 = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.columns,
                colorscale=[[0, BERRY], [0.5, "white"], [1, TEAL]], zmid=0,
                text=heatmap_data.round(2).values, texttemplate="%{text}",
            )
        )
        fig2.update_layout(**LAYOUT_KWARGS, height=440, title="Correlation Matrix — Numeric Features")
        st.plotly_chart(fig2, width="stretch", key="bi_heatmap")

        st.markdown("### Scatter Explorer")
        cx, cy, cc = st.columns(3)
        x_axis = cx.selectbox("X-axis", FEATURES_NUM, index=0, key="bi_x")
        y_axis = cy.selectbox("Y-axis", ["Price"] + FEATURES_NUM, index=0, key="bi_y")
        color_by = cc.selectbox("Color by", ["Brand", "Processor_Brand", "Price_Tier", "Is_5G"], index=2, key="bi_color")
        fig3 = px.scatter(
            filtered, x=x_axis, y=y_axis, color=color_by, hover_data=["Model", "Price"],
            color_discrete_sequence=CATEGORY_PALETTE, opacity=0.7,
            title=f"{y_axis} vs. {x_axis}, coloured by {color_by}",
        )
        fig3.update_layout(**LAYOUT_KWARGS, height=460)
        st.plotly_chart(fig3, width="stretch", key="bi_scatter")

# ══════════════════════════════════════════════════════════════════════════
# TAB — MULTIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.subheader("🧬 Multivariate Analysis")
    st.caption(FILTER_NOTE)
    if filtered.empty:
        st.warning("No listings match the current filters.")
    else:
        st.markdown("### Memory × RAM × Price, by Tier")
        fig = px.scatter_3d(
            filtered, x="Memory", y="RAM", z="Price", color="Price_Tier",
            color_discrete_sequence=[TEAL, GOLD, CORAL, BERRY],
            hover_data=["Model", "Brand"], opacity=0.75,
        )
        fig.update_layout(**LAYOUT_KWARGS, height=520, scene=dict(
            xaxis_title="Memory (GB)", yaxis_title="RAM (GB)", zaxis_title="Price (₹)"
        ))
        st.plotly_chart(fig, width="stretch", key="multi_3d")

        st.markdown("### Price by Brand and 5G Status (Top 10 Brands by Listing Count)")
        top_brands = filtered["Brand"].value_counts().head(10).index
        fig2 = px.box(
            filtered[filtered["Brand"].isin(top_brands)], x="Brand", y="Price", color="Is_5G",
            color_discrete_sequence=[TEAL, BERRY],
            title="Price Distribution by Brand, split by 5G",
        )
        fig2.update_layout(**LAYOUT_KWARGS, height=460, yaxis_title="Price (₹)")
        st.plotly_chart(fig2, width="stretch", key="multi_box")

# ══════════════════════════════════════════════════════════════════════════
# TAB — OUTLIERS & DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════
with tab_outliers:
    st.subheader("🧹 Outliers & Data Quality")
    st.caption(FILTER_NOTE)
    if filtered.empty:
        st.warning("No listings match the current filters.")
    else:
        st.markdown("### IQR Outlier Counts by Numeric Feature")
        rows = []
        for col in FEATURES_NUM:
            s = filtered[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((s < lo) | (s > hi)).sum()
            rows.append({"Feature": col, "IQR Lower": round(lo, 1), "IQR Upper": round(hi, 1),
                         "Outliers (count)": int(n_out), "Outliers (%)": round(100 * n_out / len(s), 1)})
        st.dataframe(pd.DataFrame(rows), width="stretch")

        st.markdown("### Boxplots — Visual Outlier Check")
        box_feature = st.selectbox("Feature", FEATURES_NUM, index=0, key="out_box_feature")
        fig = px.box(filtered, y=box_feature, color_discrete_sequence=[TEAL], points="outliers",
                     title=f"{box_feature} — Boxplot")
        fig.update_layout(**LAYOUT_KWARGS, height=400)
        st.plotly_chart(fig, width="stretch", key="out_boxplot")

        st.markdown("### Flagged Data-Quality Cluster")
        st.caption(
            "15 listings (2.8% of the full dataset) had a screen-size entry stored in "
            "inconsistent units and were explicitly corrected during cleaning. They're flagged "
            "rather than dropped, and remain the hardest-to-predict segment for the model."
        )
        dq_rows = filtered[filtered["is_flagged_dq"]]
        if dq_rows.empty:
            st.info("No flagged data-quality rows in the current filter selection.")
        else:
            st.dataframe(
                dq_rows[["Model", "Brand", "Screen_Size_cm", "Price", "Price_Tier"]],
                width="stretch", height=250,
            )

        st.markdown("### Missing-Value Completeness Check")
        na_pct = (filtered[FEATURES_NUM + FEATURES_CAT].isna().mean() * 100).round(2)
        st.dataframe(na_pct.rename("Missing (%)").to_frame(), width="stretch")

# ══════════════════════════════════════════════════════════════════════════
# TAB — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("🎯 Model Performance")
    st.caption(
        "Nine algorithms were compared with group-aware cross-validation (the same phone model "
        "never appears in both train and test splits). The held-out test set was used for final "
        "comparison among a small, pre-selected shortlist — not for hyperparameter search — "
        "which introduces mild selection optimism in the test figures below."
    )

    leaderboard = pd.DataFrame([
        {
            "Model": name,
            "Test R²": vals["test_r2"],
            "Test MAE (₹)": vals["test_mae"],
            "Training-Subset Pooled OOF R²": vals["oof_r2"],
            "OOF Verified": "✅" if vals["verified"] else "⚠️ unverified",
        }
        for name, vals in MODEL_RESULTS.items()
    ])

    fig = px.bar(
        leaderboard, x="Model", y="Test R²", color="Test R²",
        color_continuous_scale=[BERRY, GOLD, TEAL], text="Test R²", title="Regression Leaderboard — Test R²",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(**LAYOUT_KWARGS, height=420, coloraxis_showscale=False, yaxis_range=[-0.1, 0.85])
    st.plotly_chart(fig, width="stretch", key="perf_bar")

    st.dataframe(
        leaderboard.style.format({
            "Test R²": "{:.4f}", "Test MAE (₹)": "₹{:,.0f}", "Training-Subset Pooled OOF R²": "{:.4f}",
        }),
        width="stretch",
    )
    st.caption(
        "Ridge's Training-Subset Pooled OOF R² (0.5115) is sourced from the notebook's Section 8D "
        "GroupKFold CV leaderboard (Ridge, α=12.743) and verified against it, same as every other "
        "row's OOF figure."
    )
    st.caption(
        f"Nine algorithms were evaluated in total; the {len(MODEL_RESULTS)} shown above are the "
        "principal candidates carried through to the leaderboard — the remaining models were not "
        "competitive enough to report individually."
    )
    st.info(
        "**Training-Subset Pooled OOF R²** (428-row subset, computed during model selection) and "
        f"**full-dataset Grouped-CV mean R² ({FULL_DATA_CV['mean_r2']:.3f}, range "
        f"{FULL_DATA_CV['lo']:.3f}–{FULL_DATA_CV['hi']:.3f})** are two different metrics on two "
        "different samples — they should never be read as the same number."
    )
    st.success(
        f"**{FINAL_MODEL} retained as the final model**, selected after comparing the leading "
        "CV-shortlisted candidates on the held-out test split. XGBoost led the training-subset "
        f"pooled-OOF ranking (R²={MODEL_RESULTS['XGBoost (tuned)']['oof_r2']:.3f}), while "
        f"{FINAL_MODEL} achieved the highest held-out test R² "
        f"({MODEL_RESULTS[FINAL_MODEL]['test_r2']:.3f}) — XGBoost's CV lead did not hold on the "
        f"test split (test R²={MODEL_RESULTS['XGBoost (tuned)']['test_r2']:.3f}). A separate "
        f"full-dataset GroupKFold stability check produced a mean R² of {FULL_DATA_CV['mean_r2']:.3f} "
        f"(range {FULL_DATA_CV['lo']:.3f}–{FULL_DATA_CV['hi']:.3f}). Note that the held-out test set "
        "was reused for this later diagnostic comparison rather than for hyperparameter search, so "
        "the test-split ranking above may carry mild selection optimism — treat it as directional, "
        "not a fully independent confirmation."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════
IMPURITY_IMPORTANCE = {
    "Front_Camera_MP": 0.4919, "Memory": 0.1169, "Battery_Level": 0.0751,
    "RAM": 0.0734, "Brand": 0.0658, "Rear_Camera_MP": 0.0509,
    "Processor_Brand": 0.0494, "Is_5G": 0.0339, "Screen_Size_cm": 0.0284,
    "Base_Colour": 0.0089, "is_flagged_dq": 0.0055,
}  # Notebook Section 10A: built-in Random Forest regression impurity importance, aggregated to the parent feature.

with tab_importance:
    st.subheader("🌟 Feature Importance")

    st.markdown("#### Impurity-Based Importance (Training Subset — Section 10A)")
    impurity_series = pd.Series(IMPURITY_IMPORTANCE).sort_values()
    fig_imp = px.bar(
        x=impurity_series.values, y=impurity_series.index, orientation="h",
                color=impurity_series.values, color_continuous_scale=[TEAL, GOLD, BERRY],
        labels={"x": "Mean decrease in regression impurity", "y": ""},
        title="Built-in Regression Impurity Importance",
    )
    fig_imp.update_layout(**LAYOUT_KWARGS, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, width="stretch", key="imp_impurity_bar")
    st.caption(
        "Calculated on the 428-row training subset used to fit the notebook's model (Section 10A), "
        "not the full 531-row dataset. The top three features (Front_Camera_MP, Memory, "
        "Battery_Level) account for **68.4%** of total impurity importance."
    )

    st.markdown("#### Grouped Permutation Importance (Held-Out Test Set — Section 10B/10C)")
    st.caption("Joint-shuffle (grouped permutation) importance — dummy columns for each categorical feature are shuffled together, not summed individually.")
    importance_data = load_feature_importance()
    fig = px.bar(
        x=importance_data.values, y=importance_data.index, orientation="h",
        color=importance_data.values, color_continuous_scale=[BERRY, GOLD, TEAL],
        labels={"x": "Mean R² drop when shuffled", "y": ""}, title="Grouped Permutation Importance",
    )
    fig.update_layout(**LAYOUT_KWARGS, height=440, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch", key="imp_bar")

    st.markdown("#### Why the Two Charts Disagree on Front_Camera_MP")
    st.info(
        "**Front_Camera_MP receives unusually high impurity importance (0.492) partly because "
        "of a project-specific artifact**: all five Apple rows in the full cleaned dataset "
        "near-unique `Front_Camera_MP=12` value, which coincides with the highest-priced brand — "
        "the tree can partly use this MP value as a proxy for \"this is an Apple phone,\" inflating "
        "its impurity score. The permutation calculation, by contrast, was run on the held-out "
        "test set, which contains **zero Apple rows** — so it's a useful robustness check on "
        "whether Front_Camera_MP's dominance survives without that specific artifact. It does "
        "survive (Front_Camera_MP still ranks 1st in test-set permutation importance), but the "
        "two methods are computed on different samples (428-row training subset vs. held-out test "
        "set, 103 rows) and shouldn't be read as two measurements of the same underlying quantity. The "
        "notebook triangulates further with SHAP (Section 10D–10F) rather than treating either "
        "impurity or permutation importance alone as ground truth."
    )
    st.info(
        "**Brand's bivariate signal mostly disappears in permutation importance** — its price "
        "signal is largely redundant with specs already in the model, not an independent lever. "
        "**Battery_Level ranks much higher here than in the raw bivariate correlation** — a "
        "relationship only visible once the flagged data-quality cluster is correctly handled."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB — LIVE PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tab_predictor:
    st.subheader("💰 Live Price Predictor")
    st.markdown(
        "Enter device specifications to get a model-estimated price. "
        "This is a **scenario estimate**, not a guaranteed market price."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Core Specs")
        memory = st.select_slider("Storage (GB)", options=sorted(df["Memory"].dropna().unique().astype(int)),
                                   value=128 if 128 in df["Memory"].values else int(df["Memory"].median()),
                                   key="pred_memory")
        ram = st.select_slider("RAM (GB)", options=sorted(df["RAM"].dropna().unique().astype(int)),
                                value=6 if 6 in df["RAM"].values else int(df["RAM"].median()),
                                key="pred_ram")
        batt_min, batt_max = int(df["Battery_Level"].min()), int(df["Battery_Level"].max())
        battery = st.slider("Battery (mAh)", batt_min, batt_max, min(5000, batt_max), step=100, key="pred_battery")
        scr_min, scr_max = float(df["Screen_Size_cm"].min()), float(df["Screen_Size_cm"].max())
        screen = st.slider("Screen Size (cm)", scr_min, scr_max, min(16.7, scr_max), step=0.1, key="pred_screen")
    with col2:
        st.markdown("#### Camera & Positioning")
        rear_mp = st.select_slider(
            "Rear Camera (MP)", options=sorted(df["Rear_Camera_MP"].dropna().unique().astype(int)),
            value=64 if 64 in df["Rear_Camera_MP"].values else int(df["Rear_Camera_MP"].median()),
            key="pred_rear_mp",
        )
        front_mp = st.select_slider(
            "Front Camera (MP)", options=sorted(df["Front_Camera_MP"].dropna().unique().astype(int)),
            value=16 if 16 in df["Front_Camera_MP"].values else int(df["Front_Camera_MP"].median()),
            key="pred_front_mp",
        )
        brand_list = sorted(df["Brand"].unique())
        default_brand = "REALME" if "REALME" in brand_list else brand_list[0]
        brand = st.selectbox("Brand", brand_list, index=brand_list.index(default_brand), key="pred_brand")
        processor = st.selectbox("Chipset Brand", sorted(df["Processor_Brand"].unique()), key="pred_processor")
        colour = st.selectbox("Colour Family", sorted(df["Base_Colour"].unique()), key="pred_colour")
        is_5g = st.checkbox("5G Enabled", value=True, key="pred_5g")

    st.markdown("---")

    if st.button("💰 Predict Price", type="primary", width="stretch", key="pred_button"):
        inputs = {
            "memory": memory, "ram": ram, "battery": battery, "rear_mp": rear_mp,
            "front_mp": front_mp, "screen": screen, "brand": brand,
            "processor": processor, "colour": colour, "is_5g": is_5g,
        }
        try:
            X_row = build_prediction_row(inputs, feature_cols)
            point_pred = model.predict(X_row)[0]
            lo, hi, spread_pct = typical_error_range(model, X_row)
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Price", f"₹{point_pred:,.0f}")
            r2.metric("Model Disagreement Range (5th–95th %ile of trees)", f"₹{lo:,.0f} – ₹{hi:,.0f}")
            r3.metric("Typical Absolute Error (held-out MAE)", f"₹{REPEATED_SPLIT_MAE['held_out']:,}")
            st.caption(
                f"Held-out test MAE (seed=42, Section 9A): ₹{REPEATED_SPLIT_MAE['held_out']:,} — "
                f"one estimate. Repeated-split mean MAE (5 seeds, Section 9C): "
                f"₹{REPEATED_SPLIT_MAE['repeated_mean']:,} (std=₹{REPEATED_SPLIT_MAE['repeated_mae_std']:,}) "
                f"— a separate estimate from a different check, not a range spanning the two. Report "
                f"R² similarly: mean={REPEATED_SPLIT_MAE['repeated_r2_mean']:.4f} "
                f"(std={REPEATED_SPLIT_MAE['repeated_r2_std']:.4f}) across the same 5 splits. These "
                "are global error references, not a calibrated per-prediction interval."
            )
            st.caption(
                "⚠️ The disagreement range reflects spread across the model's individual trees — an "
                "informal uncertainty signal, not a statistically calibrated prediction interval."
            )

            # ── Confidence checks ──
            brand_n = int((df["Brand"] == brand).sum())
            combo_n = int(((df["Memory"] == memory) & (df["RAM"] == ram)).sum())
            brand_proc_n = int(((df["Brand"] == brand) & (df["Processor_Brand"] == processor)).sum())
            in_range = (
                df["Battery_Level"].min() <= battery <= df["Battery_Level"].max()
                and df["Screen_Size_cm"].min() <= screen <= df["Screen_Size_cm"].max()
            )
            high_disagreement = spread_pct > 25  # trees disagree by >25% of the mean prediction

            reasons = []
            if brand_n < 10:
                reasons.append(f"{brand} has only {brand_n} listings in training data")
            if combo_n < 5:
                reasons.append(f"Memory={memory}GB/RAM={ram}GB combination has only {combo_n} training examples")
            if brand_proc_n < 5:
                reasons.append(f"{brand} × {processor} combination has only {brand_proc_n} training examples")
            if not in_range:
                reasons.append("Battery or screen size falls outside the observed training range — extrapolating")
            if high_disagreement:
                reasons.append(f"the forest's individual trees disagree by {spread_pct:.0f}% of the predicted price")

            if reasons:
                st.warning("⚠️ **Low Confidence** — " + "; ".join(reasons) + ". Treat this prediction as directional, not precise.")
            else:
                st.success(
                    f"✅ **Standard Confidence** — {brand} ({brand_n} listings), this Memory/RAM combo "
                    f"({combo_n} examples), and this Brand × Processor combo ({brand_proc_n} examples) "
                    "are all well represented, and inputs fall within the observed training range."
                )

            st.caption(
                "⚠️ The notebook's flagged data-quality cluster (~3% of listings) was the single "
                "hardest-to-predict segment, with error roughly 4–5× the dataset average — if this "
                "device resembles that cluster (unusual screen-size/height entry), treat the "
                "prediction with extra caution regardless of the confidence label above."
            )

    st.markdown("---")
    st.markdown("#### Quick Scenario Comparison")
    st.caption("Compare how the price changes if you flip one spec, holding everything else fixed.")
    scenario_col1, scenario_col2 = st.columns(2)
    with scenario_col1:
        if st.button("📶 What if this phone did NOT have 5G?", width="stretch", key="pred_scenario_5g"):
            try:
                base_inputs = {"memory": memory, "ram": ram, "battery": battery, "rear_mp": rear_mp,
                                "front_mp": front_mp, "screen": screen, "brand": brand,
                                "processor": processor, "colour": colour, "is_5g": False}
                alt_inputs = dict(base_inputs, is_5g=True)
                p_no5g = model.predict(build_prediction_row(base_inputs, feature_cols))[0]
                p_5g = model.predict(build_prediction_row(alt_inputs, feature_cols))[0]
                st.write(f"Without 5G: **₹{p_no5g:,.0f}** → With 5G: **₹{p_5g:,.0f}** "
                         f"(model-estimated scenario difference: ₹{p_5g - p_no5g:+,.0f}, {(p_5g/p_no5g-1)*100:+.1f}%)")
            except Exception as e:
                st.error(f"⚠️ Scenario prediction failed: {e}")
    with scenario_col2:
        if st.button("💾 What if storage doubled?", width="stretch", key="pred_scenario_storage"):
            try:
                doubled = min(memory * 2, int(df["Memory"].max()))
                base_inputs = {"memory": memory, "ram": ram, "battery": battery, "rear_mp": rear_mp,
                                "front_mp": front_mp, "screen": screen, "brand": brand,
                                "processor": processor, "colour": colour, "is_5g": is_5g}
                alt_inputs = dict(base_inputs, memory=doubled)
                p_base = model.predict(build_prediction_row(base_inputs, feature_cols))[0]
                p_alt = model.predict(build_prediction_row(alt_inputs, feature_cols))[0]
                st.write(f"{memory}GB: **₹{p_base:,.0f}** → {doubled}GB: **₹{p_alt:,.0f}** "
                         f"(model-estimated scenario difference: ₹{p_alt - p_base:+,.0f}, {(p_alt/p_base-1)*100:+.1f}%)")
            except Exception as e:
                st.error(f"⚠️ Scenario prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════
# TAB — BUSINESS QUESTIONS
# ══════════════════════════════════════════════════════════════════════════
with tab_questions:
    st.subheader("❓ Strategic Business Questions")
    st.caption(
        "Sourced from the notebook's Section 11A (Q1–Q10). Every figure below is read directly "
        "from results generated earlier in the notebook (Sections 5–10), not recomputed for this "
        "dashboard. Associations are reported as associations, not proven causal effects, and "
        "small-sample or sparse-table findings are flagged as directional rather than precise."
    )

    with st.expander("Q1. What features are most important for price prediction?", expanded=True):
        st.markdown(
            "Specifications account for **68.4%** of impurity-based feature importance "
            "(Front_Camera_MP + Memory + Battery_Level combined); Brand has relatively low "
            "marginal (permutation) importance (0.020) once specs are already in the model — its "
            "strong bivariate effect (ε²=0.366) is mostly redundant with the specs a brand happens "
            "to ship, not an independent pricing lever. Product and pricing strategy should be "
            "built around spec bundles, not brand positioning."
        )
    with st.expander("Q2. Brand price differences among 128GB+ phones"):
        st.markdown(
            "At the identical memory tier, Apple shows a substantially higher average price than "
            "Samsung (3.47×) and Realme (4.68×) within this dataset — an observed directional "
            "difference, not a precise multiplier. **Apple's n=3 in this cut is very small**, so "
            "treat this as suggestive rather than confirmed. This is consistent with the model's "
            "single worst miss (underpricing a Samsung flagship, Galaxy S23 5G, by 32.8%)."
        )
        
    with st.expander("Q3. Memory × RAM interaction"):
        st.markdown(
            "Memory and RAM are each individually strong price drivers, but combining top-tier "
            "levels of both yields slightly less than the sum of their separate effects — a "
            "statistically confirmed sub-additive relationship (interaction coefficient = −0.0246, "
            "p = 0.017 on the log scale), not just a visual impression."
        )
    with st.expander("Q4. How cleanly do price tiers separate?"):
        st.markdown(
            "The market has two cleanly spec-differentiated zones — Budget (F1=0.91) and Flagship "
            "(F1=0.73) — and a large middle. Mid and Premium (F1≈0.65 and 0.66) account for "
            "roughly half the listings, where devices with similar specs are less clearly "
            "separated but still classified meaningfully above the naive baseline. Any "
            "pricing-tier strategy built on \"specs justify the price bracket\" is weaker for "
            "about half the market."
        )
    with st.expander("Q5. Price association by chipset supplier"):
        st.markdown(
            "Qualcomm-chipset devices show a **41.0%** higher average price than MediaTek devices "
            "within this dataset (₹20,777 vs. ₹14,738). This is an association confounded by "
            "brand mix — chip sourcing is near-exclusive per brand, so this figure reflects "
            "brand-level pricing patterns as much as chipset choice itself, not evidence that a "
            "Qualcomm partnership on its own would produce this price difference."
        )
    with st.expander("Q6. How much error is associated with the flagged data-quality cluster?"):
        st.markdown(
            "The flagged cluster (screen-size entries in inconsistent units, corrected during "
            "cleaning) is only ~1% of the test set by row count, but accounts for **~4.6%** of the "
            "model's total prediction error — a disproportion of roughly 4.6×. At full dataset "
            "scale it's 15 of 531 listings (2.82%). This is the single highest-leverage "
            "data-quality investment identifiable from the analysis: a targeted fix for whatever "
            "process produces this specific listing pattern, not a general \"clean the data better\" "
            "instruction."
        )
    with st.expander("Q7. Front camera vs. rear camera price economics"):
        st.markdown(
            "Front camera MP shows a stronger linear price association (**₹620/MP**) than rear "
            "camera MP (**₹139/MP**) — roughly 4.5× the sensitivity, despite a much narrower spec "
            "range. The reason: 48% of listings (257 of 531) already ship a 50MP rear camera, so "
            "rear MP is commoditized and barely differentiates a listing anymore. Rear-camera "
            "marketing (\"64MP camera!\") is no longer associated with a distinct price premium; "
            "front-camera specs currently carry more remaining price association."
        )
    with st.expander("Q8. Could a 10-feature 'lite' pricing tool work in the field?"):
        st.markdown(
            "Yes — a 10-feature lite model retains **97.5%** of the full model's mean grouped-CV R² "
            "(0.670 vs. 0.688) using 76% fewer inputs, at a modest cost of ~₹121 higher CV MAE. The "
            "R² gap (0.017) is smaller than the full model's own fold-to-fold standard deviation "
            "(0.073), suggesting most predictive performance can be retained with a considerably "
            "smaller feature set. This could support a lightweight pricing-sanity tool needing no "
            "brand lookup — though it should be validated on newer market data before operational "
            "use."
        )
    with st.expander("Q9. Worked example — hypothetical device pricing"):
        st.markdown(
            "For a hypothetical 128GB/6GB RAM, 5000mAh, 64MP rear/16MP front, 16.7cm screen, "
            "Realme, MediaTek, 5G device, the model's point estimate is **₹19,993**, with "
            "**₹16,786–₹23,199** shown as an illustrative global error reference (± the ₹3,206 "
            "held-out test MAE, Section 9A) — not a statistically calibrated prediction interval."
        )
    with st.expander("Q10. Modelled 5G price methodology"):
        st.markdown(
            "A ceteris-paribus simulation — flipping only the 5G flag and holding every other spec "
            "fixed, averaged across all 34 non-5G REDMI training rows — gives a mean "
            "**model-estimated scenario difference** of 41.0% (range 18.2%–64.5%). This is "
            "deliberately different from the raw REDMI group-mean gap of +134% (non-5G ₹10,361 vs. "
            "5G ₹24,234): the raw comparison conflates the 5G-associated difference with every "
            "other spec difference between REDMI's actual non-5G and 5G lineups. The 41% simulated "
            "figure — not the 134% raw comparison — is the defensible number for \"what if we only "
            "added 5G.\""
        )

    st.markdown("---")
    st.caption(
        "Every Q1–Q10 figure above traces to a specific notebook section (5A/5C, 6A/6B, 7C, 8D, "
        "9A–9C, 10A–10C, 11A) and none describes an association as a proven causal effect. Small "
        "brand samples (e.g. Apple n=2–9 depending on the cut) and the Brand×Chipset association's "
        "sparse contingency table are called out inline wherever they affect a claim's precision."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB — KEY INSIGHTS & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("💡 Key Insights & Limitations")

    st.markdown("### The Central Strategic Insight")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("#### 📊 Below Flagship Tier")
        st.markdown(
            "Price is **associated with specifications**. The overall model explains this "
            "segment's pricing reasonably well, and brand carries little independent pricing "
            "power once specs are known. **Compete on spec-per-rupee value.**"
        )
    with ic2:
        st.markdown("#### 👑 At Flagship Tier")
        st.markdown(
            "A price premium is **associated with** flagship brands (notably Apple and Google) "
            "that specs alone don't explain — though small brand sample sizes (n=2–9 for several "
            "flagship brands) mean this should be read as directional, not a precise multiplier. "
            "The model's largest errors concentrate here — underpricing a Samsung flagship by "
            "33%, a Motorola flagship by 42%. **Closing this gap plausibly requires brand-equity "
            "investment, not spec bumps** — a hypothesis suggested by the data, not something "
            "this dataset can test directly."
        )

    st.markdown("---")
    st.markdown("### Limitations")
    st.markdown(
        "- Sample size (n=531) limits statistical power, especially for brands with roughly 2–9 listings\n"
        "- All relationships reported are associations, not proven causal effects — price is "
        "market-set, not derived from a known formula\n"
        "- No manufacturing cost, competitor pricing, or time-series data was available\n"
        "- Two variables (Battery Level, Screen Size) showed misleading relationships until a "
        "15-listing data-quality cluster was correctly excluded\n"
        "- The Brand × Chipset association (Cramér's V=0.699) rests on a sparse contingency "
        "table and should be read as directional, not statistically confirmed\n"
        "- Flagged data-quality records (2.8% of listings) are low-confidence predictions and "
        "require manual validation before use in pricing decisions\n"
        "- The held-out test set was reused for post-selection diagnostics (not just final scoring), "
        "which introduces mild selection optimism into the reported test-split figures — treat the "
        "model-ranking numbers as directional rather than a fully independent confirmation\n"
        "- Repeated train/test splits (5 seeds) show real variance around the headline test R²: "
        "mean 0.584 (std 0.118, range 0.430–0.751) versus the single reported seed=42 result of "
        "0.751, which sits at the favourable end of that spread — the grouped-CV figure (0.688) "
        "and the repeated-split figure (0.584) are two different stability checks and shouldn't be "
        "merged into one number\n"
        "- Impurity-based and permutation-based feature importance are computed on different "
        "samples (428-row training subset vs. held-out test set) and can disagree for identifiable "
        "reasons (e.g. Front_Camera_MP's confound with Apple's small, high-price sample) — neither "
        "should be treated as the single ground-truth ranking on its own"
    )

st.markdown("---")
st.caption(
    "Mobile Phone Price Prediction Dashboard · NextHikes IT Solutions · "
    "Full methodology and statistical validation in the companion analysis notebook."
)
