"""
app.py
Customer Intelligence AI — Main Streamlit Application
BICS Global Internship | Data Analyst Intern — Data Science Domain
"""
import io
import sys
import numpy as np
import pandas as pd
import streamlit as st

# ── Path fix so src/ imports work ─────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    generate_sample_data,
    validate_and_load_csv,
    get_numeric_columns,
    get_categorical_columns,
    get_data_quality_report,
    clean_dataframe,
)
from src.analytics import (
    compute_kpis,
    compute_correlation_matrix,
    get_gender_distribution,
    get_age_bins,
    get_income_vs_spending,
    get_purchase_frequency_bins,
)
from src.clustering import run_kmeans, label_clusters, get_cluster_summary
from src.prediction import get_suitable_features, train_model, predict_single
from src.ai_agent import respond, SUGGESTED_QUESTIONS, _openai_available
from src import visualizations as viz

# ────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# Custom CSS — professional dark-navy/blue design
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── App background ─────────────────────────────────── */
.stApp { background: #f0f4f8; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    min-width: 220px !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.75rem !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
    min-height: 42px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.3) !important;
    border-color: rgba(59,130,246,0.5) !important;
}

/* ── KPI metric cards — prevent truncation ───────────── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 1rem 0.75rem !important;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    overflow: visible !important;
}
[data-testid="metric-container"] > div {
    overflow: visible !important;
}
/* Label row */
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    overflow: visible !important;
    line-height: 1.3 !important;
    min-height: 2.2em !important;
}
/* Value row */
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] > div {
    color: #1e40af !important;
    font-weight: 700 !important;
    font-size: 1.35rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
    line-height: 1.4 !important;
}

/* ── Main content padding ────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Page titles ─────────────────────────────────────── */
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.2rem;
    line-height: 1.3;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

/* ── Section headers ─────────────────────────────────── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #1e40af;
    border-left: 4px solid #3b82f6;
    padding-left: 0.75rem;
    margin: 1.1rem 0 0.7rem 0;
    line-height: 1.4;
}

/* ── Suggested question buttons (AI Assistant) ───────── */
div[data-testid="column"] > div > div > div > .stButton > button {
    white-space: normal !important;
    word-wrap: break-word !important;
    height: auto !important;
    min-height: 60px !important;
    text-align: center !important;
    line-height: 1.4 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.6rem !important;
}

/* ── Chat input area — ensure Send/Clear not cramped ─── */
.stTextInput > div > div > input {
    font-size: 0.95rem !important;
    padding: 0.55rem 0.75rem !important;
}

/* ── Primary / secondary buttons in main area ────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    white-space: nowrap !important;
    min-width: fit-content !important;
}
.stButton > button[kind="secondary"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
}

/* ── Chat bubbles ────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    margin-left: 10%;
    font-size: 0.9rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.chat-assistant {
    background: white;
    border: 1px solid #e2e8f0;
    color: #1e293b;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    margin-right: 5%;
    font-size: 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    line-height: 1.6;
    word-wrap: break-word;
}
.mode-badge {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    display: block;
}

/* ── Status badges ───────────────────────────────────── */
.status-ok {
    display: inline-block;
    background: #dcfce7;
    color: #16a34a;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}
.status-warn {
    display: inline-block;
    background: #fef9c3;
    color: #b45309;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}
.status-error {
    display: inline-block;
    background: #fee2e2;
    color: #b91c1c;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
}

/* Divider */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }

/* Hide default Streamlit footer/menu */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Session state initialisation
# ────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df": None,
        "is_sample_data": True,
        "data_quality": None,
        "kpis": None,
        # clustering
        "cluster_result": None,
        "cluster_df": None,
        "cluster_summary": None,
        # prediction
        "pred_result": None,
        # chat
        "chat_history": [],
        # navigation
        "page": "Dashboard Overview",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────
def load_sample():
    df = generate_sample_data()
    _set_dataframe(df, is_sample=True)


def _set_dataframe(df: pd.DataFrame, is_sample: bool = False):
    df = clean_dataframe(df)
    st.session_state.df = df
    st.session_state.is_sample_data = is_sample
    st.session_state.data_quality = get_data_quality_report(df)
    st.session_state.kpis = compute_kpis(df)
    # Invalidate stale models
    st.session_state.cluster_result = None
    st.session_state.cluster_df = None
    st.session_state.cluster_summary = None
    st.session_state.pred_result = None
    st.session_state.chat_history = []


# Auto-load sample on first run
if st.session_state.df is None:
    load_sample()

# Quick-access shorthand
df_active: pd.DataFrame = st.session_state.df


# ────────────────────────────────────────────────────────────────
# Sidebar — navigation
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
      <div style='font-size:2.2rem;'>🧠</div>
      <div style='font-size:1.1rem; font-weight:700; color:#93c5fd; margin-top:4px;'>Customer Intelligence</div>
      <div style='font-size:0.75rem; color:#94a3b8; margin-top:2px;'>BICS Global Internship</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1); margin:0.75rem 0;'>
    """, unsafe_allow_html=True)

    NAV_PAGES = [
        ("📊", "Dashboard Overview"),
        ("🔍", "Customer Explorer"),
        ("🎯", "Customer Segmentation"),
        ("📈", "Predictive Analytics"),
        ("🤖", "AI Analytics Assistant"),
        ("📂", "Dataset Management"),
        ("ℹ️", "About the Project"),
    ]

    for icon, page in NAV_PAGES:
        if st.button(f"{icon}  {page}", key=f"nav_{page}"):
            st.session_state.page = page

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    # Dataset info in sidebar
    n_customers = len(df_active)
    source = "Sample Data" if st.session_state.is_sample_data else "Uploaded CSV"
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#94a3b8; padding: 0 0.25rem;'>
        <div style='margin-bottom:6px;'>📁 <b style='color:#cbd5e1;'>Dataset:</b> {source}</div>
        <div>👥 <b style='color:#cbd5e1;'>Customers:</b> {n_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.cluster_summary:
        n_seg = st.session_state.cluster_summary.get("n_clusters", 0)
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#94a3b8; padding: 0.25rem;'>
            🎯 <b style='color:#cbd5e1;'>Segments:</b> {n_seg}
        </div>
        """, unsafe_allow_html=True)

    if _openai_available:
        st.markdown("""
        <div style='font-size:0.72rem; background:rgba(16,185,129,0.15); border-radius:8px;
             padding:0.4rem 0.6rem; color:#6ee7b7; margin-top:0.75rem;'>
            ✅ OpenAI LLM Active
        </div>
        """, unsafe_allow_html=True)

current_page = st.session_state.page
df_active = st.session_state.df  # re-read after any mutations


# ────────────────────────────────────────────────────────────────
# Shared agent state builder
# ────────────────────────────────────────────────────────────────
def _agent_state() -> dict:
    return {
        "df": st.session_state.df,
        "kpis": st.session_state.kpis,
        "cluster_summary": st.session_state.cluster_summary,
        "pred_result": st.session_state.pred_result,
        "data_quality": st.session_state.data_quality,
    }


# ════════════════════════════════════════════════════════════════
# PAGE: Dashboard Overview
# ════════════════════════════════════════════════════════════════
if current_page == "Dashboard Overview":
    st.markdown('<div class="page-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Key performance indicators and interactive visualizations for your customer dataset.</div>', unsafe_allow_html=True)

    kpis = st.session_state.kpis or {}

    # KPI cards — row 1 (3 cards)
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Customers", f"{kpis.get('total_customers', 0):,}")
    ann_inc = kpis.get('avg_annual_income')
    c2.metric("💰 Avg Annual Income", f"${ann_inc}" if ann_inc else "N/A")
    c3.metric("🛒 Avg Spending Score", kpis.get("avg_spending_score", "N/A"))

    # KPI cards — row 2 (2 cards + spacer)
    c4, c5, _ = st.columns(3)
    aov = kpis.get('avg_order_value')
    c4.metric("📦 Avg Order Value", f"${aov}" if aov else "N/A")
    c5.metric("🔁 Avg Purchase Frequency", kpis.get("avg_purchase_freq", "N/A"))

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: scatter + gender pie
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = viz.income_vs_spending_scatter(df_active)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a dataset with AnnualIncome and SpendingScore to see this chart.")
    with col2:
        fig = viz.gender_pie(df_active)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Gender column found.")

    # Row 2: age + frequency histograms
    col3, col4 = st.columns(2)
    with col3:
        fig = viz.age_histogram(df_active)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = viz.purchase_freq_histogram(df_active)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    fig = viz.correlation_heatmap(df_active)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")

    # Customer table
    st.markdown('<div class="section-header">Customer Data Table</div>', unsafe_allow_html=True)
    search_term = st.text_input("🔎 Search by CustomerID or any value", placeholder="e.g. CUST0001 or Female")
    display_df = df_active.copy()
    if search_term:
        mask = display_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(display_df.head(200), use_container_width=True, height=300)
    st.caption(f"Showing {min(len(display_df), 200)} of {len(display_df)} records.")


# ════════════════════════════════════════════════════════════════
# PAGE: Customer Explorer
# ════════════════════════════════════════════════════════════════
elif current_page == "Customer Explorer":
    st.markdown('<div class="page-title">🔍 Customer Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Drill down into individual customer profiles and compare them with their segment averages.</div>', unsafe_allow_html=True)

    df_exp = st.session_state.cluster_df if st.session_state.cluster_df is not None else df_active

    # Sidebar-style filters
    with st.expander("🔧 Filters", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            cid_search = st.text_input("Customer ID", placeholder="CUST0001")
        with fcol2:
            gender_opts = ["All"] + sorted(df_exp["Gender"].dropna().unique().tolist()) if "Gender" in df_exp.columns else ["All"]
            gender_filter = st.selectbox("Gender", gender_opts)
        with fcol3:
            seg_opts = ["All"] + sorted(df_exp["CustomerSegment"].dropna().unique().tolist()) if "CustomerSegment" in df_exp.columns else ["All"]
            seg_filter = st.selectbox("Segment", seg_opts)

        age_min, age_max = (int(df_exp["Age"].min()), int(df_exp["Age"].max())) if "Age" in df_exp.columns else (0, 100)
        fcol4, fcol5 = st.columns(2)
        with fcol4:
            age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        with fcol5:
            inc_min, inc_max = (int(df_exp["AnnualIncome"].min()), int(df_exp["AnnualIncome"].max())) if "AnnualIncome" in df_exp.columns else (0, 200000)
            inc_range = st.slider("Annual Income Range ($)", inc_min, inc_max, (inc_min, inc_max))

    # Apply filters
    result = df_exp.copy()
    if cid_search and "CustomerID" in result.columns:
        result = result[result["CustomerID"].astype(str).str.contains(cid_search, case=False, na=False)]
    if gender_filter != "All" and "Gender" in result.columns:
        result = result[result["Gender"] == gender_filter]
    if seg_filter != "All" and "CustomerSegment" in result.columns:
        result = result[result["CustomerSegment"] == seg_filter]
    if "Age" in result.columns:
        result = result[(result["Age"] >= age_range[0]) & (result["Age"] <= age_range[1])]
    if "AnnualIncome" in result.columns:
        result = result[(result["AnnualIncome"] >= inc_range[0]) & (result["AnnualIncome"] <= inc_range[1])]

    st.markdown(f"**{len(result):,}** customers match your filters.")

    if result.empty:
        st.warning("No customers match the selected filters. Adjust the criteria above.")
    else:
        st.dataframe(result, use_container_width=True, height=350)

        # Single-customer deep-dive
        if "CustomerID" in result.columns and len(result) == 1:
            customer = result.iloc[0]
            st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            numeric_display = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c]) and c not in ("ClusterID",)]
            for i, col_name in enumerate(numeric_display[:9]):
                val = customer[col_name]
                cols[i % 3].metric(col_name, f"{val:,.1f}" if isinstance(val, float) else f"{val:,}")

            if "CustomerSegment" in result.columns:
                seg_name = customer.get("CustomerSegment", "N/A")
                st.markdown(f"""
                <div class="card">
                    <b>Customer Segment:</b> <span style='color:#2563eb; font-weight:600;'>{seg_name}</span>
                </div>
                """, unsafe_allow_html=True)

                # Compare with cluster average
                if st.session_state.cluster_result and st.session_state.cluster_result.get("profile") is not None:
                    cluster_id = customer.get("ClusterID")
                    if pd.notna(cluster_id):
                        profile = st.session_state.cluster_result["profile"]
                        if int(cluster_id) in profile.index:
                            avg_row = profile.loc[int(cluster_id)]
                            compare_cols = [c for c in avg_row.index if c in result.columns]
                            cdf = pd.DataFrame({
                                "Feature": compare_cols,
                                "This Customer": [customer.get(c, "N/A") for c in compare_cols],
                                "Segment Average": [round(avg_row[c], 2) for c in compare_cols],
                            })
                            st.markdown('<div class="section-header">vs. Segment Average</div>', unsafe_allow_html=True)
                            st.dataframe(cdf, use_container_width=True, hide_index=True)
        elif len(result) > 1:
            st.info("Narrow your search to a single customer to see their detailed profile and segment comparison.")


# ════════════════════════════════════════════════════════════════
# PAGE: Customer Segmentation
# ════════════════════════════════════════════════════════════════
elif current_page == "Customer Segmentation":
    st.markdown('<div class="page-title">🎯 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Group customers into meaningful segments using K-Means clustering.</div>', unsafe_allow_html=True)

    num_cols = get_numeric_columns(df_active)
    default_feats = [c for c in ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency", "AverageOrderValue"] if c in num_cols]

    with st.expander("⚙️ Clustering Configuration", expanded=True):
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            selected_features = st.multiselect(
                "Features for Clustering",
                options=num_cols,
                default=default_feats if default_feats else num_cols[:4],
                help="Select 2 or more numeric features.",
            )
        with scol2:
            n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)

        run_btn = st.button("▶️ Run Segmentation", type="primary")

    if run_btn:
        if len(selected_features) < 2:
            st.error("Please select at least 2 features for clustering.")
        else:
            with st.spinner("Training K-Means model…"):
                result = run_kmeans(df_active, selected_features, n_clusters)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    labeled_df = label_clusters(
                        df_active,
                        result["labels"],
                        result["valid_indices"],
                        result["profile"],
                        selected_features,
                    )
                    st.session_state.cluster_result = result
                    st.session_state.cluster_df = labeled_df
                    st.session_state.cluster_summary = get_cluster_summary(labeled_df)
                    # Refresh df for display
                    df_active = labeled_df
                    st.success(f"✅ Segmentation complete — {n_clusters} clusters identified!")

    # Display results if available
    cr = st.session_state.cluster_result
    cdf = st.session_state.cluster_df
    cs = st.session_state.cluster_summary

    if cr and cdf is not None and not cr.get("error"):
        # Silhouette score
        sil = cr.get("silhouette")
        sil_col, sil_msg = ("green", f"**Silhouette Score:** {sil:.3f} — {'Good' if sil > 0.4 else 'Fair' if sil > 0.2 else 'Weak'} separation") if sil is not None else ("gray", "Silhouette score not computed.")
        st.markdown(f"""<div class="card"><span style='color:{sil_col};'>◉</span> {sil_msg}</div>""", unsafe_allow_html=True)

        # Cluster size table
        st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
        size_col1, size_col2 = st.columns([1, 2])
        with size_col1:
            if cs:
                size_data = pd.DataFrame([
                    {"Segment": s["label"], "Customers": s["size"], "Share (%)": f"{s['pct']:.1f}%"}
                    for s in cs["segments"]
                ])
                st.dataframe(size_data, use_container_width=True, hide_index=True)
        with size_col2:
            fig_bar = viz.cluster_size_bar(cdf)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter plot with axis selection
        st.markdown('<div class="section-header">Cluster Visualization</div>', unsafe_allow_html=True)
        viz_col1, viz_col2 = st.columns(2)
        feat_list = cr.get("feature_cols", [])
        with viz_col1:
            x_axis = st.selectbox("X Axis", feat_list, index=0)
        with viz_col2:
            y_default = 1 if len(feat_list) > 1 else 0
            y_axis = st.selectbox("Y Axis", feat_list, index=y_default)

        fig_scatter = viz.cluster_scatter(cdf, x_axis, y_axis)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Cluster profiles
        st.markdown('<div class="section-header">Cluster Profiles (Average Values)</div>', unsafe_allow_html=True)
        profile = cr.get("profile")
        if profile is not None:
            profile_display = profile.copy()
            if cs:
                seg_names = {s["cluster_id"]: s["label"] for s in cs["segments"]}
                profile_display.index = [seg_names.get(i, f"Cluster {i}") for i in profile_display.index]
            st.dataframe(profile_display.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

        # Download
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        csv_bytes = cdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Segmented Dataset (CSV)",
            data=csv_bytes,
            file_name="customer_segments.csv",
            mime="text/csv",
        )
    else:
        st.info("Configure the settings above and click **Run Segmentation** to begin.")


# ════════════════════════════════════════════════════════════════
# PAGE: Predictive Analytics
# ════════════════════════════════════════════════════════════════
elif current_page == "Predictive Analytics":
    st.markdown('<div class="page-title">📈 Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Train a Random Forest regression model to estimate customer Spending Score.</div>', unsafe_allow_html=True)

    TARGET_COL = "SpendingScore"
    feature_cols = get_suitable_features(df_active, TARGET_COL)

    if TARGET_COL not in df_active.columns:
        st.warning(f"The active dataset does not contain a '{TARGET_COL}' column. Prediction is unavailable. Upload a compatible dataset or use the sample data.")
    elif len(df_active) < 30:
        st.warning("Not enough records for training. Need at least 30 rows.")
    else:
        with st.expander("⚙️ Model Configuration", expanded=True):
            selected_pred_features = st.multiselect(
                "Select Feature Columns",
                options=feature_cols,
                default=feature_cols,
                help="Numeric columns used to predict the Spending Score.",
            )
            train_btn = st.button("▶️ Train Model", type="primary")

        if train_btn:
            if not selected_pred_features:
                st.error("Please select at least one feature.")
            else:
                with st.spinner("Training Random Forest model…"):
                    result = train_model(df_active, selected_pred_features, TARGET_COL)
                    st.session_state.pred_result = result
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.success("✅ Model trained successfully!")

        pr = st.session_state.pred_result
        if pr and not pr.get("error"):
            # Metrics
            st.markdown('<div class="section-header">Model Evaluation Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 MAE", f"{pr['mae']:.3f}", help="Mean Absolute Error on test set — lower is better.")
            m2.metric("📐 R²", f"{pr['r2']:.4f}", help="Coefficient of determination — 1.0 is perfect.")
            m3.metric("🏋️ Training Rows", f"{pr['n_train']:,}")
            m4.metric("🧪 Test Rows", f"{pr['n_test']:,}")
            st.caption("⚠️ Metrics are calculated on held-out test data. Results reflect the synthetic dataset and are for demonstration only.")

            # Actual vs Predicted
            col_a, col_b = st.columns(2)
            with col_a:
                fig_avp = viz.actual_vs_predicted(pr["y_test"], pr["y_pred"], TARGET_COL)
                st.plotly_chart(fig_avp, use_container_width=True)
            with col_b:
                if pr.get("feature_importance") is not None:
                    fig_fi = viz.feature_importance_bar(pr["feature_importance"])
                    st.plotly_chart(fig_fi, use_container_width=True)

            # Interactive prediction form
            st.markdown('<div class="section-header">🔮 Predict Spending Score for a New Customer</div>', unsafe_allow_html=True)
            st.markdown("Enter customer attributes to get an estimated spending score prediction.")

            form_cols = st.columns(3)
            input_vals = {}
            for i, col_name in enumerate(pr["feature_cols"]):
                col_data = df_active[col_name].dropna()
                col_min, col_max = float(col_data.min()), float(col_data.max())
                col_mean = float(col_data.mean())
                with form_cols[i % 3]:
                    input_vals[col_name] = st.number_input(
                        col_name,
                        min_value=col_min,
                        max_value=col_max,
                        value=col_mean,
                        step=max((col_max - col_min) / 100, 0.1),
                        key=f"pred_input_{col_name}",
                    )

            if st.button("🔮 Predict", type="primary"):
                pred_val = predict_single(pr["model"], pr["feature_cols"], input_vals)
                if pred_val is not None:
                    st.markdown(f"""
                    <div class="card" style='background: linear-gradient(135deg,#eff6ff,#dbeafe); border:1px solid #93c5fd;'>
                        <div style='font-size:0.9rem; color:#1e40af; font-weight:500;'>Estimated Spending Score</div>
                        <div style='font-size:2.5rem; font-weight:800; color:#1d4ed8;'>{pred_val:.1f}</div>
                        <div style='font-size:0.78rem; color:#64748b; margin-top:4px;'>
                            This is a model estimate based on the current training data. Not a guarantee.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction failed. Check your input values.")
        elif pr and pr.get("error"):
            st.error(pr["error"])
        else:
            st.info("Click **Train Model** to train the prediction model.")


# ════════════════════════════════════════════════════════════════
# PAGE: AI Analytics Assistant
# ════════════════════════════════════════════════════════════════
elif current_page == "AI Analytics Assistant":
    st.markdown('<div class="page-title">🤖 AI Analytics Assistant</div>', unsafe_allow_html=True)
    mode_badge = "🟢 OpenAI LLM Active" if _openai_available else "🟡 Built-in Mode (no API key)"
    st.markdown(f'<div class="page-subtitle">Customer Intelligence Agent — {mode_badge}</div>', unsafe_allow_html=True)

    # Suggested questions — 3 wide columns for readability
    st.markdown('<div class="section-header">💡 Suggested Questions</div>', unsafe_allow_html=True)
    sq_cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with sq_cols[i % 3]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking…"):
                    response, mode = respond(q, _agent_state(), st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": response, "mode": mode})
                st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Chat history display
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center; padding:2rem; color:#94a3b8;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>💬</div>
            <div>Ask me anything about your customer data, segments, or model results.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                mode_text = "🟢 LLM" if msg.get("mode") == "llm" else "🟡 Built-in"
                st.markdown(f'<div class="chat-assistant">{content}<br><span class="mode-badge">{mode_text}</span></div>', unsafe_allow_html=True)

    # Input area — use st.form so Enter key submits and buttons don't squish
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            label_visibility="collapsed",
            placeholder="Type your question here and press Send or hit Enter…",
        )
        form_c1, form_c2 = st.columns([6, 1])
        with form_c1:
            submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)
        with form_c2:
            cleared = st.form_submit_button("🗑️ Clear", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking…"):
            response, mode = respond(user_input.strip(), _agent_state(), st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": response, "mode": mode})
        st.rerun()

    if cleared:
        st.session_state.chat_history = []
        st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE: Dataset Management
# ════════════════════════════════════════════════════════════════
elif current_page == "Dataset Management":
    st.markdown('<div class="page-title">📂 Dataset Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload your own CSV, explore its quality, or reset to the sample dataset.</div>', unsafe_allow_html=True)

    # Sample data download
    sample_df = generate_sample_data()
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        st.info("Download the sample CSV to understand the expected format, or upload your own customer dataset below.")
    with dcol2:
        st.download_button(
            "⬇️ Download Sample CSV",
            data=sample_csv,
            file_name="sample_customers.csv",
            mime="text/csv",
        )

    # Upload
    st.markdown('<div class="section-header">Upload Your Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        df_upload, err = validate_and_load_csv(uploaded)
        if err:
            st.error(f"❌ Upload failed: {err}")
        else:
            # Column mapping if key columns missing
            num_available = get_numeric_columns(df_upload)
            st.success(f"✅ File loaded: {len(df_upload):,} rows × {len(df_upload.columns)} columns.")

            # Check if standard columns exist
            missing_standard = [c for c in ["AnnualIncome", "SpendingScore"] if c not in df_upload.columns]
            if missing_standard:
                st.warning(f"Columns {missing_standard} not found. Please map your columns below.")
                col_map = {}
                for std_col in missing_standard:
                    col_map[std_col] = st.selectbox(
                        f"Map '{std_col}' to:",
                        options=["— skip —"] + num_available,
                        key=f"map_{std_col}",
                    )
                if st.button("Apply Column Mapping"):
                    for std_col, mapped in col_map.items():
                        if mapped != "— skip —":
                            df_upload[std_col] = df_upload[mapped]
                    _set_dataframe(df_upload, is_sample=False)
                    st.success("Dataset updated with column mapping!")
                    st.rerun()
            else:
                if st.button("✅ Use This Dataset", type="primary"):
                    _set_dataframe(df_upload, is_sample=False)
                    st.success("Dataset updated!")
                    st.rerun()

            st.markdown("**Preview (first 10 rows):**")
            st.dataframe(df_upload.head(10), use_container_width=True)

    # Data Quality Report
    st.markdown('<div class="section-header">Data Quality Report</div>', unsafe_allow_html=True)
    dq = st.session_state.data_quality
    if dq:
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("📋 Records", f"{dq['total_records']:,}")
        qc2.metric("🗂️ Columns", dq["total_columns"])
        qc3.metric("❓ Missing Values", dq["missing_values_total"])
        qc4.metric("🔁 Duplicate Rows", dq["duplicate_rows"])

        ctype_col1, ctype_col2 = st.columns(2)
        with ctype_col1:
            st.markdown("**Numeric Columns:**")
            st.write(", ".join(dq["numeric_columns"]) if dq["numeric_columns"] else "None")
        with ctype_col2:
            st.markdown("**Categorical Columns:**")
            st.write(", ".join(dq["categorical_columns"]) if dq["categorical_columns"] else "None")

        if dq["missing_by_column"]:
            st.markdown("**Missing values by column:**")
            miss_df = pd.DataFrame(list(dq["missing_by_column"].items()), columns=["Column", "Missing Count"])
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<span class="status-ok">✓ No missing values detected</span>', unsafe_allow_html=True)

    # Reset
    st.markdown('<div class="section-header">Reset Dataset</div>', unsafe_allow_html=True)
    if not st.session_state.is_sample_data:
        if st.button("🔄 Reset to Sample Data", type="secondary"):
            load_sample()
            st.success("Reset to sample data. All models and results have been cleared.")
            st.rerun()
    else:
        st.markdown('<span class="status-ok">✓ Currently using sample data</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE: About the Project
# ════════════════════════════════════════════════════════════════
elif current_page == "About the Project":
    st.markdown('<div class="page-title">ℹ️ About the Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div style='font-size:1.4rem; font-weight:700; color:#1e40af; margin-bottom:0.5rem;'>
            🧠 Customer Intelligence AI
        </div>
        <div style='color:#475569; line-height:1.7;'>
            An intelligent customer segmentation and predictive analytics dashboard built for the
            <b>BICS Global</b> summer internship project by a <b>Data Analyst Intern</b> in the
            <b>Data Science Domain</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🛠️ Technology Stack
        | Component | Technology |
        |---|---|
        | Web UI | Streamlit |
        | Data Processing | Pandas, NumPy |
        | Machine Learning | Scikit-learn |
        | Visualizations | Plotly |
        | AI Assistant | Built-in + OpenAI (optional) |
        | Configuration | python-dotenv |
        """)

    with col2:
        st.markdown("""
        ### ✅ Implemented Features
        - 📊 KPI dashboard with live metrics
        - 🔍 Interactive customer explorer
        - 🎯 K-Means customer segmentation (K=2–8)
        - 📈 RandomForest spending score prediction
        - 🤖 AI analytics assistant (built-in + GPT)
        - 📂 CSV upload with data quality report
        - ⬇️ Export segmented customer data
        - 🔄 Sample data auto-generation (700 customers)
        """)

    st.markdown("""
    ### 🎬 5-Minute Demo Script

    1. **Open Dashboard Overview** — Show KPI cards and explain what each metric means.
    2. **Describe the dataset** — 700 synthetic customers, 10 attributes, realistic behaviour patterns.
    3. **Navigate to Segmentation** — Select default features, change K to 4, click Run Segmentation.
    4. **Explain Cluster Profiles** — Point out the high-value vs. budget customer groups.
    5. **Open Predictive Analytics** — Train the model, show MAE and R² scores, use the prediction form.
    6. **Open AI Assistant** — Click "Summarize the customer dataset", then "What strategies could improve customer retention?"
    7. **Dataset Management** — Show the data quality report, demonstrate CSV upload flow.
    8. **Export** — Return to Segmentation, click Download Segmented Dataset.
    """)

    st.markdown("""
    ### ❗ Known Limitations
    - Predictions and clustering are based on synthetic/uploaded data — not real customer data.
    - The built-in AI assistant uses keyword-based routing; for richer answers, configure an OpenAI API key.
    - For very small uploaded datasets (<30 rows), prediction training may be unavailable.
    - The app runs locally; no authentication or multi-user support is implemented.
    """)

    st.markdown("""
    <div class="card" style='background:#f0fdf4; border-color:#86efac;'>
        <b>🚀 Launch Instructions</b><br>
        <code>pip install -r requirements.txt</code><br>
        <code>streamlit run app.py</code><br><br>
        Optional: Add your OpenAI API key to a <code>.env</code> file for enhanced AI responses.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.75rem; margin-top:2rem; padding:1rem;
     border-top:1px solid #e2e8f0;'>
    Customer Intelligence AI · BICS Global Internship · Built with Streamlit
</div>
""", unsafe_allow_html=True)
