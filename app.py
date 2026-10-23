
import io
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    generate_sample_data, validate_and_load_csv, get_numeric_columns,
    get_categorical_columns, get_data_quality_report, clean_dataframe,
)
from src.analytics import (
    compute_kpis, compute_correlation_matrix, get_gender_distribution,
    get_age_bins, get_income_vs_spending, get_purchase_frequency_bins,
)
from src.clustering import run_kmeans, label_clusters, get_cluster_summary
from src.prediction import get_suitable_features, train_model, predict_single
from src.ai_agent import respond, SUGGESTED_QUESTIONS, _openai_available
from src import visualizations as viz


st.set_page_config(
    page_title="Customer Intelligence AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #F4F7FC;
    --surface: #FFFFFF;
    --text: #172033;
    --muted: #526176;
    --border: #E1E8F0;
    --blue: #2563EB;
}

html, body, [class*="css"], button, input, textarea {
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(244,247,252,0.96);
}

.main .block-container {
    max-width: 1600px;
    padding: 2rem clamp(1rem, 3vw, 2.5rem) 3rem;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #172B4D;
    letter-spacing: -0.025em;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: var(--text);
}

.page-title {
    color: #172B4D;
    font-size: clamp(1.6rem, 2.4vw, 2.15rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.3;
    margin-bottom: 0.4rem;
}

.page-subtitle {
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.7;
    margin-bottom: 1.6rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111D33 0%, #1D3557 100%);
    border-right: 1px solid #D8E0EC;
}

[data-testid="stSidebar"] > div {
    padding: 1.25rem 1rem;
}

[data-testid="stSidebar"] * {
    color: #E8EEF8;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #CBD5E1;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 44px;
    white-space: normal;
    text-align: left;
    background: rgba(255,255,255,0.06);
    color: #F1F5F9;
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 500;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.25);
    border-color: #60A5FA;
    color: white;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.1rem;
    min-width: 0;
    box-shadow: 0 4px 14px rgba(15,23,42,0.045);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #526176 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    line-height: 1.5 !important;
}

[data-testid="stMetricValue"] {
    color: #1D4ED8 !important;
    font-size: clamp(1.15rem, 1.8vw, 1.8rem) !important;
    font-weight: 800 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    line-height: 1.45 !important;
}

/* Panels */
.card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.25rem;
    margin: 0.6rem 0 1rem;
    box-shadow: 0 4px 14px rgba(15,23,42,0.035);
}

.section-header {
    color: #1E40AF;
    font-size: 1rem;
    font-weight: 750;
    border-left: 4px solid #3B82F6;
    padding: 0.2rem 0 0.2rem 0.75rem;
    margin: 1.5rem 0 0.9rem;
    line-height: 1.5;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* Buttons */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] .stDownloadButton > button,
[data-testid="stMain"] .stFormSubmitButton > button {
    min-height: 40px;
    border-radius: 9px;
    font-weight: 600;
    transition: 0.18s ease;
}

[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stDownloadButton > button[kind="primary"] {
    background: #2563EB;
    color: white;
    border: 1px solid #2563EB;
}

[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
}

/* Inputs */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    color: #172033;
}

[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {
    color: #78869A;
}

/* AI suggested questions */
[data-testid="stMain"] [data-testid="column"] .stButton > button {
    white-space: normal;
    line-height: 1.5;
    min-height: 48px;
}

/* Chat */
.chat-user {
    background: #2563EB;
    color: #FFFFFF;
    border-radius: 16px 16px 4px 16px;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0 0.6rem 8%;
    font-size: 0.92rem;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.chat-assistant {
    background: #FFFFFF;
    border: 1px solid #DCE4EF;
    color: #172033;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.1rem;
    margin: 0.6rem 5% 0.6rem 0;
    font-size: 0.92rem;
    line-height: 1.75;
    box-shadow: 0 3px 10px rgba(15,23,42,0.035);
    overflow-wrap: anywhere;
}

.mode-badge {
    color: #526176;
    font-size: 0.75rem;
    display: inline-block;
    margin-top: 0.55rem;
}

.status-ok {
    display: inline-block;
    color: #166534;
    background: #DCFCE7;
    border-radius: 20px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 700;
}

.status-warn {
    display: inline-block;
    color: #92400E;
    background: #FEF3C7;
    border-radius: 20px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 700;
}

.status-error {
    display: inline-block;
    color: #991B1B;
    background: #FEE2E2;
    border-radius: 20px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 700;
}

.divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.25rem 0;
}

@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 0.8rem 2rem;
    }

    [data-testid="stMetric"] {
        padding: 0.8rem;
    }

    .chat-user {
        margin-left: 2%;
    }

    .chat-assistant {
        margin-right: 2%;
    }
}
</style>
""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "df": None,
        "is_sample_data": True,
        "data_quality": None,
        "kpis": None,
        "cluster_result": None,
        "cluster_df": None,
        "cluster_summary": None,
        "pred_result": None,
        "chat_history": [],
        "page": "Dashboard Overview",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _set_dataframe(df: pd.DataFrame, is_sample: bool = False):
    df = clean_dataframe(df)
    st.session_state.df = df
    st.session_state.is_sample_data = is_sample
    st.session_state.data_quality = get_data_quality_report(df)
    st.session_state.kpis = compute_kpis(df)
    st.session_state.cluster_result = None
    st.session_state.cluster_df = None
    st.session_state.cluster_summary = None
    st.session_state.pred_result = None
    st.session_state.chat_history = []


def load_sample():
    _set_dataframe(generate_sample_data(), is_sample=True)


if st.session_state.df is None:
    load_sample()

df_active = st.session_state.df


with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
      <div style="font-size:2.5rem;">🧠</div>
      <div style="font-size:1.12rem;font-weight:800;color:#FFFFFF;">Customer Intelligence AI</div>
      <div style="font-size:0.78rem;color:#CBD5E1;margin-top:5px;">BICS Global · Internship Project</div>
    </div>
    <hr style="border-color:rgba(255,255,255,.16);">
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
        if st.button(f"{icon}  {page}", key=f"nav_{page}", width="stretch"):
            st.session_state.page = page

    st.markdown("<hr style='border-color:rgba(255,255,255,.16);'>", unsafe_allow_html=True)
    st.caption("ACTIVE DATASET")
    st.markdown(f"**Source:** {'Sample Data' if st.session_state.is_sample_data else 'Uploaded CSV'}")
    st.markdown(f"**Customers:** {len(df_active):,}")

    if st.session_state.cluster_summary:
        st.markdown(f"**Segments:** {st.session_state.cluster_summary.get('n_clusters', 0)}")

    if _openai_available:
        st.success("OpenAI assistant available")
    else:
        st.info("Built-in assistant mode")


current_page = st.session_state.page


def _agent_state() -> dict:
    return {
        "df": st.session_state.df,
        "kpis": st.session_state.kpis,
        "cluster_summary": st.session_state.cluster_summary,
        "pred_result": st.session_state.pred_result,
        "data_quality": st.session_state.data_quality,
    }


if current_page == "Dashboard Overview":
    st.markdown('<div class="page-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">A clear view of customer behaviour, spending patterns, and key business indicators.</div>', unsafe_allow_html=True)

    kpis = st.session_state.kpis or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Customers", f"{kpis.get('total_customers', 0):,}")
    ann_inc = kpis.get("avg_annual_income")
    c2.metric("💰 Average Annual Income", f"${ann_inc:,.2f}" if isinstance(ann_inc, (int, float, np.number)) else "N/A")
    c3.metric("🛒 Average Spending Score", kpis.get("avg_spending_score", "N/A"))

    c4, c5, c6 = st.columns(3)
    aov = kpis.get("avg_order_value")
    c4.metric("📦 Average Order Value", f"${aov:,.2f}" if isinstance(aov, (int, float, np.number)) else "N/A")
    c5.metric("🔁 Average Purchase Frequency", kpis.get("avg_purchase_freq", "N/A"))
    c6.metric("🗂️ Dataset Columns", f"{len(df_active.columns):,}")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = viz.income_vs_spending_scatter(df_active)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Upload a dataset with AnnualIncome and SpendingScore to see this chart.")
    with col2:
        fig = viz.gender_pie(df_active)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No Gender column found.")

    col3, col4 = st.columns(2)
    with col3:
        fig = viz.age_histogram(df_active)
        if fig:
            st.plotly_chart(fig, width="stretch")
    with col4:
        fig = viz.purchase_freq_histogram(df_active)
        if fig:
            st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    fig = viz.correlation_heatmap(df_active)
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Need at least two numeric columns for a correlation heatmap.")

    st.markdown('<div class="section-header">Customer Data Table</div>', unsafe_allow_html=True)
    search_term = st.text_input("Search customers", placeholder="Enter a customer ID or value")
    display_df = df_active.copy()
    if search_term:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(display_df.head(200), width="stretch", height=300)
    st.caption(f"Showing {min(len(display_df), 200):,} of {len(display_df):,} matching records.")


elif current_page == "Customer Explorer":
    st.markdown('<div class="page-title">🔍 Customer Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Find customers, apply filters, and inspect individual profiles.</div>', unsafe_allow_html=True)

    df_exp = st.session_state.cluster_df if st.session_state.cluster_df is not None else df_active

    with st.expander("🔧 Customer filters", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            cid_search = st.text_input("Customer ID", placeholder="CUST0001")
        with fcol2:
            gender_opts = ["All"] + sorted(df_exp["Gender"].dropna().astype(str).unique().tolist()) if "Gender" in df_exp.columns else ["All"]
            gender_filter = st.selectbox("Gender", gender_opts)
        with fcol3:
            seg_opts = ["All"] + sorted(df_exp["CustomerSegment"].dropna().astype(str).unique().tolist()) if "CustomerSegment" in df_exp.columns else ["All"]
            seg_filter = st.selectbox("Segment", seg_opts)

        if "Age" in df_exp.columns and pd.api.types.is_numeric_dtype(df_exp["Age"]):
            age_min, age_max = int(df_exp["Age"].min()), int(df_exp["Age"].max())
            age_range = st.slider("Age range", age_min, max(age_min, age_max), (age_min, max(age_min, age_max)))
        else:
            age_range = None

        if "AnnualIncome" in df_exp.columns and pd.api.types.is_numeric_dtype(df_exp["AnnualIncome"]):
            inc_min, inc_max = int(df_exp["AnnualIncome"].min()), int(df_exp["AnnualIncome"].max())
            inc_range = st.slider("Annual income range", inc_min, max(inc_min, inc_max), (inc_min, max(inc_min, inc_max)))
        else:
            inc_range = None

    result = df_exp.copy()
    if cid_search and "CustomerID" in result.columns:
        result = result[result["CustomerID"].astype(str).str.contains(cid_search, case=False, na=False)]
    if gender_filter != "All" and "Gender" in result.columns:
        result = result[result["Gender"].astype(str) == gender_filter]
    if seg_filter != "All" and "CustomerSegment" in result.columns:
        result = result[result["CustomerSegment"].astype(str) == seg_filter]
    if age_range and "Age" in result.columns:
        result = result[result["Age"].between(*age_range)]
    if inc_range and "AnnualIncome" in result.columns:
        result = result[result["AnnualIncome"].between(*inc_range)]

    st.markdown(f"**{len(result):,} customers** match your filters.")

    if result.empty:
        st.warning("No customers match these filters. Try adjusting your selections.")
    else:
        st.dataframe(result, width="stretch", height=350)
        if "CustomerID" in result.columns and len(result) == 1:
            customer = result.iloc[0]
            st.markdown('<div class="section-header">Customer profile</div>', unsafe_allow_html=True)
            numeric_display = [
                col for col in result.columns
                if pd.api.types.is_numeric_dtype(result[col]) and col != "ClusterID"
            ]
            cols = st.columns(3)
            for i, col_name in enumerate(numeric_display[:9]):
                value = customer[col_name]
                cols[i % 3].metric(
                    col_name,
                    f"{value:,.2f}" if isinstance(value, (float, np.floating)) else f"{value:,}"
                )
            if "CustomerSegment" in result.columns:
                st.markdown(
                    f'<div class="card"><b>Customer segment:</b> {customer["CustomerSegment"]}</div>',
                    unsafe_allow_html=True,
                )

            if st.session_state.cluster_result and st.session_state.cluster_result.get("profile") is not None:
                cluster_id = customer.get("ClusterID")
                profile = st.session_state.cluster_result["profile"]
                if pd.notna(cluster_id) and int(cluster_id) in profile.index:
                    avg_row = profile.loc[int(cluster_id)]
                    compare_cols = [c for c in avg_row.index if c in result.columns]
                    cdf = pd.DataFrame({
                        "Feature": compare_cols,
                        "This Customer": [customer.get(c, "N/A") for c in compare_cols],
                        "Segment Average": [round(avg_row[c], 2) for c in compare_cols],
                    })
                    st.markdown('<div class="section-header">Comparison with segment average</div>', unsafe_allow_html=True)
                    st.dataframe(cdf, width="stretch", hide_index=True)
        elif len(result) > 1:
            st.info("Narrow your search to a single customer to see their detailed profile.")


elif current_page == "Customer Segmentation":
    st.markdown('<div class="page-title">🎯 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Discover customer groups using K-Means clustering and compare their characteristics.</div>', unsafe_allow_html=True)

    num_cols = get_numeric_columns(df_active)
    default_feats = [
        c for c in ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency", "AverageOrderValue"]
        if c in num_cols
    ]

    with st.expander("⚙️ Clustering configuration", expanded=True):
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            selected_features = st.multiselect(
                "Features for clustering",
                options=num_cols,
                default=default_feats if default_feats else num_cols[:4],
                help="Select at least two numeric features.",
            )
        with scol2:
            n_clusters = st.slider("Number of clusters (K)", 2, 8, 4)
        run_btn = st.button("▶ Run segmentation", type="primary", width="stretch")

    if run_btn:
        if len(selected_features) < 2:
            st.error("Please select at least two features.")
        else:
            with st.spinner("Training K-Means model…"):
                result = run_kmeans(df_active, selected_features, n_clusters)
            if result.get("error"):
                st.error(result["error"])
            else:
                labeled_df = label_clusters(
                    df_active, result["labels"], result["valid_indices"],
                    result["profile"], selected_features,
                )
                st.session_state.cluster_result = result
                st.session_state.cluster_df = labeled_df
                st.session_state.cluster_summary = get_cluster_summary(labeled_df)
                st.success(f"Segmentation complete — {n_clusters} clusters identified.")

    cr = st.session_state.cluster_result
    cdf = st.session_state.cluster_df
    cs = st.session_state.cluster_summary

    if cr and cdf is not None and not cr.get("error"):
        sil = cr.get("silhouette")
        if sil is not None:
            quality = "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Weak"
            st.markdown(
                f'<div class="card"><b>Silhouette score:</b> {sil:.3f} &nbsp; · &nbsp; {quality} cluster separation</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Silhouette score was not available for this configuration.")

        st.markdown('<div class="section-header">Cluster distribution</div>', unsafe_allow_html=True)
        size_col1, size_col2 = st.columns([1, 2])
        with size_col1:
            if cs:
                size_data = pd.DataFrame([
                    {"Segment": s["label"], "Customers": s["size"], "Share (%)": f'{s["pct"]:.1f}%'}
                    for s in cs["segments"]
                ])
                st.dataframe(size_data, width="stretch", hide_index=True)
        with size_col2:
            fig_bar = viz.cluster_size_bar(cdf)
            if fig_bar:
                st.plotly_chart(fig_bar, width="stretch")

        st.markdown('<div class="section-header">Cluster visualization</div>', unsafe_allow_html=True)
        feat_list = cr.get("feature_cols", [])
        if feat_list:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                x_axis = st.selectbox("X axis", feat_list, index=0)
            with viz_col2:
                y_axis = st.selectbox("Y axis", feat_list, index=min(1, len(feat_list) - 1))
            fig_scatter = viz.cluster_scatter(cdf, x_axis, y_axis)
            if fig_scatter:
                st.plotly_chart(fig_scatter, width="stretch")

        st.markdown('<div class="section-header">Cluster profiles</div>', unsafe_allow_html=True)
        profile = cr.get("profile")
        if profile is not None:
            profile_display = profile.copy()
            if cs:
                seg_names = {s["cluster_id"]: s["label"] for s in cs["segments"]}
                profile_display.index = [seg_names.get(i, f"Cluster {i}") for i in profile_display.index]
            st.dataframe(profile_display.style.background_gradient(cmap="Blues", axis=0), width="stretch")

        st.markdown('<div class="section-header">Export results</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download segmented dataset (CSV)",
            data=cdf.to_csv(index=False).encode("utf-8"),
            file_name="customer_segments.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("Configure the clustering settings and click Run segmentation.")


elif current_page == "Predictive Analytics":
    st.markdown('<div class="page-title">📈 Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Train a Random Forest model and estimate customer spending scores.</div>', unsafe_allow_html=True)

    TARGET_COL = "SpendingScore"
    feature_cols = get_suitable_features(df_active, TARGET_COL)

    if TARGET_COL not in df_active.columns:
        st.warning(f"The active dataset does not contain '{TARGET_COL}'. Prediction is unavailable.")
    elif len(df_active) < 30:
        st.warning("At least 30 records are needed for prediction training.")
    else:
        with st.expander("⚙️ Model configuration", expanded=True):
            selected_pred_features = st.multiselect(
                "Feature columns",
                options=feature_cols,
                default=feature_cols,
                help="Numeric columns used to predict Spending Score.",
            )
            train_btn = st.button("▶ Train model", type="primary", width="stretch")

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
                    st.success("Model trained successfully.")

        pr = st.session_state.pred_result
        if pr and not pr.get("error"):
            st.markdown('<div class="section-header">Model evaluation</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{pr['mae']:.3f}")
            m2.metric("R²", f"{pr['r2']:.4f}")
            m3.metric("Training rows", f"{pr['n_train']:,}")
            m4.metric("Test rows", f"{pr['n_test']:,}")
            st.caption("Metrics are calculated on held-out test data. Results are for demonstration.")

            col_a, col_b = st.columns(2)
            with col_a:
                fig_avp = viz.actual_vs_predicted(pr["y_test"], pr["y_pred"], TARGET_COL)
                st.plotly_chart(fig_avp, width="stretch")
            with col_b:
                if pr.get("feature_importance") is not None:
                    fig_fi = viz.feature_importance_bar(pr["feature_importance"])
                    st.plotly_chart(fig_fi, width="stretch")

            st.markdown('<div class="section-header">Predict a new customer spending score</div>', unsafe_allow_html=True)
            st.write("Enter customer attributes to generate an estimate.")
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
                        max_value=max(col_min, col_max),
                        value=min(max(col_mean, col_min), max(col_min, col_max)),
                        step=max((col_max - col_min) / 100, 0.1),
                        key=f"pred_input_{col_name}",
                    )

            if st.button("🔮 Predict spending score", type="primary"):
                pred_val = predict_single(pr["model"], pr["feature_cols"], input_vals)
                if pred_val is not None:
                    st.markdown(
                        f'<div class="card"><div style="color:#1E40AF;font-weight:700;">Estimated spending score</div><div style="font-size:2.5rem;font-weight:800;color:#1D4ED8;">{pred_val:.1f}</div><div style="color:#526176;font-size:.85rem;">This is a model estimate, not a guarantee.</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Prediction failed. Check the input values.")
        elif pr and pr.get("error"):
            st.error(pr["error"])
        else:
            st.info("Configure your features and click Train model.")


elif current_page == "AI Analytics Assistant":
    st.markdown('<div class="page-title">🤖 AI Analytics Assistant</div>', unsafe_allow_html=True)
    mode_badge = "🟢 OpenAI LLM available" if _openai_available else "🟡 Built-in assistant"
    st.markdown(
        f'<div class="page-subtitle">Ask questions about your current customer data, segments, and model results. {mode_badge}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Suggested questions</div>', unsafe_allow_html=True)
    sq_cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with sq_cols[i % 3]:
            if st.button(q, key=f"sq_{i}", width="stretch"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Analyzing your data…"):
                    response, mode = respond(q, _agent_state(), st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": response, "mode": mode})
                st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem 1.5rem;">
            <div style="font-size:2.5rem;">💬</div>
            <div style="font-weight:700;font-size:1.1rem;color:#172B4D;">Your Customer Intelligence Agent</div>
            <div style="color:#526176;margin-top:.5rem;">Ask about customer behaviour, segment characteristics, or predictive results.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                mode_text = "🟢 LLM" if msg.get("mode") == "llm" else "🟡 Built-in"
                st.markdown(
                    f'<div class="chat-assistant">{content}<br><span class="mode-badge">{mode_text}</span></div>',
                    unsafe_allow_html=True,
                )

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            label_visibility="collapsed",
            placeholder="Ask about customer segments or predictive analytics…",
        )
        form_c1, form_c2 = st.columns([5, 1])
        with form_c1:
            submitted = st.form_submit_button("Send ➤", type="primary", width="stretch")
        with form_c2:
            cleared = st.form_submit_button("Clear", width="stretch")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Analyzing your question…"):
            response, mode = respond(user_input.strip(), _agent_state(), st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": response, "mode": mode})
        st.rerun()

    if cleared:
        st.session_state.chat_history = []
        st.rerun()


elif current_page == "Dataset Management":
    st.markdown('<div class="page-title">📂 Dataset Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a customer CSV, inspect data quality, or reset the dataset.</div>', unsafe_allow_html=True)

    sample_df = generate_sample_data()
    st.download_button(
        "⬇ Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_customers.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown('<div class="section-header">Upload your dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        df_upload, err = validate_and_load_csv(uploaded)
        if err:
            st.error(f"Upload failed: {err}")
        else:
            num_available = get_numeric_columns(df_upload)
            st.success(f"File loaded: {len(df_upload):,} rows × {len(df_upload.columns)} columns.")
            missing_standard = [c for c in ["AnnualIncome", "SpendingScore"] if c not in df_upload.columns]

            if missing_standard:
                st.warning(f"Standard columns not found: {', '.join(missing_standard)}")
                col_map = {}
                for std_col in missing_standard:
                    col_map[std_col] = st.selectbox(
                        f"Map '{std_col}' to",
                        options=["— skip —"] + num_available,
                        key=f"map_{std_col}",
                    )
                if st.button("Apply column mapping", width="stretch"):
                    for std_col, mapped in col_map.items():
                        if mapped != "— skip —":
                            df_upload[std_col] = df_upload[mapped]
                    _set_dataframe(df_upload, is_sample=False)
                    st.success("Dataset updated.")
                    st.rerun()
            elif st.button("Use this dataset", type="primary", width="stretch"):
                _set_dataframe(df_upload, is_sample=False)
                st.success("Dataset updated.")
                st.rerun()

            st.markdown("**Preview**")
            st.dataframe(df_upload.head(10), width="stretch")

    st.markdown('<div class="section-header">Data quality report</div>', unsafe_allow_html=True)
    dq = st.session_state.data_quality
    if dq:
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Records", f"{dq['total_records']:,}")
        qc2.metric("Columns", dq["total_columns"])
        qc3.metric("Missing values", dq["missing_values_total"])
        qc4.metric("Duplicate rows", dq["duplicate_rows"])

        ctype_col1, ctype_col2 = st.columns(2)
        with ctype_col1:
            st.markdown("**Numeric columns**")
            st.write(", ".join(dq["numeric_columns"]) if dq["numeric_columns"] else "None")
        with ctype_col2:
            st.markdown("**Categorical columns**")
            st.write(", ".join(dq["categorical_columns"]) if dq["categorical_columns"] else "None")

        if dq["missing_by_column"]:
            st.markdown("**Missing values by column**")
            miss_df = pd.DataFrame(list(dq["missing_by_column"].items()), columns=["Column", "Missing Count"])
            st.dataframe(miss_df, width="stretch", hide_index=True)
        else:
            st.markdown('<span class="status-ok">✓ No missing values detected</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Reset dataset</div>', unsafe_allow_html=True)
    if not st.session_state.is_sample_data:
        if st.button("🔄 Reset to sample data", width="stretch"):
            load_sample()
            st.success("Reset complete. Previous models and results were cleared.")
            st.rerun()
    else:
        st.markdown('<span class="status-ok">✓ Currently using sample data</span>', unsafe_allow_html=True)


elif current_page == "About the Project":
    st.markdown('<div class="page-title">ℹ️ About the Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div style="font-size:1.45rem;font-weight:800;color:#1E40AF;margin-bottom:.6rem;">
        🧠 Customer Intelligence AI
      </div>
      <div style="color:#526176;line-height:1.8;">
        An intelligent customer segmentation and predictive analytics dashboard
        developed for the BICS Global internship project in the Data Science domain.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🛠️ Technology stack

        | Component | Technology |
        |---|---|
        | Web interface | Streamlit |
        | Data processing | Pandas, NumPy |
        | Machine learning | Scikit-learn |
        | Visualizations | Plotly |
        | AI assistant | Built-in + optional OpenAI |
        | Configuration | python-dotenv |
        """)
    with col2:
        st.markdown("""
        ### ✅ Features

        - Interactive KPI dashboard
        - Customer exploration and filtering
        - K-Means customer segmentation
        - Random Forest spending prediction
        - AI analytics assistant
        - CSV upload and data-quality reporting
        - Segmented dataset export
        - Reproducible sample dataset
        """)

    st.markdown('<div class="section-header">Five-minute demo plan</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Show the dashboard KPIs and customer visualizations.
    2. Explain the sample customer dataset.
    3. Run K-Means segmentation and inspect the cluster profiles.
    4. Train the predictive model and explain MAE and R².
    5. Ask the AI assistant to summarize the segments.
    6. Demonstrate CSV upload and data quality.
    7. Download the segmented customer dataset.
    """)

    st.markdown('<div class="section-header">Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
    - Sample results use synthetic data and are for demonstration.
    - Predictive results depend on the selected dataset and features.
    - Built-in AI responses may be more limited than LLM-generated responses.
    - This local application does not implement authentication or multi-user access.
    """)

    st.markdown("""
    <div class="card">
      <b>Launch commands</b><br><br>
      <code>pip install -r requirements.txt</code><br>
      <code>streamlit run app.py</code>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;color:#64748B;font-size:.78rem;margin-top:2.5rem;
border-top:1px solid #E1E8F0;padding:1.25rem 0;">
Customer Intelligence AI · BICS Global Internship · Built with Streamlit
</div>
""", unsafe_allow_html=True)
