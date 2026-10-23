"""
visualizations.py
Plotly chart builders for the dashboard.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── consistent colour palette ──────────────────────────────────
PALETTE = px.colors.qualitative.Bold
CLUSTER_COLORS = px.colors.qualitative.Vivid
BG = "rgba(0,0,0,0)"
FONT_COLOR = "#1e293b"
GRID_COLOR = "rgba(100,116,139,0.15)"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", color=FONT_COLOR, size=12),
    margin=dict(l=20, r=20, t=55, b=20),
    legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=14, color="#1e40af"), x=0.01, xanchor="left"))
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def gender_pie(df: pd.DataFrame) -> go.Figure | None:
    if "Gender" not in df.columns:
        return None
    vc = df["Gender"].value_counts().reset_index()
    vc.columns = ["Gender", "Count"]
    fig = px.pie(
        vc, names="Gender", values="Count",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return _apply_base(fig, "Customer Gender Distribution")


def income_vs_spending_scatter(df: pd.DataFrame) -> go.Figure | None:
    needed = ["AnnualIncome", "SpendingScore"]
    if not all(c in df.columns for c in needed):
        return None

    color_col = "CustomerSegment" if "CustomerSegment" in df.columns else (
        "Gender" if "Gender" in df.columns else None
    )
    hover_data = [c for c in ["CustomerID", "Age", "PurchaseFrequency"] if c in df.columns]

    fig = px.scatter(
        df.dropna(subset=needed),
        x="AnnualIncome",
        y="SpendingScore",
        color=color_col,
        hover_data=hover_data,
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.75,
        size_max=10,
    )
    fig.update_traces(marker=dict(size=7))
    return _apply_base(fig, "Annual Income vs Spending Score")


def age_histogram(df: pd.DataFrame) -> go.Figure | None:
    if "Age" not in df.columns:
        return None
    fig = px.histogram(
        df.dropna(subset=["Age"]),
        x="Age",
        nbins=20,
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _apply_base(fig, "Customer Age Distribution")


def purchase_freq_histogram(df: pd.DataFrame) -> go.Figure | None:
    col = "PurchaseFrequency"
    if col not in df.columns:
        return None
    fig = px.histogram(
        df.dropna(subset=[col]),
        x=col,
        nbins=15,
        color_discrete_sequence=["#8b5cf6"],
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _apply_base(fig, "Purchase Frequency Distribution")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"ClusterID"}
    num = [c for c in num if c not in exclude]
    if len(num) < 2:
        return None
    corr = df[num].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    return _apply_base(fig, "Feature Correlation Heatmap")


def cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    if "ClusterID" not in df.columns:
        return None

    plot_df = df[[x_col, y_col, "ClusterID"]].dropna().copy()
    if "CustomerSegment" in df.columns:
        plot_df["CustomerSegment"] = df.loc[plot_df.index, "CustomerSegment"]
        color_col = "CustomerSegment"
    else:
        plot_df["ClusterID"] = plot_df["ClusterID"].astype(str)
        color_col = "ClusterID"

    hover_cols = [c for c in ["CustomerID", "Age", "AnnualIncome"] if c in df.columns]
    for c in hover_cols:
        plot_df[c] = df.loc[plot_df.index, c]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=8))
    return _apply_base(fig, f"Customer Clusters: {x_col} vs {y_col}")


def actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, target_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            marker=dict(color="#3b82f6", size=6, opacity=0.7),
            name="Predictions",
        )
    )
    mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect Fit",
        )
    )
    fig.update_layout(
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
    )
    return _apply_base(fig, "Actual vs Predicted Values")


def feature_importance_bar(importance: pd.Series) -> go.Figure:
    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        color=importance.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return _apply_base(fig, "Feature Importance")


def cluster_size_bar(df: pd.DataFrame) -> go.Figure | None:
    if "CustomerSegment" not in df.columns:
        return None
    vc = df["CustomerSegment"].value_counts().reset_index()
    vc.columns = ["Segment", "Count"]
    fig = px.bar(
        vc, x="Count", y="Segment", orientation="h",
        color="Segment",
        color_discrete_sequence=CLUSTER_COLORS,
    )
    fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
    return _apply_base(fig, "Customers per Segment")
