"""
visualizations.py
Modern Plotly chart builders for the Customer Intelligence AI dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# LIGHT THEME / DESIGN TOKENS
# ============================================================

PALETTE = [
    "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B",
    "#EC4899", "#8B5CF6", "#F97316", "#14B8A6",
]
CLUSTER_COLORS = [
    "#4F46E5", "#06B6D4", "#10B981", "#F59E0B",
    "#EC4899", "#8B5CF6", "#F97316", "#64748B",
]

BG = "#FFFFFF"
PLOT_BG = "#FFFFFF"
FONT_COLOR = "#1E293B"
MUTED_COLOR = "#64748B"
GRID_COLOR = "#E9EEF5"
BORDER_COLOR = "#E2E8F0"

LAYOUT_BASE = {
    "template": "plotly_white",
    "paper_bgcolor": BG,
    "plot_bgcolor": PLOT_BG,
    "font": {"family": "Inter, Arial, sans-serif", "color": FONT_COLOR, "size": 12},
    "margin": {"l": 24, "r": 24, "t": 65, "b": 30},
    "hoverlabel": {
        "bgcolor": "#FFFFFF",
        "bordercolor": BORDER_COLOR,
        "font": {"family": "Inter, Arial, sans-serif", "size": 12, "color": FONT_COLOR},
        "align": "left",
    },
    "legend": {
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": BORDER_COLOR,
        "borderwidth": 0,
        "font": {"size": 11, "color": FONT_COLOR},
    },
    "title": {
        "font": {"size": 16, "color": FONT_COLOR},
        "x": 0.02,
        "xanchor": "left",
        "y": 0.97,
        "yanchor": "top",
    },
    "hovermode": "closest",
    "autosize": True,
}


# ============================================================
# SHARED HELPERS
# ============================================================

def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared theme without passing duplicate layout arguments."""
    layout = LAYOUT_BASE.copy()
    layout["title"] = {
        "text": title,
        "font": {"size": 16, "color": FONT_COLOR},
        "x": 0.02,
        "xanchor": "left",
        "y": 0.97,
        "yanchor": "top",
    }
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=BORDER_COLOR,
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=BORDER_COLOR,
        automargin=True,
    )
    return fig


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check that a DataFrame contains all required columns."""
    return isinstance(df, pd.DataFrame) and all(c in df.columns for c in columns)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a column to numeric values, coercing invalid values to NaN."""
    return pd.to_numeric(df[column], errors="coerce")


def _empty_figure(message: str, title: str = "") -> go.Figure:
    """Build a themed placeholder when the data cannot produce a chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": MUTED_COLOR},
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_base(fig, title)


# ============================================================
# CUSTOMER DEMOGRAPHICS
# ============================================================

def gender_pie(df: pd.DataFrame) -> go.Figure | None:
    """Display customer distribution by gender."""
    if not _has_columns(df, ["Gender"]):
        return None

    genders = df["Gender"].dropna().astype(str)
    genders = genders[genders.str.strip() != ""]
    if genders.empty:
        return _empty_figure("No gender data available.", "Customer Gender Distribution")

    counts = genders.value_counts().rename_axis("Gender").reset_index(name="Count")
    fig = px.pie(
        counts,
        names="Gender",
        values="Count",
        hole=0.62,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont={"size": 12, "color": "#FFFFFF"},
        marker={"line": {"color": "#FFFFFF", "width": 3}},
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        sort=False,
    )
    return _apply_base(fig, "Customer Gender Distribution")


# ============================================================
# INCOME AND SPENDING
# ============================================================

def income_vs_spending_scatter(df: pd.DataFrame) -> go.Figure | None:
    """Plot annual income against spending score."""
    needed = ["AnnualIncome", "SpendingScore"]
    if not _has_columns(df, needed):
        return None

    plot_df = df.copy()
    for column in needed:
        plot_df[column] = _numeric_series(plot_df, column)
    plot_df = plot_df.dropna(subset=needed)

    if plot_df.empty:
        return _empty_figure(
            "No valid income and spending data available.",
            "Annual Income vs Spending Score",
        )

    color_col = (
        "CustomerSegment" if "CustomerSegment" in plot_df.columns
        else "Gender" if "Gender" in plot_df.columns
        else None
    )
    hover_data = [
        c for c in ["CustomerID", "Age", "PurchaseFrequency"]
        if c in plot_df.columns
    ]

    fig = px.scatter(
        plot_df,
        x="AnnualIncome",
        y="SpendingScore",
        color=color_col,
        hover_data=hover_data,
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.82,
    )
    fig.update_traces(
        marker={"size": 9, "line": {"width": 1, "color": "rgba(255,255,255,0.9)"}}
    )
    fig.update_layout(
        xaxis_title="Annual Income",
        yaxis_title="Spending Score",
        legend_title_text="Customer Group",
    )
    return _apply_base(fig, "Annual Income vs Spending Score")


# ============================================================
# AGE DISTRIBUTION
# ============================================================

def age_histogram(df: pd.DataFrame) -> go.Figure | None:
    """Display the age distribution of customers."""
    if not _has_columns(df, ["Age"]):
        return None

    plot_df = pd.DataFrame({"Age": _numeric_series(df, "Age")}).dropna()
    if plot_df.empty:
        return _empty_figure("No valid age data available.", "Customer Age Distribution")

    fig = px.histogram(
        plot_df,
        x="Age",
        nbins=20,
        color_discrete_sequence=["#4F46E5"],
        opacity=0.95,
    )
    fig.update_traces(
        marker={"line": {"color": "#FFFFFF", "width": 1.5}},
        hovertemplate="Age: %{x}<br>Customers: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Customers",
        bargap=0.12,
    )
    return _apply_base(fig, "Customer Age Distribution")


# ============================================================
# PURCHASE FREQUENCY
# ============================================================

def purchase_freq_histogram(df: pd.DataFrame) -> go.Figure | None:
    """Display the distribution of purchase frequency."""
    column = "PurchaseFrequency"
    if not _has_columns(df, [column]):
        return None

    plot_df = pd.DataFrame({column: _numeric_series(df, column)}).dropna()
    if plot_df.empty:
        return _empty_figure(
            "No valid purchase frequency data available.",
            "Purchase Frequency Distribution",
        )

    fig = px.histogram(
        plot_df,
        x=column,
        nbins=15,
        color_discrete_sequence=["#8B5CF6"],
        opacity=0.95,
    )
    fig.update_traces(
        marker={"line": {"color": "#FFFFFF", "width": 1.5}},
        hovertemplate="Purchase Frequency: %{x}<br>Customers: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Purchase Frequency",
        yaxis_title="Number of Customers",
        bargap=0.12,
    )
    return _apply_base(fig, "Purchase Frequency Distribution")


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Display correlations between numeric features."""
    if not isinstance(df, pd.DataFrame):
        return None

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.drop(columns=["ClusterID"], errors="ignore")
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    if numeric_df.shape[1] < 2:
        return _empty_figure(
            "At least two varying numeric features are required.",
            "Feature Correlation Heatmap",
        )

    corr = numeric_df.corr().replace([np.inf, -np.inf], np.nan)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, "#DC2626"],
                [0.25, "#FCA5A5"],
                [0.5, "#F8FAFC"],
                [0.75, "#93C5FD"],
                [1.0, "#1D4ED8"],
            ],
            zmin=-1,
            zmax=1,
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": FONT_COLOR},
            colorbar={"title": {"text": "Correlation"}, "thickness": 12, "outlinewidth": 0},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        xaxis={"tickangle": -35, "side": "bottom"},
        yaxis={"autorange": "reversed"},
    )
    return _apply_base(fig, "Feature Correlation Heatmap")


# ============================================================
# CUSTOMER CLUSTERS
# ============================================================

def cluster_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> go.Figure | None:
    """Visualize customer clusters using two selected numeric features."""
    if not _has_columns(df, [x_col, y_col, "ClusterID"]):
        return None

    plot_df = df.copy()
    plot_df[x_col] = _numeric_series(plot_df, x_col)
    plot_df[y_col] = _numeric_series(plot_df, y_col)
    plot_df = plot_df.dropna(subset=[x_col, y_col, "ClusterID"])

    chart_title = f"Customer Clusters: {x_col} vs {y_col}"
    if plot_df.empty:
        return _empty_figure("No valid clustered customer records are available.", chart_title)

    if "CustomerSegment" in plot_df.columns:
        color_col = "CustomerSegment"
    else:
        plot_df["ClusterID"] = plot_df["ClusterID"].astype(str)
        color_col = "ClusterID"

    hover_cols = [
        c for c in [
            "CustomerID", "Age", "AnnualIncome",
            "SpendingScore", "PurchaseFrequency",
        ]
        if c in plot_df.columns and c not in [x_col, y_col]
    ]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.88,
    )
    fig.update_traces(
        marker={"size": 10, "line": {"width": 1, "color": "#FFFFFF"}}
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title_text="Customer Segment",
    )
    return _apply_base(fig, chart_title)


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

def actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_col: str,
) -> go.Figure:
    """Compare actual target values with model predictions."""
    actual = np.asarray(y_test, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(actual) != len(predicted):
        return _empty_figure(
            "Actual and predicted values have different lengths.",
            "Actual vs Predicted Values",
        )

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return _empty_figure(
            "No valid predictions are available to display.",
            "Actual vs Predicted Values",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            name="Predictions",
            marker={
                "color": "#4F46E5",
                "size": 8,
                "opacity": 0.75,
                "line": {"color": "#FFFFFF", "width": 1},
            },
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        )
    )

    minimum = float(min(actual.min(), predicted.min()))
    maximum = float(max(actual.max(), predicted.max()))
    if minimum == maximum:
        padding = abs(minimum) * 0.05 or 1.0
        minimum -= padding
        maximum += padding

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect Prediction",
            line={"color": "#EF4444", "dash": "dash", "width": 2},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
        legend_title_text="Model Evaluation",
    )
    return _apply_base(fig, "Actual vs Predicted Values")


def feature_importance_bar(importance: pd.Series) -> go.Figure:
    """Display model feature importance as a horizontal bar chart."""
    if importance is None or len(importance) == 0:
        return _empty_figure("Feature importance is not available.", "Feature Importance")

    values = pd.to_numeric(importance, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return _empty_figure(
            "No valid feature importance values are available.",
            "Feature Importance",
        )

    values = values.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=values.values,
            y=values.index.astype(str),
            orientation="h",
            marker={
                "color": values.values,
                "colorscale": [
                    [0.0, "#C7D2FE"],
                    [0.5, "#818CF8"],
                    [1.0, "#4338CA"],
                ],
                "line": {"width": 0},
            },
            text=[f"{v:.3f}" for v in values.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="Feature")
    return _apply_base(fig, "Feature Importance")


# ============================================================
# SEGMENT SIZES
# ============================================================

def cluster_size_bar(df: pd.DataFrame) -> go.Figure | None:
    """Compare the number of customers in each segment."""
    column = "CustomerSegment"
    if not _has_columns(df, [column]):
        return None

    segments = df[column].dropna().astype(str)
    if segments.empty:
        return _empty_figure(
            "No customer segment labels are available.",
            "Customers per Segment",
        )

    counts = (
        segments.value_counts()
        .rename_axis("Segment")
        .reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )

    fig = px.bar(
        counts,
        x="Count",
        y="Segment",
        orientation="h",
        color="Segment",
        color_discrete_sequence=CLUSTER_COLORS,
        text="Count",
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Customers: %{x:,}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Number of Customers",
        yaxis_title="Customer Segment",
        bargap=0.28,
    )
    return _apply_base(fig, "Customers per Segment")
