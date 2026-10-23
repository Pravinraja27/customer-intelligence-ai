"""
analytics.py
KPI calculations and dashboard analytics derived from the active dataset.
"""
import numpy as np
import pandas as pd
from src.data_loader import get_numeric_columns, get_categorical_columns


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate the core KPI cards shown on the dashboard overview.
    All values are derived from the actual dataset.
    """
    kpis = {"total_customers": len(df)}

    def _col_mean(col_name: str, fmt: str = ".0f") -> str | None:
        if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            val = df[col_name].dropna().mean()
            if not np.isnan(val):
                return f"{val:{fmt}}"
        return None

    kpis["avg_annual_income"] = _col_mean("AnnualIncome", ",.0f")
    kpis["avg_spending_score"] = _col_mean("SpendingScore", ".1f")
    kpis["avg_order_value"] = _col_mean("AverageOrderValue", ",.0f")
    kpis["avg_purchase_freq"] = _col_mean("PurchaseFrequency", ".1f")
    kpis["avg_satisfaction"] = _col_mean("CustomerSatisfaction", ".2f")
    kpis["avg_recency_days"] = _col_mean("RecencyDays", ".0f")

    return kpis


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Return a Pearson correlation matrix for numeric columns.
    Returns None when fewer than 2 numeric columns exist.
    """
    num_cols = get_numeric_columns(df)
    if len(num_cols) < 2:
        return None
    return df[num_cols].corr()


def get_gender_distribution(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return value counts for the Gender column, or None if absent."""
    if "Gender" in df.columns:
        vc = df["Gender"].value_counts().reset_index()
        vc.columns = ["Gender", "Count"]
        return vc
    return None


def get_age_bins(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame | None:
    """Return a histogram-friendly DataFrame for Age, or None if Age is absent."""
    if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
        series = df["Age"].dropna()
        counts, edges = np.histogram(series, bins=bins)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
        return pd.DataFrame({"AgeRange": labels, "Count": counts})
    return None


def get_income_vs_spending(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return rows with both AnnualIncome and SpendingScore, for a scatter plot."""
    needed = ["AnnualIncome", "SpendingScore"]
    if all(c in df.columns for c in needed):
        sub = df[needed + (["Age"] if "Age" in df.columns else []) +
                  (["Gender"] if "Gender" in df.columns else []) +
                  (["CustomerID"] if "CustomerID" in df.columns else [])].dropna(subset=needed)
        return sub
    return None


def get_purchase_frequency_bins(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a histogram-friendly DataFrame for PurchaseFrequency."""
    col = "PurchaseFrequency"
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        series = df[col].dropna()
        counts, edges = np.histogram(series, bins=8)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
        return pd.DataFrame({"FrequencyRange": labels, "Count": counts})
    return None


def summarize_dataset(df: pd.DataFrame, cluster_summary: dict | None = None) -> str:
    """
    Build a concise text summary of the dataset used by the AI assistant.
    """
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    lines = [
        f"Dataset: {len(df)} customers, {len(df.columns)} columns.",
        f"Numeric columns: {', '.join(num_cols) if num_cols else 'None'}.",
        f"Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}.",
    ]

    for col in ["AnnualIncome", "SpendingScore", "Age", "PurchaseFrequency", "AverageOrderValue"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            lines.append(
                f"{col}: mean={s.mean():.1f}, min={s.min():.0f}, max={s.max():.0f}."
            )

    if "Gender" in df.columns:
        vc = df["Gender"].value_counts()
        lines.append("Gender breakdown: " + ", ".join(f"{k}={v}" for k, v in vc.items()) + ".")

    if cluster_summary:
        lines.append(f"Clustering: {cluster_summary.get('n_clusters')} clusters identified.")
        for seg in cluster_summary.get("segments", []):
            lines.append(
                f"  - {seg['label']} ({seg['size']} customers, {seg['pct']:.1f}%): "
                + ", ".join(f"{k}={v}" for k, v in seg["profile"].items())
            )

    return "\n".join(lines)
