"""
clustering.py
K-Means customer segmentation using scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 4,
) -> dict:
    """
    Fit K-Means on the selected numeric features.
    Returns a result dict with labels, model, scaler, profile, silhouette_score, and error.
    """
    result = {
        "labels": None,
        "model": None,
        "scaler": None,
        "profile": None,
        "silhouette": None,
        "error": None,
        "n_clusters": n_clusters,
        "feature_cols": feature_cols,
    }

    # Validation
    if not feature_cols:
        result["error"] = "No features selected for clustering."
        return result

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        result["error"] = f"Columns not found in dataset: {missing}"
        return result

    X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce")
    X.dropna(inplace=True)

    if len(X) < n_clusters * 2:
        result["error"] = (
            f"Not enough valid rows ({len(X)}) for {n_clusters} clusters. "
            "Reduce clusters or clean your data."
        )
        return result

    if n_clusters < 2:
        result["error"] = "Number of clusters must be at least 2."
        return result

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Silhouette (requires >= 2 clusters and at least 2 labels per class)
    sil = None
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2:
        try:
            sil = float(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X))))
        except Exception:
            sil = None

    # Profile: mean of original (unscaled) features per cluster
    profile_df = X.copy()
    profile_df["_Cluster"] = labels
    profile = profile_df.groupby("_Cluster")[feature_cols].mean().round(2)

    result["labels"] = labels
    result["model"] = km
    result["scaler"] = scaler
    result["profile"] = profile
    result["silhouette"] = sil
    result["valid_indices"] = X.index.tolist()

    return result


def label_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    valid_indices: list,
    profile: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Attach cluster labels and human-readable descriptions to the original dataframe.
    Returns the annotated dataframe.
    """
    df = df.copy()
    df["ClusterID"] = np.nan
    df.loc[valid_indices, "ClusterID"] = labels
    df["ClusterID"] = df["ClusterID"].astype("Int64")

    descriptions = _generate_cluster_descriptions(profile, feature_cols)
    df["CustomerSegment"] = df["ClusterID"].map(
        lambda x: descriptions.get(int(x), f"Cluster {x}") if pd.notna(x) else "Unassigned"
    )
    return df


def _generate_cluster_descriptions(profile: pd.DataFrame, feature_cols: list[str]) -> dict[int, str]:
    """
    Derive human-readable segment names from actual cluster profile averages.
    Logic is purely data-driven — no fixed assumptions about cluster order.
    """
    descriptions = {}

    has_spending = "SpendingScore" in feature_cols
    has_income = "AnnualIncome" in feature_cols
    has_freq = "PurchaseFrequency" in feature_cols
    has_recency = "RecencyDays" in feature_cols

    for cluster_id, row in profile.iterrows():
        tags = []

        if has_spending:
            spend = row.get("SpendingScore", 50)
            spend_max = profile["SpendingScore"].max()
            spend_min = profile["SpendingScore"].min()
            spend_range = spend_max - spend_min if spend_max != spend_min else 1
            spend_norm = (spend - spend_min) / spend_range
            if spend_norm >= 0.66:
                tags.append("High-Spend")
            elif spend_norm <= 0.33:
                tags.append("Low-Spend")
            else:
                tags.append("Mid-Spend")

        if has_income:
            inc = row.get("AnnualIncome", 50000)
            inc_max = profile["AnnualIncome"].max()
            inc_min = profile["AnnualIncome"].min()
            inc_range = inc_max - inc_min if inc_max != inc_min else 1
            inc_norm = (inc - inc_min) / inc_range
            if inc_norm >= 0.66:
                tags.append("High-Income")
            elif inc_norm <= 0.33:
                tags.append("Low-Income")

        if has_freq:
            freq = row.get("PurchaseFrequency", 5)
            freq_max = profile["PurchaseFrequency"].max()
            freq_min = profile["PurchaseFrequency"].min()
            freq_range = freq_max - freq_min if freq_max != freq_min else 1
            freq_norm = (freq - freq_min) / freq_range
            if freq_norm >= 0.66:
                tags.append("Frequent")
            elif freq_norm <= 0.33:
                tags.append("Infrequent")

        if has_recency:
            rec = row.get("RecencyDays", 30)
            rec_max = profile["RecencyDays"].max()
            rec_min = profile["RecencyDays"].min()
            rec_range = rec_max - rec_min if rec_max != rec_min else 1
            rec_norm = (rec - rec_min) / rec_range
            if rec_norm <= 0.33:
                tags.append("Recent")
            elif rec_norm >= 0.66:
                tags.append("Lapsed")

        label = " ".join(tags) if tags else f"Cluster {cluster_id}"
        # Simplify common combos
        replacements = {
            "High-Spend High-Income Frequent Recent": "High-Value VIPs",
            "High-Spend High-Income Frequent": "High-Value Frequent",
            "High-Spend Frequent Recent": "Active Premium",
            "Low-Spend Low-Income Infrequent Lapsed": "At-Risk Budget",
            "Low-Spend Infrequent Lapsed": "Churned Budget",
            "Mid-Spend Frequent Recent": "Loyal Mid-Tier",
            "High-Spend Frequent": "Premium Regulars",
            "Low-Spend Low-Income": "Budget Shoppers",
        }
        label = replacements.get(label, label)
        descriptions[int(cluster_id)] = label

    return descriptions


def get_cluster_summary(df: pd.DataFrame) -> dict | None:
    """
    Build a cluster summary dict for the AI assistant.
    Returns None if clustering hasn't been run.
    """
    if "ClusterID" not in df.columns:
        return None

    labeled = df[df["ClusterID"].notna()].copy()
    n_clusters = labeled["ClusterID"].nunique()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["ClusterID"]
    profile_cols = [c for c in numeric_cols if c not in exclude][:6]

    segments = []
    for cid in sorted(labeled["ClusterID"].unique()):
        group = labeled[labeled["ClusterID"] == cid]
        seg_label = group["CustomerSegment"].iloc[0] if "CustomerSegment" in group.columns else f"Cluster {cid}"
        profile_vals = {c: round(float(group[c].mean()), 1) for c in profile_cols if c in group.columns}
        segments.append({
            "cluster_id": int(cid),
            "label": seg_label,
            "size": len(group),
            "pct": 100 * len(group) / len(labeled),
            "profile": profile_vals,
        })

    return {"n_clusters": n_clusters, "segments": segments}
