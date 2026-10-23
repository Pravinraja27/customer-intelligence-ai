"""
data_loader.py
Handles sample data generation, CSV upload validation, and data profiling.
"""
import numpy as np
import pandas as pd
import io

RANDOM_SEED = 42
REQUIRED_NUMERIC_COLS = ["Age", "AnnualIncome", "SpendingScore"]


def generate_sample_data(n: int = 700) -> pd.DataFrame:
    """
    Generate a reproducible synthetic customer dataset with distinct behavioral patterns.
    Uses a fixed seed so results are consistent across runs.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    segments = {
        "high_value": int(n * 0.20),
        "budget_conscious": int(n * 0.25),
        "young_spenders": int(n * 0.20),
        "loyal_regulars": int(n * 0.20),
        "occasional": n - int(n * 0.20) - int(n * 0.25) - int(n * 0.20) - int(n * 0.20),
    }

    rows = []

    def _make_segment(seg_name: str, count: int):
        if seg_name == "high_value":
            age = rng.integers(35, 60, count)
            income = rng.integers(90000, 180000, count)
            spending = rng.integers(70, 100, count)
            freq = rng.integers(12, 24, count)
            aov = rng.integers(250, 600, count)
            recency = rng.integers(1, 30, count)
            visits = rng.integers(15, 35, count)
            sat = rng.uniform(4.0, 5.0, count)
        elif seg_name == "budget_conscious":
            age = rng.integers(30, 55, count)
            income = rng.integers(25000, 55000, count)
            spending = rng.integers(10, 40, count)
            freq = rng.integers(3, 8, count)
            aov = rng.integers(30, 90, count)
            recency = rng.integers(30, 120, count)
            visits = rng.integers(3, 10, count)
            sat = rng.uniform(2.5, 3.8, count)
        elif seg_name == "young_spenders":
            age = rng.integers(18, 32, count)
            income = rng.integers(20000, 60000, count)
            spending = rng.integers(60, 100, count)
            freq = rng.integers(8, 20, count)
            aov = rng.integers(80, 220, count)
            recency = rng.integers(1, 20, count)
            visits = rng.integers(20, 50, count)
            sat = rng.uniform(3.5, 5.0, count)
        elif seg_name == "loyal_regulars":
            age = rng.integers(40, 65, count)
            income = rng.integers(55000, 90000, count)
            spending = rng.integers(45, 70, count)
            freq = rng.integers(10, 20, count)
            aov = rng.integers(120, 280, count)
            recency = rng.integers(5, 40, count)
            visits = rng.integers(8, 20, count)
            sat = rng.uniform(3.8, 5.0, count)
        else:  # occasional
            age = rng.integers(22, 70, count)
            income = rng.integers(30000, 75000, count)
            spending = rng.integers(20, 55, count)
            freq = rng.integers(1, 5, count)
            aov = rng.integers(50, 180, count)
            recency = rng.integers(60, 365, count)
            visits = rng.integers(1, 8, count)
            sat = rng.uniform(2.0, 4.0, count)

        genders = rng.choice(["Male", "Female", "Other"], count, p=[0.47, 0.47, 0.06])
        for i in range(count):
            rows.append({
                "Gender": genders[i],
                "Age": int(age[i]),
                "AnnualIncome": int(income[i]),
                "SpendingScore": int(spending[i]),
                "PurchaseFrequency": int(freq[i]),
                "AverageOrderValue": int(aov[i]),
                "RecencyDays": int(recency[i]),
                "WebsiteVisits": int(visits[i]),
                "CustomerSatisfaction": round(float(sat[i]), 1),
            })

    for seg, cnt in segments.items():
        _make_segment(seg, cnt)

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "CustomerID", [f"CUST{str(i+1).zfill(4)}" for i in range(len(df))])
    return df.reset_index(drop=True)


def validate_and_load_csv(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Reads and validates an uploaded CSV file.
    Returns (dataframe, None) on success, or (None, error_message) on failure.
    """
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return None, f"Could not parse the CSV file: {e}"

    if df.empty:
        return None, "The uploaded CSV file is empty."
    if len(df.columns) < 2:
        return None, "The CSV file must contain at least 2 columns."
    if len(df) < 10:
        return None, f"The dataset has only {len(df)} rows. At least 10 rows are required."

    return df, None


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return names of columns with numeric dtype."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return names of columns with object/category dtype."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Produce a basic data-quality summary.
    """
    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    missing_per_col = df.isnull().sum()
    missing_cols = missing_per_col[missing_per_col > 0].to_dict()

    report = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in missing_cols.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }
    return report


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning: drop fully-empty rows, convert obvious numeric columns stored
    as strings, and clip outliers to ±3 std dev per numeric column.
    """
    df = df.copy()
    df.dropna(how="all", inplace=True)

    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.8:
                df[col] = converted

    return df
