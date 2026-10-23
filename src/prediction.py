"""
prediction.py
RandomForest regression model for spending score prediction.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_SEED = 42
DEFAULT_TARGET = "SpendingScore"
MIN_ROWS_FOR_TRAINING = 30


def get_suitable_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Return numeric columns that can serve as features (exclude target and ID-style cols).
    """
    exclude = {target_col, "CustomerID", "ClusterID"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in exclude]
    return features


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = DEFAULT_TARGET,
) -> dict:
    """
    Train a RandomForestRegressor on the provided feature columns.
    Returns a result dict with model, metrics, predictions, feature importance, and any error.
    """
    result = {
        "model": None,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "mae": None,
        "r2": None,
        "y_test": None,
        "y_pred": None,
        "feature_importance": None,
        "error": None,
        "n_train": None,
        "n_test": None,
    }

    # Validate
    if target_col not in df.columns:
        result["error"] = f"Target column '{target_col}' not found in dataset."
        return result

    if not feature_cols:
        result["error"] = "No feature columns available for training."
        return result

    sub = df[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()

    if len(sub) < MIN_ROWS_FOR_TRAINING:
        result["error"] = (
            f"Only {len(sub)} valid rows available. At least {MIN_ROWS_FOR_TRAINING} are required."
        )
        return result

    # Check for constant target
    if sub[target_col].std() == 0:
        result["error"] = f"Target column '{target_col}' has no variance — cannot train."
        return result

    X = sub[feature_cols]
    y = sub[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    result["model"] = model
    result["mae"] = round(mae, 3)
    result["r2"] = round(r2, 4)
    result["y_test"] = y_test.values
    result["y_pred"] = y_pred
    result["feature_importance"] = importance
    result["n_train"] = len(X_train)
    result["n_test"] = len(X_test)

    return result


def predict_single(model, feature_cols: list[str], input_values: dict) -> float | None:
    """
    Run a prediction for a single customer given a dict of feature values.
    Returns the predicted value or None on error.
    """
    if model is None:
        return None
    try:
        row = pd.DataFrame([{c: float(input_values.get(c, 0)) for c in feature_cols}])
        pred = model.predict(row)[0]
        return round(float(pred), 2)
    except Exception:
        return None


def get_model_metrics_summary(result: dict) -> str:
    """
    Return a text summary of model metrics for the AI assistant.
    """
    if result.get("error"):
        return f"Prediction model unavailable: {result['error']}"
    return (
        f"Prediction target: {result['target_col']}. "
        f"Features used: {', '.join(result['feature_cols'])}. "
        f"Training rows: {result['n_train']}, Test rows: {result['n_test']}. "
        f"MAE: {result['mae']}, R²: {result['r2']}."
    )
