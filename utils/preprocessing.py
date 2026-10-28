"""Shared, train-only preprocessing for GadgetWorth regression pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "brand", "ram_gb", "storage_gb", "age_years",
    "battery_health", "condition", "original_price",
]
CATEGORICAL_FEATURES = ["brand", "condition"]
NUMERIC_FEATURES = [
    "ram_gb", "storage_gb", "age_years", "battery_health", "original_price",
]
REQUIRED_COLUMNS = ["device", *FEATURES, "price"]
SUPPORTED_DEVICES = ("Laptop", "Smartphone")


def build_preprocessor(scale_numeric=False):
    """Create a fresh preprocessor; fit it only inside the training pipeline."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
