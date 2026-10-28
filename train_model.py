"""Train one laptop and one smartphone model from the inspected CSV."""

from pathlib import Path
import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.preprocessing import (
    FEATURES,
    REQUIRED_COLUMNS,
    SUPPORTED_DEVICES,
    build_preprocessor,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "used_device_data.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_FRACTION_OF_DEVELOPMENT = 0.25
VERIFIED_SYNTHETIC_DATA_SHA256 = "19EF36F400142BF2F5A80A7BCCAD969BD7A390FC9B349ADE8481338E29852F4E"


def load_training_data(path=DATA_PATH):
    """Read the dataset and remove rows that cannot be safely modeled."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    df = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy().drop_duplicates()
    df["device"] = df["device"].astype("string").str.strip()
    for column in ["ram_gb", "storage_gb", "age_years", "battery_health", "original_price", "price"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["brand"] = df["brand"].astype("string").str.strip()
    df["condition"] = df["condition"].astype("string").str.strip()

    valid = (
        df["device"].isin(SUPPORTED_DEVICES)
        & df["price"].gt(0)
        & df["original_price"].gt(0)
        & df["ram_gb"].gt(0)
        & df["storage_gb"].gt(0)
        & df["age_years"].ge(0)
        & df["battery_health"].between(0, 100)
        & df["brand"].notna() & df["brand"].ne("")
        & df["condition"].notna() & df["condition"].ne("")
    )
    cleaned = df.loc[valid].copy()
    if cleaned.empty:
        raise ValueError("No valid training rows remain after checking values and device categories.")
    return cleaned


def candidate_models():
    return {
        "Linear Regression": (LinearRegression(), True),
        "Random Forest": (RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ), False),
        "Gradient Boosting": (GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ), False),
    }


def train_device_model(part, device, output_dir=MODEL_DIR, dataset_path=DATA_PATH):
    if len(part) < 20:
        raise ValueError(f"At least 20 valid {device} rows are required; found {len(part)}.")
    X = part[FEATURES]
    y = part["price"].astype(float)
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development,
        test_size=VALIDATION_FRACTION_OF_DEVELOPMENT,
        random_state=RANDOM_STATE,
    )

    validation_scores, fitted, validation_rows = {}, {}, []
    for name, (estimator, scale_numeric) in candidate_models().items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        validation_predictions = pipeline.predict(X_validation)
        validation_mae = float(mean_absolute_error(y_validation, validation_predictions))
        validation_rows.append({"Model": name, "Validation MAE": validation_mae})
        validation_scores[name] = validation_mae
        fitted[name] = pipeline

    best_name = min(validation_scores, key=validation_scores.get)
    results = []
    evaluations = {}
    for name, pipeline in fitted.items():
        # Refit on development data after model selection, keeping the test split untouched.
        pipeline.fit(X_development, y_development)
        predictions = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Validation MAE": validation_scores[name],
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": float(r2_score(y_test, predictions)),
        })
        evaluations[name] = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions})
    results = pd.DataFrame(results).sort_values("Validation MAE", ascending=True).reset_index(drop=True)
    best_pipeline = fitted[best_name]
    model = best_pipeline.named_steps["model"]
    importances = []
    if hasattr(model, "feature_importances_"):
        transformed_names = best_pipeline.named_steps["preprocessing"].get_feature_names_out()
        importances = sorted(
            ({"Feature": str(f), "Importance": float(i)} for f, i in zip(transformed_names, model.feature_importances_)),
            key=lambda item: item["Importance"], reverse=True,
        )

    try:
        source_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest().upper()
    except OSError:
        source_hash = ""
    is_known_synthetic_copy = source_hash == VERIFIED_SYNTHETIC_DATA_SHA256
    artifact = {
        "pipeline": best_pipeline,
        "features": FEATURES,
        "device": device,
        "model_name": best_name,
        "metrics": results.to_dict(orient="records"),
        "mae": float(results.iloc[0]["MAE"]),
        "evaluation": evaluations[best_name].to_dict(orient="records"),
        "feature_importance": importances,
        "categories": {
            "brand": sorted(part["brand"].dropna().astype(str).unique().tolist()),
            "condition": sorted(part["condition"].dropna().astype(str).unique().tolist()),
        },
        "value_options": {
            "ram_gb": sorted(part["ram_gb"].astype(float).unique().tolist()),
            "storage_gb": sorted(part["storage_gb"].astype(float).unique().tolist()),
        },
        "numeric_ranges": {
            column: {"min": float(part[column].min()), "max": float(part[column].max()), "median": float(part[column].median())}
            for column in ["age_years", "battery_health", "original_price", "price"]
        },
        "dataset": Path(dataset_path).name,
        "dataset_rows": int(len(part)),
        "synthetic_data": is_known_synthetic_copy,
        "data_provenance": (
            "Exact match for the 1,200-row file previously generated by the project's seeded synthetic-data function; not observed market transactions."
            if is_known_synthetic_copy else
            "Dataset source is not verified. Treat predictions and evaluation as unverified and do not claim market accuracy."
        ),
        "price_unit": "Unverified numeric units; currency and source are undocumented.",
        "random_state": RANDOM_STATE,
        "selection_criterion": "Lowest validation MAE; final metrics use an untouched test set (60/20/20 train/validation/test split, random_state=42).",
        "validation_metrics": sorted(validation_rows, key=lambda row: row["Validation MAE"]),
        "selected_validation_mae": validation_scores[best_name],
        "artifact_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{device.lower()}_model.joblib"
    joblib.dump(artifact, output_path)
    print(f"\n{device} model comparison (selected using validation MAE; test MAE/RMSE lower and R² higher are better):")
    print(results.to_string(index=False))
    print(f"Selected {best_name}; saved {output_path}")
    return artifact


def main():
    df = load_training_data()
    print(f"Training data: {DATA_PATH.name}; {len(df)} valid rows; columns: {df.columns.tolist()}")
    print("Device counts:", df["device"].value_counts().to_dict())
    print("Provenance:", "exact match to prior seeded synthetic generator" if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest().upper() == VERIFIED_SYNTHETIC_DATA_SHA256 else "source unverified")
    print("Currency: unverified; displayed price units remain unspecified.")
    for device in SUPPORTED_DEVICES:
        part = df[df["device"] == device].copy()
        train_device_model(part, device)


if __name__ == "__main__":
    main()
