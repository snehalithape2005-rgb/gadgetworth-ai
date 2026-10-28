"""Model loading, input checks, prediction, and deal classification."""

import math
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from utils.preprocessing import FEATURES, SUPPORTED_DEVICES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models"


def format_price(value):
    """Format numeric dataset units without assuming a currency."""
    return format(round(float(value)), ",.0f")


def validate_prediction_input(values):
    """Validate and normalize the features expected by the trained pipeline."""
    missing = set(FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing required features: {', '.join(sorted(missing))}")
    result = dict(values)
    for key in ("ram_gb", "storage_gb", "age_years", "battery_health", "original_price"):
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.") from None
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be a finite number.")
    if result["ram_gb"] <= 0 or result["storage_gb"] <= 0:
        raise ValueError("RAM and storage must be positive.")
    if result["age_years"] < 0:
        raise ValueError("Device age cannot be negative.")
    if not 0 <= result["battery_health"] <= 100:
        raise ValueError("Battery health must be between 0 and 100%.")
    if result["original_price"] <= 0:
        raise ValueError("Original price must be positive.")
    for key in ("brand", "condition"):
        if pd.isna(result[key]) or not str(result[key]).strip():
            raise ValueError(f"{key.capitalize()} is required.")
        result[key] = str(result[key]).strip()
    return result


def load_model(device, model_dir=None):
    """Load the saved artifact for a supported device category."""
    if not isinstance(device, str) or device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unknown device category: {device!r}. Choose Laptop or Smartphone.")
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    path = directory / f"{device.lower()}_model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found for {device}: {path}. Run `python train_model.py`.")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Could not read the saved {device} model: {exc}") from exc
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Saved artifact for {device} is invalid. Run `python train_model.py`.")
    artifact_version = artifact.get("artifact_versions", {}).get("scikit_learn")
    if artifact_version and artifact_version != sklearn.__version__:
        raise ValueError(
            f"The {device} model was trained with scikit-learn {artifact_version}, "
            f"but this environment has {sklearn.__version__}. Activate the project "
            "environment and retrain with `python train_model.py`."
        )
    return artifact


def predict_price(device, values, model_dir=None):
    """Return one finite non-negative model prediction in the dataset's units."""
    artifact = load_model(device, model_dir=model_dir)
    normalized = validate_prediction_input(values)
    frame = pd.DataFrame([{name: normalized[name] for name in artifact.get("features", FEATURES)}])
    prediction = float(artifact["pipeline"].predict(frame)[0])
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("The model returned an invalid price estimate.")
    return prediction


def classify_deal(asking, fair_price):
    asking, fair_price = float(asking), float(fair_price)
    if not math.isfinite(asking) or not math.isfinite(fair_price) or asking <= 0 or fair_price <= 0:
        raise ValueError("Prices must be valid positive amounts.")
    difference_pct = (asking - fair_price) / fair_price * 100
    label = "Great Deal" if difference_pct <= -15 else "Fair Price" if difference_pct <= 10 else "Overpriced"
    return label, asking - fair_price, difference_pct
