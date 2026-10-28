import numpy as np
import pandas as pd
import pytest
import sklearn
from pathlib import Path

from utils.prediction import (
    classify_deal,
    format_price,
    load_model,
    predict_price,
    validate_prediction_input,
)


def sample_input():
    return {
        "brand": "Dell", "ram_gb": 8, "storage_gb": 256,
        "age_years": 2, "battery_health": 85, "condition": "Good",
        "original_price": 50000,
    }


@pytest.fixture
def trained_models():
    missing = [d for d in ("Laptop", "Smartphone") if not (Path(__file__).resolve().parents[1] / "models" / f"{d.lower()}_model.joblib").is_file()]
    if missing:
        pytest.skip("Run `python train_model.py` to create integration-test model artifacts.")
    return Path(__file__).resolve().parents[1] / "models"


def test_deal_classification_boundaries():
    assert classify_deal(8500, 10000)[0] == "Great Deal"
    assert classify_deal(8501, 10000)[0] == "Fair Price"
    assert classify_deal(11000, 10000)[0] == "Fair Price"
    assert classify_deal(11001, 10000)[0] == "Overpriced"


def test_deal_difference_values():
    label, difference, pct = classify_deal(8000, 10000)
    assert label == "Great Deal"
    assert difference == -2000
    assert pct == pytest.approx(-20)


@pytest.mark.parametrize("asking,fair", [(-1, 10), (0, 10), (1, 0), (float("inf"), 10)])
def test_deal_rejects_invalid_prices(asking, fair):
    with pytest.raises(ValueError):
        classify_deal(asking, fair)


def test_prediction_validation_normalizes_numbers():
    values = sample_input()
    values["ram_gb"] = "8"
    assert validate_prediction_input(values)["ram_gb"] == 8.0


@pytest.mark.parametrize("key,value", [
    ("storage_gb", 0), ("ram_gb", -1), ("age_years", -0.5),
    ("battery_health", 101), ("original_price", -1),
])
def test_prediction_rejects_invalid_values(key, value):
    values = sample_input()
    values[key] = value
    with pytest.raises(ValueError):
        validate_prediction_input(values)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device category"):
        load_model("Tablet", model_dir="missing")


@pytest.mark.parametrize("device", ["Laptop", "Smartphone"])
def test_device_models_load_and_predict_in_dataset_units(device, trained_models):
    artifact = load_model(device, model_dir=trained_models)
    prediction = predict_price(device, sample_input(), model_dir=trained_models)
    assert artifact["device"] == device
    assert artifact["artifact_versions"]["scikit_learn"] == sklearn.__version__
    assert np.isfinite(prediction) and prediction > 0


def test_missing_model_has_helpful_error():
    missing_dir = Path(__file__).resolve().parent / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="Run `python train_model.py`"):
        load_model("Laptop", model_dir=missing_dir)


def test_model_from_different_sklearn_version_is_rejected(monkeypatch, trained_models):
    monkeypatch.setattr(
        "utils.prediction.joblib.load",
        lambda _path: {"pipeline": object(), "artifact_versions": {"scikit_learn": "0.0"}},
    )
    with pytest.raises(ValueError, match="trained with scikit-learn"):
        load_model("Laptop", model_dir=trained_models)


def test_inference_uses_saved_training_preprocessing(trained_models):
    artifact = load_model("Laptop", model_dir=trained_models)
    values = validate_prediction_input(sample_input())
    expected = float(artifact["pipeline"].predict(pd.DataFrame([values]))[0])
    actual = predict_price("Laptop", values, model_dir=trained_models)
    assert actual == pytest.approx(expected)
    assert artifact["pipeline"].feature_names_in_.tolist() == artifact["features"]


def test_price_format_does_not_assume_currency():
    assert format_price(25000) == "25,000"
