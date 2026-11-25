from datetime import datetime, timezone
from pathlib import Path

import pytest

from utils.enhancements import (
    HISTORY_KEY,
    build_price_report,
    clear_history,
    compare_prices,
    global_feature_importance,
    history_csv,
    initialize_history,
    make_history_record,
    record_history,
)


def test_compare_prices_equal_and_zero_cases():
    assert compare_prices(100, 100) == {
        "difference": 0.0, "percentage_difference": 0.0,
        "lower_price": "Both have the same predicted price",
    }
    zero = compare_prices(0, 0)
    assert zero["percentage_difference"] is None
    assert zero["lower_price"] == "Both have the same predicted price"
    result = compare_prices(75, 100)
    assert result["difference"] == 25
    assert result["percentage_difference"] == 25
    assert result["lower_price"] == "Gadget A"


@pytest.mark.parametrize("a,b", [(-1, 2), (float("nan"), 3), (1, float("inf"))])
def test_compare_prices_rejects_invalid_numbers(a, b):
    with pytest.raises(ValueError):
        compare_prices(a, b)


def test_history_initialization_record_clear_and_csv():
    state = {}
    initialize_history(state)
    initialize_history(state)
    assert state[HISTORY_KEY] == []
    record_history(state, {"device": "Laptop", "predicted_price": 123})
    assert len(state[HISTORY_KEY]) == 1
    csv = history_csv(state[HISTORY_KEY]).decode("utf-8-sig")
    assert "Laptop" in csv and "predicted_price" in csv
    assert history_csv([]) == b""
    clear_history(state)
    assert state[HISTORY_KEY] == []


def test_report_contains_prediction_and_safe_optional_fields():
    record = make_history_record(
        "Laptop", {"Brand": "A&B", "Condition": "Good"}, 25000,
        model_name="Gradient Boosting", now=datetime(2026, 10, 29, tzinfo=timezone.utc),
    )
    report = build_price_report(record)
    assert "GadgetWorth AI" in report
    assert "25,000" not in report  # report reflects the numeric value passed in, without inventing formatting
    assert "25000.0" in report
    assert "A&amp;B" in report
    assert "not a guaranteed sale price" in report
    assert "Asking price" not in report


def test_report_includes_asking_and_absolute_comparison():
    record = make_history_record(
        "Smartphone", {"Brand": "Samsung"}, 10000, asking_price=8000,
        deal="Great Deal", difference=-2000, difference_pct=-20,
    )
    report = build_price_report(record)
    assert "Asking price" in report
    assert "Absolute price difference</th><td>2000.0" in report
    assert "-20.0%" in report
    assert "Great Deal" in report


def test_feature_importance_returns_empty_if_unsupported():
    assert global_feature_importance({}) == []
    assert global_feature_importance({"pipeline": object()}) == []


@pytest.mark.parametrize("device", ["Laptop", "Smartphone"])
def test_saved_model_importance_is_mapped_when_supported(device):
    from utils.prediction import load_model
    model_dir = Path(__file__).resolve().parents[1] / "models"
    if not (model_dir / f"{device.lower()}_model.joblib").is_file():
        pytest.skip("Saved project model artifact is unavailable")
    artifact = load_model(device, model_dir=model_dir)
    importance = global_feature_importance(artifact)
    if hasattr(artifact["pipeline"].named_steps["model"], "feature_importances_"):
        features = {row["Feature"] for row in importance}
        assert {"Brand", "Condition", "RAM", "Storage", "Device age", "Battery health", "Original price"} <= features
        assert sum(row["Importance"] for row in importance) == pytest.approx(1.0)
