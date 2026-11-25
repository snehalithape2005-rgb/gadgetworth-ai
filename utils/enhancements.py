from __future__ import annotations

from datetime import datetime
from html import escape
from io import StringIO
import math
from typing import Any, Mapping

import pandas as pd

HISTORY_KEY = "prediction_history"


def initialize_history(state: Any) -> None:
    """Initialize history once in a mapping such as Streamlit session state."""
    if HISTORY_KEY not in state:
        state[HISTORY_KEY] = []


def record_history(state: Any, record: Mapping[str, Any]) -> None:
    initialize_history(state)
    state[HISTORY_KEY].append(dict(record))


def clear_history(state: Any) -> None:
    state[HISTORY_KEY] = []


def history_csv(records: list[Mapping[str, Any]]) -> bytes:
    """Serialize history without an index; return UTF-8 bytes for download."""
    if not records:
        return b""
    buffer = StringIO()
    flattened = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "specifications"}
        row.update(record.get("specifications", {}))
        flattened.append(row)
    pd.DataFrame(flattened).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def compare_prices(price_a: float, price_b: float) -> dict[str, Any]:
    """Calculate absolute and relative price differences safely."""
    a, b = float(price_a), float(price_b)
    if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0:
        raise ValueError("Prices must be finite and non-negative.")
    difference = abs(a - b)
    lower = "Both have the same predicted price" if a == b else ("Gadget A" if a < b else "Gadget B")
    pct = difference / max(a, b) * 100 if max(a, b) else None
    return {"difference": difference, "percentage_difference": pct, "lower_price": lower}


def global_feature_importance(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Aggregate fitted tree importance across encoded columns to original features."""
    pipeline = artifact.get("pipeline")
    if pipeline is None:
        return []
    named_steps = getattr(pipeline, "named_steps", {})
    model = named_steps.get("model")
    raw_importance = getattr(model, "feature_importances_", None)
    preprocessor = named_steps.get("preprocessing")
    if raw_importance is None or preprocessor is None:
        return []
    try:
        names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        return []
    if len(names) != len(raw_importance):
        return []
    grouped: dict[str, float] = {}
    for transformed_name, importance in zip(names, raw_importance):
        name = str(transformed_name).split("__", 1)[-1]
        if name.startswith("brand_"):
            name = "brand"
        elif name.startswith("condition_"):
            name = "condition"
        grouped[name] = grouped.get(name, 0.0) + float(importance)
    labels = {
        "brand": "Brand", "ram_gb": "RAM", "storage_gb": "Storage",
        "age_years": "Device age", "battery_health": "Battery health",
        "condition": "Condition", "original_price": "Original price",
    }
    return sorted(
        ({"Feature": labels.get(name, name), "Importance": score} for name, score in grouped.items()),
        key=lambda item: item["Importance"], reverse=True,
    )


def build_price_report(record: Mapping[str, Any]) -> str:
    """Build a self-contained HTML report for one prediction."""
    specs = record.get("specifications", {})
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in specs.items()
    )
    optional_rows = []
    for key, label in (("asking_price", "Asking price"), ("deal", "Deal classification")):
        if record.get(key) not in (None, ""):
            optional_rows.append(f"<tr><th>{escape(label)}</th><td>{escape(str(record[key]))}</td></tr>")
    if record.get("difference") is not None:
        optional_rows.append(
            f"<tr><th>Absolute price difference</th><td>{escape(str(abs(float(record['difference']))))}</td></tr>"
        )
    if record.get("difference_pct") is not None:
        pct_display = f"{float(record['difference_pct']):+.1f}%"
        optional_rows.append(
            f"<tr><th>Difference percentage</th><td>{escape(pct_display)}</td></tr>"
        )
    optional = "".join(optional_rows)
    generated = escape(str(record.get("timestamp", "")))
    price = escape(str(record.get("predicted_price", "")))
    device = escape(str(record.get("device", "")))
    model = escape(str(record.get("model_name", "")))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>GadgetWorth AI price report</title>
<style>body{{font:16px Arial,sans-serif;max-width:760px;margin:40px auto;color:#202820}}h1{{color:#166534}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #d1d5db;text-align:left}}th{{width:38%;color:#4b5563}}.price{{font-size:28px;font-weight:bold;color:#15803d}}.note{{background:#f3f4f3;padding:16px;border-left:4px solid #15803d}}</style></head>
<body><h1>GadgetWorth AI</h1><p>Price estimate report · generated {generated}</p><h2>{device}</h2><p class="price">{price} dataset units</p>
<table>{rows}{optional}<tr><th>Selected model</th><td>{model}</td></tr></table>
<div class="note">This is a model prediction based on patterns learned from its training data. It is not a guaranteed sale price or a professional valuation. Dataset currency and units may be unverified; use this estimate as a starting point and check the device carefully.</div></body></html>"""


def make_history_record(device: str, specs: Mapping[str, Any], predicted_price: float,
                        asking_price: float | None = None, deal: str | None = None,
                        difference: float | None = None, difference_pct: float | None = None,
                        model_name: str = "", now: datetime | None = None) -> dict[str, Any]:
    timestamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "device": device,
        "specifications": dict(specs),
        "predicted_price": float(predicted_price),
        "asking_price": asking_price,
        "deal": deal,
        "difference": difference,
        "difference_pct": difference_pct,
        "model_name": model_name,
    }
