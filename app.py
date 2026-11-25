from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.prediction import classify_deal, format_price, load_model, predict_price
from utils.enhancements import (
    build_price_report, clear_history, compare_prices, global_feature_importance,
    history_csv, initialize_history, make_history_record, record_history,
)
from utils.preprocessing import SUPPORTED_DEVICES

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="GadgetWorth AI", page_icon="💻", layout="wide")
st.markdown("""
<style>
:root {
  color-scheme: dark;
  --heading-h1: #F0FDF4;
  --heading-h2: #BBF7D0;
  --heading-h3: #86A995;
  --card-title: var(--heading-h3);
  --text-body: #C1D5C8;
  --text-muted: #86A995;
  --page-background: #000000;
  --sidebar-background: #000000;
  --card-background: #10271D;
  --surface-secondary: #143A2A;
  --card-border: #244936;
  --notice-background: #112C20;
  --accent-highlight: #86EFAC;
  --font-sans: Inter, "Segoe UI", Arial, sans-serif;
  --font-page-title: clamp(2rem, 3vw, 2.5rem);
  --font-section-title: clamp(1.375rem, 2.1vw, 1.75rem);
  --font-subsection-title: clamp(1.0625rem, 1.55vw, 1.3125rem);
  --font-card-title: 18px;
  --font-body: 15px;
  --font-caption: 13px;
}
html, body, [data-testid="stAppViewContainer"], button, input, textarea { font-family: var(--font-sans); }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--page-background) !important; color: var(--heading-h1); }
h1, h2, h3, h4 { letter-spacing: -0.035em; }
h1 { color: var(--heading-h1) !important; font-size: var(--font-page-title) !important; font-weight: 740 !important; line-height: 1.12; }
h2 { color: var(--heading-h2) !important; font-size: var(--font-section-title) !important; font-weight: 650 !important; line-height: 1.25; }
h3, h4 { color: var(--heading-h3) !important; font-size: var(--font-subsection-title) !important; font-weight: 650 !important; line-height: 1.3; }
.card-title { display: block; color: var(--card-title) !important; font-size: var(--font-card-title); font-weight: 650; letter-spacing: -.02em; line-height: 1.3; margin: 0 0 12px; }
p, li { color: var(--text-body); font-size: var(--font-body); line-height: 1.6; }
[data-testid="stWidgetLabel"] { color: var(--text-body) !important; }
[data-testid="stMain"] label { color: var(--text-body); }
[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: var(--sidebar-background) !important; border-right: 1px solid var(--card-border); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: var(--text-body); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--text-muted); }
[data-testid="stSidebar"] hr { border-color: var(--card-border); }
[data-testid="stCaptionContainer"] { color: var(--text-muted); font-size: var(--font-caption); }
[data-testid="stAlert"] { background: var(--notice-background); border: 1px solid var(--card-border); border-radius: 12px; color: var(--text-body); }
[data-testid="stAlert"] svg { color: var(--accent-highlight); }
[data-testid="stMetric"] { background: var(--card-background); border: 1px solid var(--card-border); border-radius: 16px; padding: 17px 18px; box-shadow: 0 12px 28px rgba(0, 0, 0, .16); }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--accent-highlight) !important; font-size: 1.65rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: var(--accent-highlight) !important; }
input, textarea, [data-baseweb="select"] > div { color: var(--heading-h1) !important; background-color: var(--surface-secondary) !important; border-color: var(--card-border) !important; }
input::placeholder, textarea::placeholder { color: var(--text-muted) !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background: var(--surface-secondary) !important; color: var(--heading-h1) !important; }
[role="option"], [role="option"] span { color: var(--text-body) !important; }
[data-testid="stForm"] { background: var(--card-background); border: 1px solid var(--card-border); border-radius: 20px; padding: 24px 26px; box-shadow: 0 16px 40px rgba(0, 0, 0, .17); }
[data-testid="stVerticalBlockBorderWrapper"] > div { background-color: var(--card-background); border-color: var(--card-border) !important; border-radius: 18px !important; box-shadow: 0 12px 30px rgba(0, 0, 0, .12); }
.stButton > button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button { min-height: 2.75rem; border-radius: 11px; font-weight: 650; transition: transform .15s ease, background-color .15s ease, border-color .15s ease, box-shadow .15s ease; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover { transform: translateY(-1px); }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible { outline: 2px solid var(--accent-highlight); outline-offset: 2px; }
[data-testid="stFormSubmitButton"] button[kind="primary"], .stButton > button[kind="primary"] { background: #166534; border: 1px solid #166534; color: var(--heading-h1); }
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover { background: #15803D; border-color: #15803D; }
[data-testid="stDownloadButton"] button, .stButton > button[kind="secondary"] { background: var(--surface-secondary); border: 1px solid var(--card-border); color: var(--text-body); }
[data-testid="stDownloadButton"] button[kind="primary"] { background: #166534; border: 1px solid #166534; color: var(--heading-h1); }
[data-testid="stDownloadButton"] button[kind="primary"]:hover { background: #15803D; border-color: #15803D; }
[data-testid="stSidebar"] .stButton > button { width: 100%; min-width: 0; min-height: 46px; justify-content: flex-start; text-align: left; white-space: nowrap; overflow: hidden; background: transparent; border: 1px solid transparent; border-radius: 10px; color: var(--text-muted); font-size: 17px; font-weight: 650; line-height: 1.25; padding: 10px 12px 10px 14px; margin-bottom: 4px; }
[data-testid="stSidebar"] .stButton > button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 17px; font-weight: 650; line-height: 1.25; }
[data-testid="stSidebar"] .stButton > button:hover { background: var(--card-background); border-color: var(--card-border); color: var(--heading-h1); }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #166534; border-color: #166534; color: var(--heading-h1); }
.brand-lockup { display: flex; align-items: center; gap: 12px; padding: 10px 2px 18px; }
.brand-mark { display: grid; place-items: center; width: 46px; height: 46px; flex: 0 0 46px; border: 1px solid var(--card-border); border-radius: 15px; color: var(--accent-highlight); background: var(--surface-secondary); font-weight: 800; letter-spacing: -.05em; }
.brand-copy { min-width: 0; }
.brand-copy strong { display: block; color: #F0FDF4; font-size: 27px; font-weight: 780; line-height: 1.08; letter-spacing: -.045em; white-space: nowrap; }
.brand-copy small { display: block; margin-top: 6px; color: var(--text-muted); font-size: 10px; line-height: 1.2; letter-spacing: .09em; white-space: nowrap; }
.hero { position: relative; isolation: isolate; overflow: hidden; background: radial-gradient(ellipse at 92% 0%, rgba(134, 239, 172, .08), transparent 38%), linear-gradient(120deg, #10271D 0%, #10271D 62%, #143A2A 100%); color: var(--heading-h1); padding: clamp(30px, 5vw, 54px) clamp(24px, 5vw, 56px); border: 1px solid var(--card-border); border-radius: 24px; margin: 8px 0 16px; }
.hero-kicker { display: inline-block; color: var(--accent-highlight); text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; margin-bottom: 8px; }
.hero h1 { color: var(--heading-h1) !important; font-size: clamp(2.5rem, 5vw, 3.75rem) !important; font-weight: 780 !important; max-width: 800px; margin: .2rem 0 .9rem; line-height: 1.04; }
.hero p { color: var(--text-body) !important; max-width: 660px; margin-bottom: 0; font-size: 1.06rem; }
.eyebrow { color: var(--text-muted); text-transform: uppercase; letter-spacing: .16em; font-weight: 700; font-size: .72rem; }
.stat-card { display: flex; flex-direction: column; justify-content: space-between; background: var(--card-background); border: 1px solid var(--card-border); border-radius: 16px; padding: 18px 20px; min-height: 122px; }
.stat-label { display: block; color: var(--text-muted); font-size: .82rem; font-weight: 600; letter-spacing: .02em; }
.stat-value { display: block; color: var(--accent-highlight); font-size: clamp(1.45rem, 2vw, 1.8rem); font-weight: 750; margin-top: 16px; letter-spacing: -.025em; }
.step-card { min-height: 164px; padding: 20px; background: var(--card-background); border: 1px solid var(--card-border); border-radius: 16px; }
.step-number { display: inline-block; color: var(--accent-highlight); font-size: .76rem; font-weight: 750; letter-spacing: .1em; margin-bottom: 12px; }
.step-card h3 { margin: 0 0 8px; color: var(--heading-h2) !important; font-size: 1.12rem !important; }
.step-card p { color: var(--text-body); margin: 0; font-size: .9rem; }
[data-testid="stDataFrame"] { border: 1px solid var(--card-border); border-radius: 14px; overflow: hidden; }
.page-lead { color: var(--text-muted); font-size: 1rem; max-width: 820px; margin-top: -.45rem; margin-bottom: 1.35rem; }
.deal-result, .deal-result.fair, .deal-result.overpriced { border: 1px solid var(--card-border); background: var(--notice-background); color: var(--text-body); border-radius: 14px; padding: 16px 18px; margin: 8px 0 14px; }
.deal-result strong { display: block; color: var(--heading-h1); font-size: 1.08rem; margin-bottom: 4px; }
.price-result { padding: 26px 30px; border: 1px solid var(--card-border); border-radius: 20px; background: var(--card-background); box-shadow: 0 18px 42px rgba(0, 0, 0, .19); }
.price-result-label { color: var(--card-title); text-transform: uppercase; letter-spacing: .1em; font-size: 17px; font-weight: 650; }
.price-result-value { display: block; color: var(--accent-highlight); font-size: clamp(2.5rem, 5vw, 3.8rem); letter-spacing: -.055em; font-weight: 760; line-height: 1.1; margin: 7px 0 12px; }
.price-result-device { color: var(--text-muted); font-size: .92rem; }
.spec-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.spec-badge { padding: 7px 10px; border: 1px solid var(--card-border); border-radius: 999px; background: var(--card-background); color: var(--text-body); font-size: .78rem; }
.deal-stat { padding: 15px 16px; border: 1px solid var(--card-border); border-radius: 15px; background: var(--card-background); }
.deal-stat span { display: block; color: var(--text-muted); font-size: .76rem; }
.deal-stat strong { display: block; color: var(--accent-highlight); font-size: clamp(1.15rem, 2vw, 1.55rem); margin-top: 6px; letter-spacing: -.025em; }
.compare-product { padding: 20px; border: 1px solid var(--card-border); border-radius: 18px; background: var(--card-background); box-shadow: 0 14px 32px rgba(0, 0, 0, .15); }
.compare-product.violet { background: var(--card-background); }
.compare-product span { display: block; color: var(--text-muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .1em; }
.compare-product .card-title { margin-bottom: 10px; }
.compare-product strong { display: block; color: var(--accent-highlight); font-size: clamp(1.8rem, 3vw, 2.5rem); margin: 8px 0 4px; letter-spacing: -.04em; }
.compare-product small { color: var(--text-muted); }
.compare-delta { margin: 18px 0; padding: 16px 20px; text-align: center; border: 1px solid var(--card-border); border-radius: 16px; background: var(--card-background); color: var(--text-body); }
.compare-delta strong { color: var(--accent-highlight); font-size: 1.18rem; }
.about-tech { height: 100%; min-height: 90px; padding: 14px; border: 1px solid var(--card-border); border-radius: 14px; background: var(--card-background); box-shadow: 0 8px 22px rgba(0, 0, 0, .13); }
.about-tech strong { display: block; color: var(--card-title); margin-bottom: 5px; }
.about-tech span { color: var(--text-muted); font-size: 13px; }
@media (max-width: 700px) { .hero { padding: 27px 24px; } [data-testid="stForm"] { padding: 18px; } .brand-copy strong { font-size: 25px; } }
</style>
""", unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#10271D", plot_bgcolor="#10271D",
        font={"color": "#C1D5C8"}, title_font={"color": "#F0FDF4"},
        legend={"font": {"color": "#C1D5C8"}},
        colorway=["#86EFAC", "#BBF7D0", "#86A995", "#4D8A68"],
    )
    fig.update_xaxes(title_font={"color": "#86A995"}, tickfont={"color": "#C1D5C8"}, gridcolor="#244936", zerolinecolor="#244936")
    fig.update_yaxes(title_font={"color": "#86A995"}, tickfont={"color": "#C1D5C8"}, gridcolor="#244936", zerolinecolor="#244936")
    return fig


@st.cache_resource
def load_artifact(device):
    try:
        return load_model(device, model_dir=MODEL_DIR)
    except (FileNotFoundError, ValueError):
        return None


def show_missing(device):
    st.error(f"The {device.lower()} model is not available yet. Train it with `python train_model.py`.")
    st.info("Train models from `data/used_device_data.csv` with `python train_model.py`.")


def go_to_prediction():
    st.session_state["nav"] = "Price Prediction"


def card_title(title):
    """Render a consistent white title within a dark surface card."""
    st.markdown(f'<h3 class="card-title">{escape(str(title))}</h3>', unsafe_allow_html=True)


def collect_device_specs(device, key_prefix, artifact):
    """Render the supported model inputs and return a validated input mapping."""
    options = artifact.get("categories", {})
    brands, conditions = options.get("brand", []), options.get("condition", [])
    values = artifact.get("value_options", {})
    ram_options, storage_options = values.get("ram_gb", []), values.get("storage_gb", [])
    if not brands or not conditions or not ram_options or not storage_options:
        st.error(f"The {device.lower()} model is missing supported input categories. Retrain the models.")
        return None
    ranges = artifact.get("numeric_ranges", {})
    st.markdown("## Device Information")
    brand = st.selectbox("Brand", brands, key=f"{key_prefix}_brand", help="Choose the closest supported brand.")
    st.markdown("### Technical Specifications")
    col_a, col_b = st.columns(2)
    with col_a:
        ram = st.selectbox("RAM (GB)", ram_options, key=f"{key_prefix}_ram", help="More RAM can affect the resale value, depending on the model and training data.")
    with col_b:
        storage = st.selectbox("Storage (GB)", storage_options, key=f"{key_prefix}_storage", help="Choose the device's installed storage capacity.")
    st.markdown("## Condition and Age")
    col_a, col_b = st.columns(2)
    with col_a:
        condition = st.selectbox("Physical condition", conditions, key=f"{key_prefix}_condition", help="Select the closest supported condition category.")
        age_bounds = ranges.get("age_years", {"min": 0, "max": 20, "median": 2})
        age = st.number_input("Device age (years)", min_value=float(age_bounds["min"]), max_value=float(age_bounds["max"]), value=float(age_bounds["median"]), step=.5, key=f"{key_prefix}_age", help="Enter age in years; half-year increments are supported.")
    with col_b:
        battery_bounds = ranges.get("battery_health", {"min": 0, "max": 100, "median": 85})
        battery = st.slider("Battery health (%)", int(np.ceil(battery_bounds["min"])), int(np.floor(battery_bounds["max"])), int(round(battery_bounds["median"])), key=f"{key_prefix}_battery", help="Use the reported battery health where available.")
        original_bounds = ranges.get("original_price", {"min": 1, "max": 10000000, "median": 60000})
        original = st.number_input("Original price (dataset units; currency unverified)", min_value=int(np.ceil(original_bounds["min"])), max_value=int(np.floor(original_bounds["max"])), value=int(round(original_bounds["median"])), step=1000, key=f"{key_prefix}_original")
    return {"brand": brand, "ram_gb": ram, "storage_gb": storage, "age_years": age,
            "battery_health": battery, "condition": condition, "original_price": original}


def readable_specs(specs):
    return {
        "Brand": specs["brand"], "RAM": f"{specs['ram_gb']} GB",
        "Storage": f"{specs['storage_gb']} GB", "Age": f"{specs['age_years']:g} years",
        "Battery health": f"{specs['battery_health']}%", "Condition": specs["condition"],
        "Original price": f"{format_price(specs['original_price'])} dataset units",
    }


def show_prediction_result(record):
    badges = "".join(
        f'<span class="spec-badge">{escape(str(label))}: {escape(str(value))}</span>'
        for label, value in record["specifications"].items()
    )
    st.markdown(
        f'<section class="price-result"><div class="price-result-label">Estimated resale value</div>'
        f'<strong class="price-result-value">{escape(format_price(record["predicted_price"]))}</strong>'
        f'<div class="price-result-device">{escape(record["device"])} · dataset units, currency unverified</div>'
        f'<div class="spec-badges">{badges}</div></section>',
        unsafe_allow_html=True,
    )
    if record.get("deal"):
        with st.container(border=True):
            card_title("Deal analysis")
            label = record["deal"]
            if label == "Great Deal":
                explanation, style = "At least 15% below the estimated fair value.", "great"
            elif label == "Fair Price":
                explanation, style = "Between 15% below and 10% above estimated fair value.", "fair"
            else:
                explanation, style = "More than 10% above the estimated fair value.", "overpriced"
            st.markdown(f'<div class="deal-result {style}"><strong>{escape(label)}</strong>{escape(explanation)}</div>', unsafe_allow_html=True)
            x, y, z = st.columns(3)
            for col, metric_label, value in [
                (x, "Seller asking price", format_price(record["asking_price"])),
                (y, "Estimated fair price", format_price(record["predicted_price"])),
                (z, "Difference", f"{format_price(record['difference'])} · {record['difference_pct']:+.1f}%"),
            ]:
                with col:
                    st.markdown(f'<div class="deal-stat"><span>{escape(metric_label)} · dataset units</span><strong>{escape(value)}</strong></div>', unsafe_allow_html=True)
        st.warning("A low price does not guarantee a good purchase. Verify authenticity, repairs, warranty and battery condition.")
    with st.expander("Explain this prediction", expanded=True):
        st.write(f"The {record['model_name']} model estimates this price from patterns learned from the training data, using the specifications shown above.")
        artifact = load_artifact(record["device"])
        importance = global_feature_importance(artifact) if artifact else []
        if importance:
            st.caption("These are global feature importances for the selected model, aggregated across its training data. They describe how much the model relied on each input overall; they do not explain the cause of this individual estimate and do not prove causation.")
            st.dataframe(pd.DataFrame(importance).assign(Importance=lambda df: (df["Importance"] * 100).round(1).astype(str) + "%"), hide_index=True, width="stretch")
        else:
            st.caption("This selected estimator does not expose a supported global feature-importance measure. No per-prediction feature contributions are available.")
    st.caption("Point estimate only; no defensible prediction interval is available. Currency and price units are unverified.")
    st.download_button("Download price report (HTML)", data=build_price_report(record), file_name=f"gadgetworth_{record['device'].lower()}_price_report.html", mime="text/html", key=f"report_{record['timestamp']}")

def prediction_form(device, analyzer=False):
    artifact = load_artifact(device)
    if artifact is None:
        show_missing(device)
        return
    if artifact.get("synthetic_data"):
        st.warning("This dataset exactly matches the project's previous seeded synthetic-data generator. Estimates are demonstration values, not observed market prices. Currency is unverified.")
    if analyzer:
        st.markdown("## Device details and asking price")
    else:
        st.markdown("### Enter the details used by the selected model")
    with st.form("device_form"):
        specs = collect_device_specs(device, f"{device.lower()}_{'deal' if analyzer else 'estimate'}", artifact)
        asking = None
        if analyzer:
            st.markdown("### Seller Asking Price")
            price_bounds = artifact.get("numeric_ranges", {}).get("price", {"min": 1, "max": 10000000, "median": 30000})
            asking = st.number_input("Seller asking price (same dataset units)", min_value=max(1, int(np.ceil(price_bounds["min"]))), max_value=max(1, int(np.floor(price_bounds["max"]))), value=max(1, int(round(price_bounds["median"]))), step=500, key=f"{device.lower()}_asking")
        submitted = st.form_submit_button("Analyze Deal" if analyzer else "Predict Price", type="primary", key="analyze_deal_submit" if analyzer else "estimate_price_submit", width="stretch")
    result_key = f"latest_{device.lower()}_{'deal' if analyzer else 'estimate'}"
    if submitted and specs is not None:
        try:
            with st.spinner("Estimating resale value..."):
                fair = predict_price(device, specs, model_dir=MODEL_DIR)
            label, diff, pct = classify_deal(asking, fair) if analyzer else (None, None, None)
            record = make_history_record(device, readable_specs(specs), fair, asking, label, diff, pct, artifact["model_name"])
            record_history(st.session_state, record)
            st.session_state[result_key] = record
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The estimate could not be calculated. Check that the model artifact matches this app, then retrain it if needed.")
    record = st.session_state.get(result_key)
    if record:
        show_prediction_result(record)


def compare_gadgets():
    st.title("Compare Two Gadgets")
    st.caption("Estimate two devices independently with the existing laptop and smartphone models.")
    with st.form("compare_gadgets_form"):
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                card_title("Gadget A")
                device_a = st.selectbox("Device type", ["Laptop", "Smartphone"], key="compare_device_a")
                artifact_a = load_artifact(device_a)
                specs_a = collect_device_specs(device_a, "compare_a", artifact_a) if artifact_a else None
        with right:
            with st.container(border=True):
                card_title("Gadget B")
                device_b = st.selectbox("Device type", ["Laptop", "Smartphone"], key="compare_device_b")
                artifact_b = load_artifact(device_b)
                specs_b = collect_device_specs(device_b, "compare_b", artifact_b) if artifact_b else None
        submitted = st.form_submit_button("Compare Gadgets", type="primary", width="stretch")
    if artifact_a is None: show_missing(device_a)
    if artifact_b is None: show_missing(device_b)
    if submitted and artifact_a and artifact_b and specs_a is not None and specs_b is not None:
        try:
            price_a = predict_price(device_a, specs_a, model_dir=MODEL_DIR)
            price_b = predict_price(device_b, specs_b, model_dir=MODEL_DIR)
            comparison = compare_prices(price_a, price_b)
            st.session_state["latest_comparison"] = {"device_a": device_a, "specs_a": readable_specs(specs_a), "price_a": price_a,
                "model_a": artifact_a["model_name"], "device_b": device_b, "specs_b": readable_specs(specs_b), "price_b": price_b,
                "model_b": artifact_b["model_name"], **comparison}
            initialize_history(st.session_state)
            record_history(st.session_state, make_history_record(device_a, readable_specs(specs_a), price_a, model_name=artifact_a["model_name"]))
            record_history(st.session_state, make_history_record(device_b, readable_specs(specs_b), price_b, model_name=artifact_b["model_name"]))
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The comparison could not be calculated. Check the saved model artifacts and try again.")
    result = st.session_state.get("latest_comparison")
    if result:
        st.header("Estimated resale values")
        col_a, col_b = st.columns(2)
        for col, label, device, price, variant in [
            (col_a, "Gadget A", result["device_a"], result["price_a"], ""),
            (col_b, "Gadget B", result["device_b"], result["price_b"], " violet"),
        ]:
            with col:
                st.markdown(
                    f'<div class="compare-product{variant}"><h3 class="card-title">{label}</h3>'
                    f'<strong>{escape(format_price(price))}</strong><small>{escape(device)} · predicted price in dataset units</small></div>',
                    unsafe_allow_html=True,
                )
        price_pct = f" · {result['percentage_difference']:.1f}% vs higher estimate" if result["percentage_difference"] is not None else ""
        st.markdown(
            f'<div class="compare-delta"><strong>Difference: {escape(format_price(result["difference"]))} dataset units{escape(price_pct)}</strong><br>'
            f'Lower predicted price: {escape(str(result["lower_price"]))}</div>', unsafe_allow_html=True,
        )
        st.header("Specification comparison")
        specs_a, specs_b = result["specs_a"], result["specs_b"]
        comparison_rows = [{"Specification": "Device type", "Gadget A": result["device_a"], "Gadget B": result["device_b"], "Comparison": "Same" if result["device_a"] == result["device_b"] else "Different"}]
        comparison_rows.extend(
            {"Specification": name, "Gadget A": value, "Gadget B": specs_b.get(name, "—"),
             "Comparison": "Same" if value == specs_b.get(name) else "Different"}
            for name, value in specs_a.items()
        )
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
        st.caption("Estimates use the same unverified dataset units. A lower predicted price does not automatically mean a better purchase; condition, specifications and your needs matter.")


def prediction_history():
    initialize_history(st.session_state)
    records = st.session_state["prediction_history"]
    if not records:
        with st.container(border=True):
            card_title("Your history will appear here")
            st.caption("Run a price estimate, analyze a deal, or compare gadgets to add completed predictions to this session.")
        return
    counts = pd.Series([record["device"] for record in records]).value_counts()
    cards = st.columns(3)
    for col, label, value in [
        (cards[0], "Recorded estimates", len(records)),
        (cards[1], "Laptop estimates", int(counts.get("Laptop", 0))),
        (cards[2], "Smartphone estimates", int(counts.get("Smartphone", 0))),
    ]:
        with col:
            statistic_card(label, value)
    st.header("Recent prediction records")
    rows = []
    for record in records:
        row = {"Timestamp": record["timestamp"], "Device": record["device"], **record["specifications"],
               "Predicted price (dataset units)": format_price(record["predicted_price"]),
               "Asking price (dataset units)": format_price(record["asking_price"]) if record.get("asking_price") is not None else "",
               "Deal": record.get("deal") or "",
               "Difference vs estimate": format_price(record["difference"]) if record.get("difference") is not None else "",
               "Difference (%)": f"{record['difference_pct']:+.1f}%" if record.get("difference_pct") is not None else ""}
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.download_button("Download history as CSV", data=history_csv(records), file_name="gadgetworth_prediction_history.csv", mime="text/csv", type="primary")
    confirmed = st.checkbox("I understand this clears history for this session", key="confirm_clear_history")
    if st.button("Clear prediction history", disabled=not confirmed, type="secondary"):
        clear_history(st.session_state)
        st.rerun()

def navigate_to(page):
    st.session_state["nav"] = page


def statistic_card(label, value):
    st.markdown(
        f'<div class="stat-card"><span class="stat-label">{escape(str(label))}</span>'
        f'<span class="stat-value">{escape(str(value))}</span></div>',
        unsafe_allow_html=True,
    )


def main():
    pages = [
        "Home Dashboard", "Price Prediction", "Deal Analyzer", "Compare Gadgets",
        "Prediction History", "Model Performance", "About Project",
    ]
    st.session_state.setdefault("nav", "Home Dashboard")
    st.sidebar.markdown(
        '<div class="brand-lockup"><span class="brand-mark">GW</span>'
        '<span class="brand-copy"><strong>GadgetWorth AI</strong>'
        '<small>USED DEVICE PRICE APPRAISER</small></span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.caption("WORKSPACE")
    for destination in pages:
        st.sidebar.button(
            destination, key=f"nav_{destination.lower().replace(' ', '_')}",
            type="primary" if st.session_state["nav"] == destination else "secondary",
            on_click=navigate_to, args=(destination,),
        )
    page = st.session_state["nav"]

    if page == "Home Dashboard":
        st.markdown('<section class="hero"><div class="hero-kicker">AI-powered device valuation</div><h1>Know what your gadget is worth.</h1><p>Get a data-informed estimate and make your next device decision with confidence.</p></section>', unsafe_allow_html=True)
        cta_a, cta_b, _ = st.columns([1.15, 1.1, 4.75], gap="small")
        cta_a.button("Estimate My Gadget", key="hero_estimate", type="primary", on_click=navigate_to, args=("Price Prediction",))
        cta_b.button("Compare Devices", key="hero_compare", on_click=navigate_to, args=("Compare Gadgets",))

        artifacts = [load_artifact(device) for device in SUPPORTED_DEVICES]
        available = [artifact for artifact in artifacts if artifact is not None]
        total_records = sum(int(artifact.get("dataset_rows", 0)) for artifact in available)
        st.header("Your valuation workspace")
        stat_cols = st.columns(4)
        stats = [
            ("Supported devices", str(len(SUPPORTED_DEVICES))),
            ("Model pipelines", str(len(available))),
            ("Training records", f"{total_records:,}" if total_records else "Not available"),
            ("Prediction tool", "Available" if available else "Models missing"),
        ]
        for col, (label, value) in zip(stat_cols, stats):
            with col:
                statistic_card(label, value)
        if any(artifact.get("synthetic_data") for artifact in available):
            st.info("The current training data is synthetic demonstration data. Its prices are not observed market transactions, and its currency is unverified.")

        st.header("How it works")
        steps = st.columns(3, gap="medium")
        for col, number, title, copy in zip(
            steps, ("01", "02", "03"),
            ("Describe", "Estimate", "Decide"),
            ("Enter the device details.", "A trained regression model estimates a value.", "Compare the estimate with an asking price."),
        ):
            with col:
                st.markdown(
                    f'<section class="step-card"><span class="step-number">STEP {number}</span>'
                    f'<h3>{title}</h3><p>{copy}</p></section>',
                    unsafe_allow_html=True,
                )
        st.caption("Use estimates as a starting point. Predictions depend on training data quality and do not guarantee a sale price.")
    elif page == "Price Prediction":
        st.title("Estimate Your Gadget's Value")
        st.markdown('<p class="page-lead">Enter your device details to get a resale estimate from the selected trained model.</p>', unsafe_allow_html=True)
        device = st.selectbox("Device category", ["Laptop", "Smartphone"])
        prediction_form(device)
    elif page == "Deal Analyzer":
        st.title("Is This Deal Worth It?")
        st.markdown('<p class="page-lead">Compare the seller’s asking price with the model’s estimate for a similar used device.</p>', unsafe_allow_html=True)
        device = st.selectbox("Device category", ["Laptop", "Smartphone"])
        prediction_form(device, True)
    elif page == "Compare Gadgets":
        compare_gadgets()
    elif page == "Prediction History":
        st.title("Your Prediction History")
        st.markdown('<p class="page-lead">Review and export estimates recorded during this active session.</p>', unsafe_allow_html=True)
        prediction_history()
    elif page == "Model Performance":
        st.title("Model Performance & Evaluation")
        st.markdown('<p class="page-lead">See how the selected regression model estimates used-device prices on its held-out test data.</p>', unsafe_allow_html=True)
        device = st.selectbox("Device category", ["Laptop", "Smartphone"])
        artifact = load_artifact(device)
        if artifact is None:
            show_missing(device)
        else:
            if artifact.get("synthetic_data"):
                st.warning("Held-out results on synthetic data do not establish real-market accuracy. Currency and source units are unverified.")
            st.write(f"Selected model: **{artifact['model_name']}**")
            results = pd.DataFrame(artifact.get("metrics", []))
            selected_rows = results.loc[results["Model"] == artifact["model_name"]] if not results.empty and "Model" in results else pd.DataFrame()
            if results.empty or selected_rows.empty:
                st.info("Evaluation metrics are not available for this saved model artifact.")
            else:
                selected = selected_rows.iloc[0]
                metric_cols = st.columns(3)
                for col, metric, label, help_text in zip(
                    metric_cols, ("MAE", "RMSE", "R2"),
                    ("Test MAE", "Test RMSE", "Test R-squared"),
                    ("Average absolute test error; lower is better.", "Penalizes larger test errors more strongly; lower is better.", "Variance explained on the held-out test split; not an accuracy percentage."),
                ):
                    with col:
                        if metric in selected and pd.notna(selected[metric]):
                            st.metric(label, f"{selected[metric]:,.3f}", help=help_text)
                        else:
                            st.metric(label, "Not available", help=help_text)
                with st.container(border=True):
                    card_title("Model comparison")
                    st.dataframe(results, width="stretch", hide_index=True)
                for metric, title in [("MAE", "Mean absolute error (dataset units) - lower is better"), ("RMSE", "Root mean squared error (dataset units) - lower is better"), ("R2", "R-squared - closer to 1 is better")]:
                    if metric in results:
                        with st.container(border=True):
                            st.plotly_chart(style_chart(px.bar(results, x="Model", y=metric, color="Model", title=title)), width="stretch")
            evaluation = artifact.get("evaluation", [])
            if evaluation:
                with st.container(border=True):
                    ev = pd.DataFrame(evaluation)
                    st.plotly_chart(style_chart(px.scatter(ev, x="Actual", y="Predicted", title="Actual vs predicted", opacity=.65)), width="stretch")
            importance = global_feature_importance(artifact)
            if importance:
                with st.container(border=True):
                    st.plotly_chart(style_chart(px.bar(pd.DataFrame(importance), x="Importance", y="Feature", orientation="h", title="Global feature importance by input")), width="stretch")
            st.caption("MAE and RMSE use unverified dataset units. Scores are based on synthetic records and do not prove real-market accuracy.")
    else:
        st.title("About GadgetWorth AI")
        st.markdown('<p class="page-lead">A second-hand device valuation project built around reproducible machine learning and transparent limitations.</p>', unsafe_allow_html=True)
        sections = [
            ("Project introduction", "GadgetWorth AI is a second-hand laptop and smartphone price app. It provides a model-based estimate and tools to compare asking prices and device estimates."),
            ("Problem statement", "Used-device listings vary in age, condition, battery health and specifications. Buyers and sellers need a consistent reference point when discussing a price, while recognizing that one estimate cannot replace a device inspection or current market research."),
            ("Objectives", "Estimate a device price using its supported specifications; compare an asking price with the estimate; compare two device estimates; and present model evaluation results with clear data limitations."),
            ("Dataset and preprocessing", "Training uses `data/used_device_data.csv` with 1,200 synthetic demonstration records, split evenly between laptops and smartphones. Inputs are brand, RAM, storage, age, battery health, condition and original price. Preprocessing is fit inside each Scikit-learn pipeline to avoid fitting transforms on held-out rows. The separate laptop and smartphone CSV files have different schemas/target transformations and are not currently mixed into training."),
            ("Algorithms", "Candidate estimators are Linear Regression, Random Forest Regressor and Gradient Boosting Regressor. Validation MAE selects the saved model for each device category."),
            ("Limitations", "The training data matches the project's seeded synthetic-data generator; it does not represent observed market transactions. The price units and currency are undocumented. Held-out metrics therefore do not establish real-market accuracy, and estimates do not include a defensible prediction interval."),
            ("Future scope", "Collect verified second-hand sale prices with currency, transaction date, model, location, condition and data provenance. Retrain and evaluate on genuinely held-out observed transactions, and monitor performance as the market changes."),
        ]
        with st.container(border=True):
            card_title("Machine learning workflow")
            steps = st.columns(4)
            for col, number, heading, detail in zip(
                steps, ("01", "02", "03", "04"),
                ("Prepare", "Train", "Evaluate", "Estimate"),
                ("Load and preprocess device data.", "Fit candidate regression pipelines.", "Select using validation MAE; report held-out metrics.", "Load the saved artifact in the Streamlit app."),
            ):
                with col:
                    st.markdown(f"**{number} · {heading}**")
                    st.caption(detail)
        with st.container(border=True):
            card_title("Technologies")
            tech_columns = st.columns(4)
            for col, name, detail in zip(
                tech_columns, ("Python", "Pandas · NumPy", "Scikit-learn · Joblib", "Streamlit · Plotly"),
                ("Application and training language", "Data preparation", "Model pipelines and saved artifacts", "Interactive app and charts"),
            ):
                with col:
                    st.markdown(f'<div class="about-tech"><strong>{escape(name)}</strong><span>{escape(detail)}</span></div>', unsafe_allow_html=True)
        for heading, content in sections:
            with st.container(border=True):
                card_title(heading)
                st.markdown(content)


if __name__ == "__main__":
    main()
