from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.prediction import classify_deal, format_price, load_model, predict_price

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="GadgetWorth AI", page_icon="💻", layout="wide")
st.markdown("""
<style>
:root { color-scheme:light; }
.stApp { background:#F4F7FB; color:#263445; }
[data-testid="stAppViewContainer"] { background:#F4F7FB; color:#263445; }
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4 { color:#15304F; }
[data-testid="stAppViewContainer"] p,[data-testid="stAppViewContainer"] li,[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] { color:#263445; }
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stCaptionContainer"] { color:#4B5563 !important; }
[data-testid="stTooltipContent"], [role="tooltip"] { color:#263445 !important; background:#FFFFFF !important; }
[data-baseweb="tab"] { color:#374151 !important; }
[data-baseweb="tab"][aria-selected="true"] { color:#15304F !important; }
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary span { color:#15304F !important; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #DCE6F0; }
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#15304F !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#4B5563 !important; }
[data-testid="stMetric"], .gw-card { background:#FFFFFF; padding:20px; border:1px solid #DCE6F0; border-radius:18px; box-shadow:0 8px 24px #193b5910; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color:#15304F !important; }
input, textarea, [data-baseweb="select"] > div { color:#1F2937 !important; background-color:#FFFFFF !important; }
input::placeholder, textarea::placeholder { color:#6B7280 !important; opacity:1; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background:#FFFFFF !important; color:#1F2937 !important; }
[role="option"], [role="option"] span { color:#1F2937 !important; }
input:disabled, textarea:disabled { color:#6B7280 !important; background:#F3F4F6 !important; }
div.stButton > button, button[kind="primary"], [data-testid="stFormSubmitButton"] button { border-radius:12px; font-weight:700; background:#15304F; color:#FFFFFF !important; border:1px solid #15304F; }
div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background:#0F766E; border-color:#0F766E; color:#FFFFFF !important; }
div.st-key-estimate_my_gadget button,
div.st-key-estimate_price_submit button,
div.st-key-analyze_deal_submit button { background:#0D9488 !important; border-color:#0D9488 !important; color:#FFFFFF !important; }
div.st-key-estimate_my_gadget button:hover,
div.st-key-estimate_price_submit button:hover,
div.st-key-analyze_deal_submit button:hover { background:#0F766E !important; border-color:#0F766E !important; color:#FFFFFF !important; }
[data-testid="stDataFrame"] { background:#FFFFFF; color:#263445; }
[data-testid="stAlert"] p,[data-testid="stAlert"] span { color:inherit; }
.hero { background:linear-gradient(120deg,#15304F,#0F766E); color:#FFFFFF; padding:42px; border-radius:24px; margin:12px 0 26px; }
.hero h1,.hero p { color:#FFFFFF !important; }
.eyebrow { color:#B8FFF0; text-transform:uppercase; letter-spacing:.14em; font-weight:700; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


def style_chart(fig):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"color": "#263445"}, title_font={"color": "#15304F"},
        legend={"font": {"color": "#263445"}},
    )
    fig.update_xaxes(title_font={"color": "#374151"}, tickfont={"color": "#4B5563"}, gridcolor="#E5EAF0", zerolinecolor="#DCE6F0")
    fig.update_yaxes(title_font={"color": "#374151"}, tickfont={"color": "#4B5563"}, gridcolor="#E5EAF0", zerolinecolor="#DCE6F0")
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


def prediction_form(device, analyzer=False):
    artifact = load_artifact(device)
    if artifact is None:
        show_missing(device)
        return
    if artifact.get("synthetic_data"):
        st.warning("This dataset exactly matches the project's previous seeded synthetic-data generator. Estimates are demonstration values, not observed market prices. Currency is unverified.")
    st.markdown(f"### {'Compare an asking price' if analyzer else 'Tell us about your device'}")
    with st.form("device_form"):
        a, b = st.columns(2)
        category_options = artifact.get("categories", {})
        brands = category_options.get("brand", [])
        conditions = category_options.get("condition", [])
        if not brands or not conditions:
            st.error("This model is missing its supported input categories. Retrain the models.")
            return
        with a:
            brand = st.selectbox("Brand", brands, help="Choose the closest match.")
            numeric_ranges = artifact.get("numeric_ranges", {})
            value_options = artifact.get("value_options", {})
            ram_options = value_options.get("ram_gb", [])
            storage_options = value_options.get("storage_gb", [])
            if not ram_options or not storage_options:
                st.error("This model is missing supported RAM or storage options. Retrain the models.")
                return
            ram = st.selectbox("RAM (GB)", ram_options)
            storage = st.selectbox("Storage (GB)", storage_options)
            age_bounds = numeric_ranges.get("age_years", {"min": 0, "max": 20, "median": 2})
            age = st.number_input("Device age (years)", min_value=float(age_bounds["min"]), max_value=float(age_bounds["max"]), value=float(age_bounds["median"]), step=.5)
        with b:
            battery_bounds = numeric_ranges.get("battery_health", {"min": 0, "max": 100, "median": 85})
            battery_min = int(np.ceil(battery_bounds["min"]))
            battery_max = int(np.floor(battery_bounds["max"]))
            battery = st.slider("Battery health (%)", battery_min, battery_max, int(round(battery_bounds["median"])))
            condition = st.selectbox("Physical condition", conditions)
            original_bounds = numeric_ranges.get("original_price", {"min": 1, "max": 10000000, "median": 60000})
            original = st.number_input("Original price (dataset units; currency unverified)", min_value=int(np.ceil(original_bounds["min"])), max_value=int(np.floor(original_bounds["max"])), value=int(round(original_bounds["median"])), step=1000)
            price_bounds = numeric_ranges.get("price", {"min": 1, "max": 10000000, "median": 30000})
            asking = st.number_input("Seller asking price (same dataset units)", min_value=max(1, int(np.ceil(price_bounds["min"]))), max_value=max(1, int(np.floor(price_bounds["max"]))), value=max(1, int(round(price_bounds["median"]))), step=500) if analyzer else None
        submitted = st.form_submit_button(
            "Analyze deal" if analyzer else "Estimate my gadget",
            type="primary",
            key="analyze_deal_submit" if analyzer else "estimate_price_submit",
            width="stretch",
        )
    if submitted:
        try:
            with st.spinner("Estimating resale value…"):
                fair = predict_price(device, {"brand": brand, "ram_gb": ram, "storage_gb": storage, "age_years": age, "battery_health": battery, "condition": condition, "original_price": original}, model_dir=MODEL_DIR)
            st.markdown("### Estimated resale value")
            st.metric("Estimated price (dataset units)", format_price(fair))
            cols = st.columns(4)
            for col, (name, val) in zip(cols, [("Brand", brand), ("RAM", f"{ram} GB"), ("Storage", f"{storage} GB"), ("Age", f"{age:g} years")]):
                col.metric(name, val)
            st.caption(f"Condition: {condition} ? Battery: {battery}% ? Original price: {format_price(original)} dataset units")
            if analyzer:
                label, diff, pct = classify_deal(asking, fair)
                st.markdown("### Deal analysis")
                if label == "Great Deal": st.success(f"🟢 {label} — at least 15% below the estimated fair value.")
                elif label == "Fair Price": st.info(f"🔵 {label} — between 15% below and 10% above estimated fair value.")
                else: st.error(f"🟠 {label} — more than 10% above estimated fair value.")
                x, y, z = st.columns(3)
                x.metric("Asking price (dataset units)", format_price(asking)); y.metric("Difference vs fair", format_price(diff)); z.metric("Difference", f"{pct:+.1f}%")
                st.warning("A low price does not guarantee a good purchase. Verify authenticity, repairs, warranty and battery condition.")
            st.caption("Point estimate only; no defensible prediction interval is available.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The estimate could not be calculated. Check that the model artifact matches this app, then retrain it if needed.")


def main():
    st.sidebar.markdown("## ◈ GadgetWorth AI")
    st.sidebar.caption("USED DEVICE PRICE APPRAISER")
    pages = ["Home Dashboard", "Price Prediction", "Deal Analyzer", "Model Performance", "About Project"]
    page = st.sidebar.radio("Explore", pages, key="nav")
    if page == "Home Dashboard":
        st.markdown('<div class="hero"><div class="eyebrow">Smarter second-hand decisions</div><h1>Know what your gadget is worth.</h1><p>Estimate a used laptop or smartphone resale value, then make your next move with more context.</p></div>', unsafe_allow_html=True)
        st.button("Estimate my gadget →", type="primary", key="estimate_my_gadget", on_click=go_to_prediction)
        c1, c2 = st.columns(2)
        c1.markdown("<div class='gw-card'><h3>💻 Laptop appraisal</h3><p>Compare brand, memory, age, condition and more.</p></div>", unsafe_allow_html=True)
        c2.markdown("<div class='gw-card'><h3>📱 Smartphone appraisal</h3><p>Get a consistent model-based resale estimate.</p></div>", unsafe_allow_html=True)
        st.markdown("### How it works")
        a,b,c=st.columns(3); a.markdown("**01 · Describe**<br>Enter the device details.",unsafe_allow_html=True); b.markdown("**02 · Estimate**<br>A trained regression model predicts a value.",unsafe_allow_html=True); c.markdown("**03 · Decide**<br>Compare the estimate with an asking price.",unsafe_allow_html=True)
        st.info("For buyers and sellers: use a consistent estimate as a starting point for negotiation. Predictions depend on dataset quality and synthetic demo models are not market evidence.")
    elif page == "Price Prediction":
        st.title("Price prediction"); device=st.selectbox("Device category",["Laptop","Smartphone"]); prediction_form(device)
    elif page == "Deal Analyzer":
        st.title("Deal analyzer"); device=st.selectbox("Device category",["Laptop","Smartphone"]); prediction_form(device,True)
    elif page == "Model Performance":
        st.title("Model performance"); device=st.selectbox("Device category",["Laptop","Smartphone"]); artifact=load_artifact(device)
        if artifact is None: show_missing(device)
        else:
            if artifact.get("synthetic_data"): st.warning("Held-out results on a synthetic dataset. These scores do not establish real-market accuracy; the currency and source units are unverified.")
            st.write(f"Selected model: **{artifact['model_name']}**")
            results=pd.DataFrame(artifact["metrics"])
            st.dataframe(results,width="stretch",hide_index=True)
            for metric, title in [("MAE", "Mean absolute error (dataset units) - lower is better"), ("RMSE", "Root mean squared error (dataset units) - lower is better"), ("R2", "R squared - closer to 1 is better")]:
                st.plotly_chart(style_chart(px.bar(results,x="Model",y=metric,color="Model",title=title)),width="stretch")
            evaluation=artifact.get("evaluation",[])
            if evaluation:
                ev=pd.DataFrame(evaluation); st.plotly_chart(style_chart(px.scatter(ev,x="Actual",y="Predicted",title="Actual vs predicted",opacity=.65)),width="stretch")
            importance=artifact.get("feature_importance",[])
            if importance: st.plotly_chart(style_chart(px.bar(pd.DataFrame(importance),x="Importance",y="Feature",orientation="h",title="Selected model feature importance")),width="stretch")
            st.caption("MAE and RMSE are measured in the dataset's unverified numeric units. R squared summarizes variance explained. These held-out results use synthetic records.")
    else:
        st.title("About GadgetWorth AI")
        st.markdown("""### Problem statement
Used-device buyers and sellers need a consistent starting point for price discussions.

### Dataset and features
Training uses `data/used_device_data.csv`: 1,200 records, split evenly between laptops and smartphones, with brand, RAM, storage, age, battery health, condition, original price, and target price. All seven input features are used by the separate device models.

### Data quality and provenance
The main CSV matches every row from the project's former seeded synthetic-data generator. It is synthetic, not observed market transactions. The numeric target units and currency are undocumented, so estimates and deal comparisons are shown in unspecified dataset units. The laptop CSV has a transformed `Price` target and a different schema. The smartphone CSV uses normalized price targets and has missing values; neither is mixed into training.

### Technologies and workflow
Python, Pandas, Scikit-learn Pipelines, Joblib, Streamlit and Plotly. A reproducible train/validation/test split selects models using validation MAE and reserves the test split for final evaluation. Preprocessing is fit inside each training pipeline. Linear Regression, Random Forest and Gradient Boosting are compared; the lowest held-out MAE model is selected.

### Evaluation and limitations
MAE, RMSE and R? summarize held-out performance on synthetic records only and do not prove real-market accuracy. Real listings, verified currency, location, warranty, repair history and demand are not represented.

### Future enhancements
Collect verified market listings with explicit currency and provenance, then retrain and evaluate on genuinely held-out observed transactions.""")


if __name__ == "__main__":
    main()
