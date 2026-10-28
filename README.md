# GadgetWorth AI

A Streamlit app that estimates used laptop and smartphone prices, compares seller asks, and displays held-out regression results.

## Setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements pin versions verified in the project's Python 3.12.7 environment, including scikit-learn 1.9.1. Activate `.venv` before training, testing, or launching so model artifacts are created and loaded by the same dependency versions.

## Dataset notes

The training script uses `data/used_device_data.csv`. It has 1,200 records (600 laptops and 600 smartphones) and seven features: brand, RAM, storage, device age, battery health, condition, and original price. The target column is `price`.

The file's full contents exactly match the output of the project's former seeded synthetic-data generator. It is therefore treated as synthetic demonstration data, not as observed market transactions. Its currency and numeric price units are undocumented; the app displays prices in unspecified dataset units and does not convert or label them as INR.

The other supplied files are inspected by `python inspect_data.py` but are not mixed into training. `laptop_data_cleaned.csv` has a different schema and a transformed `Price` target. `used_device_smartphone.csv` has a different schema, normalized price targets, and missing values.

## Inspect data and train

```powershell
python inspect_data.py
python train_model.py
```

Training cleans invalid rows, then trains separate laptop and smartphone pipelines. It compares Linear Regression, Random Forest, and Gradient Boosting using a reproducible 60/20/20 train/validation/test split. Candidate selection uses validation MAE; the test split is reserved for final metrics. Preprocessing is fit inside each training pipeline. Artifacts, metrics, supported input ranges, and actual-versus-predicted test records are saved under `models/`.

Joblib model files are Python and scikit-learn version-sensitive. After changing scikit-learn versions, retrain both artifacts in the active environment with `python train_model.py`; do not edit serialized model files by hand.

MAE and RMSE are in the dataset's unspecified units. R² and all other metrics are held-out results on synthetic records; they do not establish real-market accuracy.

## Run the app

```powershell
python -m streamlit run app.py
```

The prediction and deal pages use the same units as the data. Deal labels compare the asking price to the prediction; they are not a guarantee of value or device quality.

## Tests

```powershell
python -m pytest -q
```
