"""Inspect supplied CSV schemas, quality, and price scales without altering data."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASETS = [
    "used_device_data.csv",
    "laptop_data_cleaned.csv",
    "used_device_smartphone.csv",
]
PRICE_COLUMNS = [
    "price", "original_price", "Price", "normalized_used_price", "normalized_new_price",
]


def inspect_csv(path):
    path = Path(path)
    if not path.is_file():
        print(f"\n--- {path.name}: FILE MISSING ---")
        print(f"Expected file at {path}")
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"\n--- {path.name}: COULD NOT READ ---\n{exc}")
        return None

    print(f"\n{'=' * 72}\nDATASET: {path.name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data types:\n", df.dtypes.to_string())
    print("Missing values:\n", df.isna().sum().to_string())
    print("Duplicate rows:", int(df.duplicated().sum()))
    for column in ("device", "brand", "condition", "device_brand"):
        if column in df:
            values = sorted(df[column].dropna().astype(str).unique().tolist())
            print(f"Unique {column} ({len(values)}):", values)

    numeric = df.select_dtypes(include="number")
    print("Numeric feature ranges:")
    if numeric.empty:
        print("  No numeric columns found.")
    else:
        for column in numeric:
            series = numeric[column].dropna()
            if not series.empty:
                print(f"  {column}: min={series.min():g}, max={series.max():g}")

    print("Price summary statistics:")
    found = False
    for column in PRICE_COLUMNS:
        if column in df:
            found = True
            values = pd.to_numeric(df[column], errors="coerce")
            print(f"\n{column}:\n{values.describe().to_string()}")
    if not found:
        print("  No recognized target or original price column found.")
    return df


def main():
    for filename in DATASETS:
        inspect_csv(DATA_DIR / filename)


if __name__ == "__main__":
    main()
