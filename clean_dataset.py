from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "hypertension_dataset_5000.csv"
OUTPUT_FILE = ROOT / "cleaned_hypertension_dataset.csv"


NUMERIC_RULES = {
    "Age": (18, 100),
    "BMI": (10, 70),
    "Waist_Circumference_cm": (40, 200),
    "Sleep_Hours": (0, 24),
    "Creatinine_mg_dL": (0.1, 20),
    "Fasting_Glucose_mg_dL": (40, 500),
    "Cholesterol_mg_dL": (50, 500),
    "Systolic_BP_mmHg": (70, 250),
    "Diastolic_BP_mmHg": (40, 150),
}


def clean_dataset():
    df = pd.read_csv(INPUT_FILE)
    original_rows = len(df)

    df = df.drop_duplicates()
    duplicates_removed = original_rows - len(df)

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].str.replace(r"\s+", " ", regex=True)

    categorical_aliases = {
        "Gender": {"male": "Male", "female": "Female"},
        "Salt_Intake": {
            "low": "Low",
            "normal": "Normal",
            "high": "High",
            "very high": "Very High",
        },
        "Physical_Activity": {
            "light": "Light",
            "sedentary": "Sedentary",
            "moderate": "Moderate",
            "active": "Active",
        },
        "Alcohol_Consumption": {
            "occasional": "Occasional",
            "moderate": "Moderate",
            "heavy": "Heavy",
        },
        "Smoking_Status": {
            "former": "Former",
            "current": "Current",
            "never": "Never",
        },
        "Stress_Level": {
            "low": "Low",
            "moderate": "Moderate",
            "high": "High",
            "very high": "Very High",
        },
        "Family_History_HTN": {"no": "No", "yes": "Yes"},
        "BP_Category": {
            "normal": "Normal",
            "elevated": "Elevated",
            "stage 1 hypertension": "Stage 1 Hypertension",
            "stage 2 hypertension": "Stage 2 Hypertension",
        },
        "Hypertension": {"no": "No", "yes": "Yes"},
    }

    for column, aliases in categorical_aliases.items():
        if column in df:
            normalized = df[column].str.strip().str.lower()
            df[column] = normalized.map(aliases).fillna(df[column])

    for column in df.select_dtypes(include=["object", "string"]).columns:
        missing_count = df[column].isna().sum()
        if missing_count:
            df[column] = df[column].fillna("Unknown")

    imputation_counts = {}
    out_of_range_counts = {}
    for column, (minimum, maximum) in NUMERIC_RULES.items():
        if column not in df:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        imputation_counts[column] = int(numeric.isna().sum())
        out_of_range = numeric.lt(minimum) | numeric.gt(maximum)
        out_of_range_counts[column] = int(out_of_range.sum())
        numeric = numeric.mask(out_of_range)

        median = numeric.median()
        df[column] = numeric.fillna(median)

    df = df.dropna(how="all")

    for column in df.columns:
        if df[column].dtype == "string":
            df[column] = df[column].astype("object")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Input rows: {original_rows:,}")
    print(f"Duplicate rows removed: {duplicates_removed:,}")
    print(f"Output rows: {len(df):,}")
    print(f"Output columns: {df.shape[1]}")
    print(f"Missing values remaining: {int(df.isna().sum().sum())}")
    print(f"Numeric imputation counts: {imputation_counts}")
    print(f"Out-of-range values replaced: {out_of_range_counts}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_dataset()
