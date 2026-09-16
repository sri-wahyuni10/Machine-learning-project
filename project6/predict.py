"""
predict.py — Inference script untuk model prediksi Income (Adult Census Income Dataset)

Model: XGBoost Classifier dengan native categorical support
Target: income (0 = <=50K, 1 = >50K)

Cara pakai:
    python predict.py --input data_baru.csv --output hasil_prediksi.csv
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = "model.pkl"
CATEGORY_MAPPING_PATH = "category_mapping.json"

# Kolom yang di-drop karena redundan dengan education-num (identik secara informasi)
COLUMNS_TO_DROP = ["education"]

# Kolom yang mungkin mengandung '?' sebagai representasi missing value
COLUMNS_WITH_QUESTION_MARK = ["workclass", "occupation", "native-country"]

# Nama-nama kolom sesuai urutan asli dataset UCI Adult (dataset tidak punya header)
RAW_COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


def load_artifacts():
    """Load model dan metadata kategori yang disimpan saat training."""
    model = joblib.load(MODEL_PATH)
    with open(CATEGORY_MAPPING_PATH, "r") as f:
        category_mapping = json.load(f)
    return model, category_mapping


def preprocess(df: pd.DataFrame, category_mapping: dict) -> pd.DataFrame:
    """
    Menerapkan preprocessing yang PERSIS SAMA seperti saat training:
    1. Drop kolom redundan (education)
    2. Ganti '?' menjadi np.nan pada kolom yang relevan + buat flag missingness
    3. Gabungkan kategori langka di native-country menjadi 'Other'
       (kategori yang tidak dikenal model otomatis dianggap 'Other')
    4. Convert kolom kategorikal ke tipe 'category' dengan daftar kategori
       yang SAMA seperti saat training (penting untuk XGBoost native categorical)
    """
    df = df.copy()

    # 1. Drop kolom redundan
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # 2. Tangani '?' sebagai missing value + buat flag indikator
    for col in COLUMNS_WITH_QUESTION_MARK:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col] == "?")
            df[col] = df[col].replace("?", np.nan)

    # 3. Untuk native-country: kategori yang tidak dikenal model (di luar
    #    category_mapping hasil training) di-map ke 'Other'
    if "native-country" in df.columns and "native-country" in category_mapping:
        known_categories = set(category_mapping["native-country"])
        df["native-country"] = df["native-country"].apply(
            lambda x: x if (pd.isna(x) or x in known_categories) else "Other"
        )

    # 4. Convert ke tipe category, dengan daftar kategori yang sama seperti training
    for col, categories in category_mapping.items():
        if col in df.columns and col != "income":
            df[col] = pd.Categorical(df[col], categories=categories)

    # Drop kolom target kalau ada di input (misal file punya kolom income asli)
    if "income" in df.columns:
        df = df.drop(columns=["income"])

    return df


def predict(input_path: str, output_path: str, has_header: bool = True):
    model, category_mapping = load_artifacts()

    # Load data baru. Dataset UCI asli tidak punya header -- sesuaikan
    # has_header=False kalau file input juga tanpa header (mengikuti format asli).
    if has_header:
        df_raw = pd.read_csv(input_path, skipinitialspace=True)
    else:
        df_raw = pd.read_csv(
            input_path, header=None, names=RAW_COLUMN_NAMES, skipinitialspace=True
        )

    X = preprocess(df_raw, category_mapping)

    pred_proba = model.predict_proba(X)[:, 1]
    pred_label = model.predict(X)

    result = df_raw.copy()
    result["predicted_income"] = np.where(pred_label == 1, ">50K", "<=50K")
    result["probability_>50K"] = pred_proba

    result.to_csv(output_path, index=False)
    print(f"Prediksi selesai. Hasil disimpan di: {output_path}")
    print(f"Ringkasan: {(pred_label == 1).sum()} diprediksi >50K, "
          f"{(pred_label == 0).sum()} diprediksi <=50K")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi income dari data baru")
    parser.add_argument("--input", required=True, help="Path ke file CSV input")
    parser.add_argument("--output", default="hasil_prediksi.csv", help="Path file output")
    parser.add_argument(
        "--no-header", action="store_true",
        help="Set flag ini jika file input tidak punya header (format asli UCI)"
    )
    args = parser.parse_args()

    predict(args.input, args.output, has_header=not args.no_header)