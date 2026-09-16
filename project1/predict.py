"""
SCRIPT UJI COBA MODEL: model_churn_pipeline.pkl

PENTING: model yang disimpan cuma berisi ColumnTransformer + classifier,
BUKAN langkah cleaning manual (simplifikasi 'No internet service',
perbaikan representasi True/False/0/1) yang dilakukan di script training.
Jadi data baru HARUS melewati fungsi cleaning yang sama dulu, baru masuk
ke pipeline -- kalau tidak, kategori seperti 'No internet service' akan
dianggap kategori asing (diabaikan / semua-nol) bukan dipetakan ke 'No'.
"""

import pandas as pd
import joblib

# ======================================================================
# 1. LOAD MODEL YANG SUDAH DISIMPAN
# ======================================================================
pipeline = joblib.load('model_churn_pipeline.pkl')
print("Model berhasil dimuat.")


# ======================================================================
# 2. FUNGSI CLEANING -- HARUS SAMA PERSIS DENGAN YANG DIPAKAI SAAT TRAINING
# ======================================================================
def clean_new_data(df):
    """Replikasi cleaning dari script training (tahap 3), supaya data
    baru konsisten dengan data yang dipakai saat model dilatih."""
    df = df.copy()

    cols_to_simplify = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in cols_to_simplify:
        if col in df.columns:
            df[col] = df[col].replace(
                {'No internet service': 'No', 'No phone service': 'No'}
            )

    bool_like_map = {
        'True': 'Yes', 'False': 'No', 'true': 'Yes', 'false': 'No',
        '1': 'Yes', '0': 'No',
    }
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace(bool_like_map)

    return df


# Threshold final -- ganti sesuai keputusan bisnis kamu dari tahap
# threshold tuning (0.35-0.4 kalau mengutamakan recall, 0.5 kalau default)
THRESHOLD = 0.4


# ======================================================================
# 3. CONTOH 1: PREDIKSI UNTUK 1 PELANGGAN BARU
# ======================================================================
# Isi manual, meniru struktur kolom yang dipakai saat training
# (semua kolom HARUS ada, kecuali 'customerID' dan 'Churn')
new_customer = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 'No',
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 3,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.0,
    'TotalCharges': 285.0,
}])

new_customer_clean = clean_new_data(new_customer)

proba = pipeline.predict_proba(new_customer_clean)[:, 1][0]
prediction = "Churn" if proba >= THRESHOLD else "Tidak Churn"

print(f"\n--- Prediksi 1 pelanggan ---")
print(f"Probabilitas churn : {proba:.3f}")
print(f"Threshold dipakai  : {THRESHOLD}")
print(f"Prediksi           : {prediction}")


# ======================================================================
# 4. CONTOH 2: PREDIKSI BATCH DARI FILE CSV
# ======================================================================
# Ganti path ini dengan file CSV pelanggan baru kamu.
# Format kolom harus sama seperti data training (tanpa kolom 'Churn').
try:
    new_data = pd.read_csv('pelanggan_baru.csv')

    if 'customerID' in new_data.columns:
        customer_ids = new_data['customerID']
        new_data = new_data.drop(columns=['customerID'])
    else:
        customer_ids = pd.Series(range(len(new_data)), name='row_id')

    new_data_clean = clean_new_data(new_data)

    probas = pipeline.predict_proba(new_data_clean)[:, 1]
    predictions = ["Churn" if p >= THRESHOLD else "Tidak Churn" for p in probas]

    hasil = pd.DataFrame({
        'customerID': customer_ids,
        'probabilitas_churn': probas.round(3),
        'prediksi': predictions,
    }).sort_values('probabilitas_churn', ascending=False)

    print(f"\n--- Prediksi batch ({len(hasil)} pelanggan) ---")
    print(hasil.head(10))

    hasil.to_csv('hasil_prediksi_churn.csv', index=False)
    print("\nHasil lengkap disimpan ke 'hasil_prediksi_churn.csv'")

except FileNotFoundError:
    print("\n(Lewati contoh batch: file 'pelanggan_baru.csv' belum ada di folder ini)")