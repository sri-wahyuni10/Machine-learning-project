"""
predict.py — Inference wrapper untuk model prediksi Sales (synthetic Rossmann-style).

PENTING: Fungsi di sini mereplikasi PERSIS langkah-langkah feature engineering
yang dipakai saat training, supaya preprocessing training vs inference konsisten.
Kalau kamu ubah logika feature engineering di notebook training, ubah juga di sini.

SYARAT INPUT:
- new_data: DataFrame mentah dengan kolom minimal:
  ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
  (Date sebagai datetime atau string yang bisa di-parse)
- history_data: DataFrame riwayat Sales AKTUAL toko yang sama, MINIMAL 7 hari
  sebelum tanggal paling awal di new_data. Kolom: ['Store', 'Date', 'Sales'].
  Tanpa ini, Sales_lag_7 dan Sales_rolling_mean_7 akan NaN dan prediksi gagal.

FILE YANG DIBUTUHKAN (satu folder dengan script ini):
- model.pkl           -> model LightGBM hasil training (lihat save_model_artifacts.py)
- model_metadata.pkl  -> feature_cols & cat_cols (lihat save_model_artifacts.py)
- store.csv           -> atribut statis toko
- calendar.csv        -> referensi IsLebaranShoppingWindow
"""

import numpy as np
import pandas as pd
import joblib


def load_artifacts(model_path='/model.pkl', metadata_path='model_metadata.pkl',
                    store_path='store.csv', calendar_path='calendar.csv'):
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    store_df = pd.read_csv(store_path)
    calendar_df = pd.read_csv(calendar_path, parse_dates=['Date'])
    return model, metadata, store_df, calendar_df


def engineer_features(new_data: pd.DataFrame, history_data: pd.DataFrame,
                       store_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    df = new_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # --- 1. Merge atribut toko ---
    df = df.merge(store_df, on='Store', how='left')

    # --- 2. Kolom waktu dasar ---
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['WeekofYear'] = df['Date'].dt.isocalendar().week.astype('int32')
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                 7: 'Jul', 8: 'Agu', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'}
    df['Month_abbr'] = df['Month'].map(month_map)

    # --- 3. IsMonthInPromoInterval ---
    df['PromoInterval_filled'] = df['PromoInterval'].fillna('')
    df['IsMonthInPromoInterval'] = df.apply(
        lambda row: row['Month_abbr'] in row['PromoInterval_filled'].split(','), axis=1
    )

    # --- 4. PromoStarted & IsPromo2ActiveOnThisDate ---
    df['PromoStarted'] = (
        (df['Year'] > df['Promo2SinceYear']) |
        ((df['Year'] == df['Promo2SinceYear']) & (df['WeekofYear'] >= df['Promo2SinceWeek']))
    )
    df['IsPromo2ActiveOnThisDate'] = (
        (df['Promo2'] == 1) & df['PromoStarted'] & df['IsMonthInPromoInterval']
    )

    # --- 5. IsCompetitorActive & EffectiveCompetitionDistance ---
    df['IsCompetitorActive'] = (
        (df['Year'] > df['CompetitionOpenSinceYear']) |
        ((df['Year'] == df['CompetitionOpenSinceYear']) & (df['Month'] >= df['CompetitionOpenSinceMonth']))
    )
    df['EffectiveCompetitionDistance'] = np.where(
        df['IsCompetitorActive'], df['CompetitionDistance'], 999999
    )

    # --- 6. IsLebaranShoppingWindow (merge dari calendar.csv) ---
    df = df.merge(calendar_df[['Date', 'IsLebaranShoppingWindow']], on='Date', how='left')

    # --- 7. Lag & rolling feature (WAJIB pakai history_data, bukan cuma new_data) ---
    hist = history_data.copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    combined = pd.concat([
        hist[['Store', 'Date', 'Sales']],
        df[['Store', 'Date']].assign(Sales=np.nan)  # placeholder baris yang mau diprediksi
    ], ignore_index=True)
    combined = combined.sort_values(['Store', 'Date']).drop_duplicates(subset=['Store', 'Date'])

    combined['Sales_lag_7'] = combined.groupby('Store')['Sales'].shift(7)
    combined['Sales_rolling_mean_7'] = combined.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )

    df = df.merge(
        combined[['Store', 'Date', 'Sales_lag_7', 'Sales_rolling_mean_7']],
        on=['Store', 'Date'], how='left'
    )

    return df


def predict(new_data: pd.DataFrame, history_data: pd.DataFrame,
            model_path='model.pkl', metadata_path='model_metadata.pkl',
            store_path='store.csv', calendar_path='calendar.csv') -> pd.DataFrame:
    """
    Fungsi utama. Return dataframe asli + kolom Predicted_Sales.
    """
    model, metadata, store_df, calendar_df = load_artifacts(
        model_path, metadata_path, store_path, calendar_path
    )
    feature_cols = metadata['feature_cols']
    cat_cols = metadata['cat_cols']

    df = engineer_features(new_data, history_data, store_df, calendar_df)

    # Cek lag feature yang gagal dihitung (history tidak cukup)
    missing_lag = df['Sales_lag_7'].isna().sum()
    if missing_lag > 0:
        print(f"PERINGATAN: {missing_lag} baris tidak punya cukup riwayat 7 hari "
              f"untuk hitung Sales_lag_7 — prediksinya akan NaN. "
              f"Pastikan history_data mencakup minimal 7 hari sebelum tanggal prediksi.")

    X = df[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    df['Predicted_Sales'] = model.predict(X)
    df.loc[df['Open'] == 0, 'Predicted_Sales'] = 0  # toko tutup -> pasti 0, tidak perlu model

    return df[['Store', 'Date', 'Open', 'Predicted_Sales']]


if __name__ == '__main__':
    # --- Contoh pemakaian ---
    # new_data: baris yang mau diprediksi (Sales belum diketahui)
    new_data = pd.DataFrame({
        'Store': [1, 1],
        'Date': ['2024-07-01', '2024-07-02'],
        'DayOfWeek': [1, 2],
        'Open': [1, 1],
        'Promo': [0, 1],
        'StateHoliday': ['0', '0'],
        'SchoolHoliday': [0, 0],
    })

    # history_data: Sales AKTUAL 7+ hari sebelum tanggal di atas, untuk toko yang sama
    history_data = pd.DataFrame({
        'Store': [1] * 10,
        'Date': pd.date_range('2024-06-21', periods=10, freq='D'),
        'Sales': [5200, 5400, 7800, 5100, 4900, 5300, 5600, 5500, 5700, 7900],
    })

    result = predict(new_data, history_data)
    print(result)