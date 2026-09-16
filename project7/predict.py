"""
predict.py
Inference script untuk model Home Credit Default Risk (XGBoost + scale_pos_weight).

Mereplikasi seluruh preprocessing manual yang dilakukan saat training,
karena langkah-langkah ini TIDAK tersimpan di dalam file model (.joblib):

1. Agregasi bureau.csv -> merge ke application data (per SK_ID_CURR)
2. Flag HAS_BUREAU_HISTORY + fillna(0) untuk kolom-kolom uang hasil agregasi bureau
3. Flag DAYS_EMPLOYED_ANOM + replace nilai sentinel 365243 -> NaN
4. Drop kolom-kolom properti dengan missing rate tinggi (>40%), kecuali
   EXT_SOURCE_1 dan OWN_CAR_AGE yang sengaja dipertahankan
5. Convert kolom kategorikal ke dtype 'category' (native XGBoost categorical support)
6. Prediksi menggunakan custom threshold (bukan default 0.5), karena data
   sangat imbalanced (~92:8) dan recall lebih diprioritaskan (business cost
   dari False Negative jauh lebih mahal daripada False Positive di domain kredit)

Penggunaan:
    from predict import HomeCreditPredictor

    predictor = HomeCreditPredictor(
        model_path="home_credit_xgb_model.joblib",
        threshold=0.43
    )
    results = predictor.predict(application_df, bureau_df)
"""

import numpy as np
import pandas as pd
import joblib


class HomeCreditPredictor:
    """
    Wrapper inference untuk model Home Credit Default Risk.

    Membungkus seluruh langkah preprocessing manual + model yang sudah dilatih,
    supaya prediksi terhadap data baru konsisten persis dengan proses training.
    """

    # Kolom-kolom hasil agregasi bureau yang aman diisi 0 saat klien tidak
    # punya riwayat bureau sama sekali (tidak ada utang yang diketahui = 0).
    # DAYS_CREDIT_min/max/mean SENGAJA tidak masuk sini -> dibiarkan NaN,
    # karena mengisi 0 pada kolom waktu relatif akan menyiratkan makna palsu
    # ("kredit dimulai hari ini").
    BUREAU_FILL_ZERO_COLS = [
        "AMT_CREDIT_SUM_DEBT_sum",
        "AMT_CREDIT_SUM_OVERDUE_sum",
        "CREDIT_DAY_OVERDUE_sum",
        "CREDIT_DAY_OVERDUE_max",
    ]

    # Sentinel value anomali pada DAYS_EMPLOYED (~1000 tahun), placeholder
    # untuk klien yang tidak bekerja (pensiunan, dll).
    DAYS_EMPLOYED_ANOM_VALUE = 365243

    # Kolom-kolom kategorikal yang perlu di-convert ke dtype 'category'
    # supaya bisa langsung dipakai native XGBoost (enable_categorical=True).
    CATEGORICAL_COLS = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE",
    ]

    # Kolom-kolom properti (missing rate tinggi ~60-70%) yang di-drop saat
    # training, KECUALI EXT_SOURCE_1 dan OWN_CAR_AGE yang sengaja dipertahankan
    # meski missing tinggi juga, karena EXT_SOURCE_1 adalah salah satu prediktor
    # terkuat, dan OWN_CAR_AGE punya makna tersendiri (NaN = tidak punya mobil).
    PROPERTY_COLS_TO_DROP = [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
        "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",
        "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
        "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
        "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
    ]

    def __init__(self, model_path: str, threshold: float = 0.43):
        """
        Parameters
        ----------
        model_path : str
            Path ke file model hasil joblib.dump() (misal 'home_credit_xgb_model.joblib').
        threshold : float
            Threshold klasifikasi custom (default 0.43), dipilih berdasarkan
            analisis precision-recall curve pada data training -- BUKAN
            default 0.5, karena data sangat imbalanced (~92:8) dan recall
            diprioritaskan (menangkap klien berisiko default lebih penting
            daripada menghindari false alarm, mengingat business cost yang
            asimetris di domain kredit).
        """
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def _aggregate_bureau(self, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        Agregasi bureau.csv dari level (banyak baris per klien) menjadi
        1 baris per SK_ID_CURR, siap di-merge ke tabel utama.
        """
        agg = bureau_df.groupby("SK_ID_CURR").agg({
            "AMT_CREDIT_SUM_DEBT": "sum",
            "AMT_CREDIT_SUM_OVERDUE": "sum",
            "DAYS_CREDIT": ["min", "max", "mean"],
            "CREDIT_DAY_OVERDUE": ["sum", "max"],
        })
        agg.columns = ["_".join(col) for col in agg.columns]
        agg = agg.reset_index()
        return agg

    def _add_bureau_features(self, df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """Merge fitur agregasi bureau + flag HAS_BUREAU_HISTORY + fillna selektif."""
        agg_bureau = self._aggregate_bureau(bureau_df)
        df = df.merge(agg_bureau, on="SK_ID_CURR", how="left")

        # Flag: apakah klien punya riwayat kredit di bureau sama sekali
        df["HAS_BUREAU_HISTORY"] = df["DAYS_CREDIT_min"].notnull().astype(int)

        # Isi 0 untuk kolom uang -- tidak ada riwayat = tidak ada utang yang diketahui.
        # Kolom DAYS_CREDIT_min/max/mean sengaja dibiarkan NaN untuk XGBoost.
        fill_cols = [c for c in self.BUREAU_FILL_ZERO_COLS if c in df.columns]
        df[fill_cols] = df[fill_cols].fillna(0)

        return df

    def _fix_days_employed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag anomali DAYS_EMPLOYED (sentinel 365243) lalu replace jadi NaN."""
        df["DAYS_EMPLOYED_ANOM"] = (
            df["DAYS_EMPLOYED"] == self.DAYS_EMPLOYED_ANOM_VALUE
        ).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            self.DAYS_EMPLOYED_ANOM_VALUE, np.nan
        )
        return df

    def _drop_high_missing_property_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop kolom-kolom properti dengan missing rate tinggi."""
        cols_present = [c for c in self.PROPERTY_COLS_TO_DROP if c in df.columns]
        return df.drop(columns=cols_present)

    def _set_categorical_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert kolom kategorikal ke dtype 'category' untuk native XGBoost."""
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def preprocess(self, application_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        Menjalankan seluruh pipeline preprocessing manual secara berurutan.

        Parameters
        ----------
        application_df : pd.DataFrame
            Data aplikasi pinjaman (setara application_train/test.csv),
            wajib punya kolom SK_ID_CURR.
        bureau_df : pd.DataFrame
            Data riwayat kredit dari bureau (setara bureau.csv),
            wajib punya kolom SK_ID_CURR.

        Returns
        -------
        pd.DataFrame
            Data yang sudah siap dipakai untuk prediksi (fitur saja,
            SK_ID_CURR sudah dilepas dari fitur tapi index tetap sejalan
            dengan urutan baris input).
        """
        df = application_df.copy()

        df = self._add_bureau_features(df, bureau_df)
        df = self._fix_days_employed(df)
        df = self._drop_high_missing_property_cols(df)
        df = self._set_categorical_dtypes(df)

        return df

    def predict(self, application_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing + prediksi end-to-end.

        Returns
        -------
        pd.DataFrame dengan kolom:
            - SK_ID_CURR
            - default_probability : probabilitas default (kelas 1) mentah dari model
            - prediction : 0/1 hasil klasifikasi berdasarkan self.threshold
        """
        processed = self.preprocess(application_df, bureau_df)

        sk_id_curr = processed["SK_ID_CURR"].copy()
        X = processed.drop(columns=["SK_ID_CURR"])
        if "TARGET" in X.columns:
            X = X.drop(columns=["TARGET"])

        proba = self.model.predict_proba(X)[:, 1]
        prediction = (proba >= self.threshold).astype(int)

        return pd.DataFrame({
            "SK_ID_CURR": sk_id_curr,
            "default_probability": proba,
            "prediction": prediction,
        })


if __name__ == "__main__":
    # Contoh penggunaan
    application_df = pd.read_csv("application_test.csv")
    bureau_df = pd.read_csv("bureau.csv")

    predictor = HomeCreditPredictor(
        model_path="home_credit_xgb_model.joblib",
        threshold=0.43,
    )

    results = predictor.predict(application_df, bureau_df)
    print(results.head())
    results.to_csv("predictions.csv", index=False)
    print(f"\nTotal klien: {len(results)}")
    print(f"Diprediksi berisiko (1): {results['prediction'].sum()} "
          f"({results['prediction'].mean():.1%})")