# Home Credit Default Risk — XGBoost + SHAP

Prediksi risiko gagal bayar (default) pada aplikasi pinjaman, menggunakan data multi-tabel (aplikasi + riwayat kredit bureau), native categorical support di XGBoost, dan SHAP untuk interpretability.

## Ringkasan Bisnis

Home Credit ingin memprediksi kemampuan bayar klien sebelum menyetujui pinjaman. Kesalahan model di sini punya biaya bisnis yang **asimetris**:
- **False Negative** (klien default, tapi diloloskan) → kerugian pokok pinjaman, jauh lebih mahal.
- **False Positive** (klien baik, tapi ditolak) → kehilangan potensi margin bunga, lebih murah.

Karena itu, model ini dioptimalkan untuk **recall tinggi** (menangkap sebanyak mungkin klien berisiko), dengan trade-off precision yang lebih rendah — keputusan yang dipertimbangkan secara sadar berdasarkan konteks bisnis, bukan default metrik.

## Dataset

Sumber: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

Project ini menggunakan 2 dari 9 tabel yang tersedia:
- `application_train.csv` — 307,511 aplikasi pinjaman, 122 kolom fitur
- `bureau.csv` — 1,716,428 baris riwayat kredit klien di lembaga keuangan lain, diagregasi menjadi fitur per klien

Target: `TARGET` (biner) — 1 = klien mengalami kesulitan pembayaran signifikan, 0 = lancar. Imbalance ekstrem: **~92% lancar vs 8% default**.

## Pipeline

### 1. Feature Engineering Multi-Tabel
`bureau.csv` (banyak baris per klien) diagregasi menjadi 1 baris per `SK_ID_CURR`:
- Total sisa utang & total tunggakan (`sum`)
- Rentang waktu riwayat kredit (`min`/`max`/`mean` dari `DAYS_CREDIT`)
- Total & maksimum keterlambatan bayar (`CREDIT_DAY_OVERDUE`)

Hasil agregasi di-merge ke tabel utama (`left join`), dengan flag `HAS_BUREAU_HISTORY` untuk ~44,020 klien yang tidak punya riwayat bureau sama sekali.

### 2. Penanganan Data Quality Issues
- **`DAYS_EMPLOYED` anomaly**: nilai sentinel `365243` (≈1000 tahun) — placeholder untuk klien tidak bekerja, ditemukan overlap 100% dengan `ORGANIZATION_TYPE == 'XNA'`. Di-flag (`DAYS_EMPLOYED_ANOM`) lalu direplace jadi NaN.
- **Kolom properti missing tinggi (60-70%)**: 47 kolom terkait kondisi bangunan/apartemen di-drop, dengan pengecualian sadar terhadap `EXT_SOURCE_1` dan `OWN_CAR_AGE` yang tetap dipertahankan meski missing tinggi, karena punya nilai prediktif/semantik tersendiri.
- **Missing value pada kolom numerik lain**: dibiarkan NaN, diserahkan ke native XGBoost handling (`enable_categorical=True`).

### 3. Modeling
- XGBoost native dengan 12 kolom `category` dtype langsung (tanpa one-hot encoding manual)
- `scale_pos_weight ≈ 11.4` untuk menangani imbalance ekstrem
- Metrik utama: **PR-AUC** (bukan accuracy, yang menyesatkan untuk data imbalanced)

### 4. Threshold Tuning
Threshold klasifikasi dipilih dari analisis `precision_recall_curve`, bukan default 0.5 — dipilih **0.43** sebagai kompromi antara recall (~0.68) dan precision (~0.16), mengingat prioritas bisnis pada recall.

### 5. Interpretability (SHAP)
`TreeExplainer` diterapkan pada model final untuk memahami driver prediksi:
- **`EXT_SOURCE_1/2/3`** (skor risiko eksternal) adalah prediktor terkuat, dengan hubungan **linear** terhadap risiko — konsisten di semua rentang, tidak berinteraksi kuat dengan besar pinjaman.
- **`AMT_CREDIT`** menunjukkan hubungan **non-linear dengan titik jenuh** — risiko turun tajam pada pinjaman kecil, lalu mendatar di pinjaman besar. Warna interaksi dengan `AMT_INCOME_TOTAL` menunjukkan model secara implisit menangkap pola *debt-to-income* meski rasio itu belum dibuat sebagai fitur eksplisit.

## Hasil

| Setup | Recall (kelas 1) | Precision (kelas 1) | PR-AUC |
|---|---|---|---|
| Baseline (threshold 0.5, tanpa weighting) | 0.04 | 0.46 | 0.233 |
| + `scale_pos_weight` (threshold 0.5) | 0.60 | 0.18 | 0.232 |
| + `scale_pos_weight` + threshold 0.43 | **0.68** | 0.16 | 0.232 |

**Catatan penting**: PR-AUC nyaris tidak berubah antar setup — karena `scale_pos_weight` dan threshold tuning **tidak mengubah kemampuan ranking model**, hanya menggeser titik potong klasifikasi. Ini menegaskan bahwa peningkatan performa lebih lanjut membutuhkan **perbaikan model** (fitur baru, tuning), bukan sekadar penyesuaian threshold.

## Keterbatasan & Langkah Lanjutan

Project ini adalah baseline yang solid, bukan model siap produksi. Untuk pemakaian nyata, hal-hal berikut perlu ditambahkan:

- **Kalibrasi probabilitas** — `scale_pos_weight` mendistorsi probabilitas mentah; perlu `CalibratedClassifierCV` agar skor bisa dipakai untuk perhitungan *expected loss* riil.
- **Threshold berbasis biaya bisnis riil** — bukan asumsi kualitatif, tapi dihitung dari estimasi kerugian aktual per jenis kesalahan.
- **Feature engineering lanjutan** — tabel `previous_application.csv` dan `installments_payments.csv` (riwayat pembayaran cicilan) berpotensi menaikkan PR-AUC secara signifikan.
- **Fairness/bias check** — evaluasi apakah model secara tidak sengaja mendiskriminasi atribut sensitif (gender, usia), yang relevan secara regulasi di domain kredit.
- **Cross-validation / time-based split** untuk validasi yang lebih robust dibanding satu kali train-test split.

## Struktur File

```
├── predict.py           # OOP inference wrapper (preprocessing + prediksi)
├── requirements.txt      # Dependencies
├── README.md             # Dokumen ini
└── home_credit_xgb_model.joblib  # Model terlatih (tidak disertakan di repo, generate sendiri)
```

## Cara Pakai

```python
from predict import HomeCreditPredictor

predictor = HomeCreditPredictor(
    model_path="home_credit_xgb_model.joblib",
    threshold=0.43
)

results = predictor.predict(application_df, bureau_df)
```
