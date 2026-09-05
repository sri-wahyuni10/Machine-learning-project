# Diabetes Hospital Readmission Prediction
Project ini sekarang sudah selesai secara teknis end-to-end. Rekap besar pencapaian:

EDA + deteksi missing value tersembunyi
Domain-informed cleaning
Group-based split (cegah leakage pasien)
ICD-9 grouping manual (692 → ~10 kategori medis)
Pipeline + ColumnTransformer
Model comparison dengan reasoning berbasis domain, bukan cuma accuracy
Threshold tuning kustom
Custom OOP wrapper (ThresholdClassifier) untuk joblib
predict.py yang terverifikasi end-to-end dengan data baru

Prediksi risiko readmisi pasien diabetes ke rumah sakit (dalam <30 hari, >30 hari, atau tidak readmit) menggunakan dataset **Diabetes 130-US Hospitals (1999-2008)** dari UCI ML Repository.

Project ini adalah lanjutan dari rangkaian project ML sebelumnya (Health Condition Classifier → Telco Customer Churn → Chronic Kidney Disease), dengan level kompleksitas yang lebih tinggi: multi-class classification, dataset besar (~100rb baris, 50 kolom), missing value tersembunyi, dan group-based data leakage.

## Ringkasan Masalah

Rumah sakit ingin mengidentifikasi pasien diabetes yang berisiko readmit dalam **kurang dari 30 hari** setelah pulang — indikator umum kualitas perawatan yang dipantau ketat oleh regulator kesehatan AS. Model ini membantu tenaga medis memberi perhatian ekstra ke pasien berisiko tinggi sebelum mereka dipulangkan.

**Target (`readmitted`)**:
| Label | Kode | Deskripsi |
|---|---|---|
| `NO` | 0 | Tidak readmit |
| `>30` | 1 | Readmit setelah 30 hari |
| `<30` | 2 | Readmit dalam <30 hari (**kelas kritis**) |

## Tantangan Teknis Utama

1. **Missing value tersembunyi** — banyak kolom (`payer_code`, `race`, `medical_specialty`, dll) menggunakan `"?"` sebagai penanda missing, bukan `NaN` asli, sehingga tidak terdeteksi oleh `.isnull()` secara default.
2. **Group leakage** — satu pasien (`patient_nbr`) bisa punya banyak baris (multiple encounters). Split train/test dilakukan dengan `GroupShuffleSplit` agar seluruh baris satu pasien tetap berada di satu sisi saja, mencegah model "menghafal" identitas pasien.
3. **High-cardinality feature** — kolom diagnosis (`diag_1`, `diag_2`, `diag_3`) memiliki 700+ kode ICD-9 unik. Alih-alih one-hot encoding mentah (yang akan meledakkan dimensi fitur), kode-kode ini dikelompokkan secara manual ke ~10 kategori medis besar (Diabetes, Circulatory, Respiratory, dll) berdasarkan rentang resmi ICD-9.
4. **Class imbalance signifikan** — kelas paling kritis (`<30`) hanya ~11% dari data, sementara kelas mayoritas (`NO`) mencapai ~54%.

## Pendekatan & Keputusan Desain

### Preprocessing
- Kolom `weight` (>96% missing) di-drop.
- `max_glu_serum` dan `A1Cresult` (>80% missing) diperlakukan sebagai informasi klinis (missing = "tidak dites"), bukan sekadar data hilang.
- Kolom kategorikal dengan missing sedang (`payer_code`, `medical_specialty`, `race`) diisi `"Unknown"`.
- `Pipeline` + `ColumnTransformer` sklearn digunakan untuk menyatukan imputasi, scaling, dan one-hot encoding menjadi satu objek yang konsisten dipakai saat training maupun inference.

### Model Selection
Dua model dibandingkan: **Logistic Regression** dan **Random Forest**, keduanya dengan `class_weight='balanced'`.

| Model | Accuracy | Recall kelas `<30` |
|---|---|---|
| Logistic Regression | 0.50 | **0.40** |
| Random Forest | 0.58 | 0.01 |

**Logistic Regression dipilih** meskipun accuracy-nya lebih rendah — karena recall kelas kritis jauh lebih baik. Random Forest, meski akurasi keseluruhannya lebih tinggi, hampir sepenuhnya mengabaikan kelas `<30` (recall 0.01), yang secara medis jauh lebih berbahaya daripada accuracy yang terlihat bagus di atas kertas.

### Threshold Tuning
Alih-alih argmax standar, digunakan custom threshold: probabilitas kelas `<30` ≥ 0.3 langsung diklasifikasikan sebagai kelas tersebut. Ini menaikkan recall kelas `<30` dari **0.40 → 0.65**, dengan trade-off precision dan accuracy keseluruhan yang lebih rendah — pilihan yang disengaja karena false negative (pasien berisiko lolos tanpa terdeteksi) dianggap lebih mahal daripada false positive (perhatian ekstra yang ternyata tidak diperlukan).

### Production-Readiness
Pipeline dan threshold dibungkus dalam custom class `ThresholdClassifier`, sehingga satu file `.joblib` menyimpan seluruh logika (preprocessing, model, dan keputusan threshold) sebagai satu objek yang bisa langsung dipakai untuk inference tanpa risiko lupa mengulang salah satu langkah.

## Struktur File

```
├── diabetes_readmission.ipynb      # Notebook eksplorasi, cleaning, training
├── predict.py                      # Script inference untuk data pasien baru
├── readmission_diabetic_model.joblib  # Model tersimpan (pipeline + threshold)
├── requirements.txt
└── README.md
```

## Cara Menjalankan Inference

```python
import joblib
import pandas as pd

model = joblib.load('readmission_diabetic_model.joblib')
# data_baru harus melalui preprocessing manual yang sama seperti training
# (replace '?', drop weight, fillna 'Unknown', ICD-9 grouping) — lihat predict.py
prediction = model.predict(data_baru)
```

## Batasan & Pengembangan Lanjutan

- Precision kelas `<30` masih rendah (0.16) — banyak false positive. Threshold bisa dikalibrasi lebih lanjut sesuai kapasitas resource rumah sakit.
- Feature importance/koefisien belum dieksplorasi secara mendalam untuk validasi klinis.
- SMOTE atau teknik resampling lain belum dicoba sebagai pembanding `class_weight='balanced'`.
- Hyperparameter tuning belum dieksplorasi secara ekstensif (belum ditemukan indikasi kuat bahwa ini akan memberi gain signifikan).

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning Repository.
