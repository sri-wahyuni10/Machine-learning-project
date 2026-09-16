# Prediksi Income (Adult Census Income) — XGBoost + SHAP Interaction Values

Model klasifikasi biner untuk memprediksi apakah pendapatan seseorang di atas
atau di bawah $50K/tahun, menggunakan data sensus AS (UCI Adult Income
Dataset). Project ini fokus pada teknik interpretability tingkat lanjut:
**SHAP interaction values**, bukan sekadar feature importance/summary plot dasar.

## Ringkasan Model

| Item | Detail |
|---|---|
| Algoritma | XGBoost (native categorical support) |
| Target | `income` (0 = `<=50K`, 1 = `>50K`) |
| Fitur | 16 (numerik + kategorikal + 3 flag missingness) |
| Tuning | RandomizedSearchCV (20 iterasi, 3-fold CV) + early stopping |
| Metric utama | PR-AUC (data imbalanced ~76% / 24%) |

### Hasil Evaluasi (Test Set)

| Model | Precision (>50K) | Recall (>50K) | F1 (>50K) | PR-AUC |
|---|---|---|---|---|
| Baseline (default params) | 0.77 | 0.67 | 0.72 | 0.830 |
| Setelah tuning + `scale_pos_weight` | 0.61 | 0.88 | 0.72 | 0.835 |

**Catatan trade-off:** model hasil tuning jauh lebih agresif menangkap kasus
`>50K` (recall naik dari 67% → 88%) dengan konsekuensi precision turun (77% →
61%). F1-score kebetulan identik di kedua model, tapi menyembunyikan
trade-off precision-recall yang signifikan. Model mana yang lebih cocok
tergantung use-case: recall tinggi cocok untuk targeting/marketing,
precision tinggi lebih aman untuk keputusan berisiko tinggi seperti approval
kredit.

## Data & Preprocessing

- **Sumber:** [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult)
- Dataset tidak memiliki header asli — kolom ditambahkan manual mengikuti
  dokumentasi UCI.
- **Missing value:** direpresentasikan sebagai `'?'` pada kolom `workclass`,
  `occupation`, `native-country`. Diganti menjadi `np.nan` (bukan diimputasi),
  memanfaatkan native missing-value handling XGBoost, plus dibuatkan kolom
  indikator missingness (`*_missing`) untuk menangkap potensi sinyal
  informatif dari missingness itu sendiri.
- **Redundansi fitur:** kolom `education` di-drop karena punya hubungan 1-ke-1
  dengan `education-num` (multicollinearity sempurna).
- **Rare category grouping:** negara dengan kemunculan < 100 baris di
  `native-country` digabung menjadi kategori `Other`.
- **Encoding:** menggunakan native categorical support XGBoost
  (`enable_categorical=True`), bukan One-Hot Encoding — kolom kategorikal
  dikonversi ke tipe `category` pandas.

## Interpretability (SHAP)

### Fitur paling berpengaruh (Summary Plot)
Urutan importance: `marital-status`, `age`, `relationship`, `capital-gain`,
`education-num`, `occupation`, `hours-per-week`, `capital-loss`.

**Temuan menarik:** `capital-gain` sangat sparse (>90% nol) sehingga korelasi
Pearson-nya dengan fitur lain rendah (~0.08), tapi ternyata merupakan salah
satu prediktor terkuat secara SHAP — pola non-linear seperti ini tidak
terlihat dari korelasi linear biasa.

### Dependence Plot: Age
Hubungan `age` terhadap prediksi berbentuk **non-monoton**: SHAP value naik
tajam dari usia 17–45, plateau di usia 45–60, lalu menurun kembali di usia
60+. Pola ini tidak terlihat dari boxplot EDA sederhana yang hanya
menunjukkan median usia lebih tinggi pada kelompok `>50K`.

### SHAP Interaction Values
Dihitung menggunakan native XGBoost (`pred_interactions=True`) pada sample
500 baris test set, karena kompleksitas komputasi meningkat secara kuadratik
terhadap jumlah fitur.

**Temuan kunci — interaksi `age` × `marital-status`:** pada usia yang sama,
orang dengan status `Married-civ-spouse` memiliki SHAP value yang jauh lebih
positif dibanding `Never-married`. Ini mengindikasikan `marital-status` bukan
sekadar proxy independen untuk usia (seperti dugaan awal di EDA), melainkan
berinteraksi secara nyata dengan usia dalam menentukan prediksi income.

## Cara Menjalankan Inference

```bash
pip install -r requirements.txt

python predict.py --input data_baru.csv --output hasil_prediksi.csv

# Jika file input tanpa header (format asli UCI):
python predict.py --input adult.test --output hasil_prediksi.csv --no-header
```

## Struktur File

```
├── model.pkl                  # Model XGBoost terlatih (joblib)
├── category_mapping.json      # Daftar kategori tiap kolom (untuk konsistensi inference)
├── predict.py                 # Script inference
├── requirements.txt
└── README.md
```

## Batasan & Catatan Analisis

- Dataset sensus AS tahun 1994 — pola sosioekonomi mungkin tidak
  representatif untuk konteks saat ini atau negara lain.
- Korelasi antara `marital-status` dan `income` **bukan hubungan kausal**.
  Kemungkinan besar keduanya sama-sama berkorelasi dengan tahap karier/usia.
- Kolom `capital-gain` memiliki nilai top-coded (dibatasi ~99999) untuk
  alasan privasi sensus, menyebabkan cluster tidak wajar di nilai tersebut.
