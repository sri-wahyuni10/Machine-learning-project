"""
PROJECT LEVEL 2: Klasifikasi dengan data campuran (numerik + kategorikal)
menggunakan Pipeline & ColumnTransformer

Dataset yang disarankan: Telco Customer Churn (Kaggle)
Ganti path CSV di bawah sesuai file yang sudah kamu unduh.

BEDA UTAMA dengan project diabetes_ml.py sebelumnya:
- Sebelumnya: semua kolom numerik, imputasi & scaling ditulis manual per kolom.
- Sekarang: kolom numerik DAN kategorikal, treatment beda tiap tipe,
  semuanya dibungkus jadi SATU objek Pipeline -> lebih aman dari leakage
  (karena ikut di-refit di setiap fold CV) dan lebih gampang dipakai ulang
  saat deployment (cukup load 1 file, bukan model+scaler+encoder terpisah).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, confusion_matrix
)

# ======================================================================
# 1. LOAD DATA
# ======================================================================
data = pd.read_csv('telco_churn.csv')  # ganti sesuai nama file kamu
print(data.shape)
print(data.dtypes)
# TIP: perhatikan kolom 'TotalCharges' -> biasanya ke-load sebagai object
# (teks), padahal isinya angka. Ini SINYAL ada string kosong/tersembunyi
# di dalamnya, sama seperti nilai 0 di project diabetes kemarin, tapi
# bentuknya beda (string, bukan angka).

# ======================================================================
# 2. CARI MISSING VALUE TERSEMBUNYI (khusus data campuran)
# ======================================================================
# Kolom kategorikal/object yang harusnya numerik -> paksa jadi numerik,
# apapun yang gagal dikonversi (string kosong, spasi, dll) otomatis jadi NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(f"Missing di TotalCharges setelah konversi: {data['TotalCharges'].isnull().sum()}")

# cek juga apakah ada kolom kategorikal yang punya kategori aneh
# semacam 'No internet service' yang sebenarnya representasi lain dari 'No'
# -> ini domain knowledge, perlu dicek manual per kolom, sama prinsipnya
# dengan 'pregnancies' yang tidak boleh disamaratakan di project kemarin.
for col in data.select_dtypes(include='object').columns:
    print(col, '->', data[col].unique()[:6])

# ======================================================================
# 3. DATA CLEANING EKSPLISIT (dilakukan SEBELUM masuk Pipeline)
# ======================================================================
# Kenapa dilakukan manual di sini, bukan diserahkan ke Pipeline?
# - SimpleImputer di Pipeline nanti HANYA menangani NaN (missing value).
#   "No internet service" / "No phone service" BUKAN NaN -> string valid,
#   jadi imputer tidak akan menyentuhnya sama sekali.
# - Ini soal REDUNDANSI kategori, bukan soal missing value. Kolom seperti
#   OnlineSecurity, TechSupport, StreamingTV dst otomatis bernilai
#   "No internet service" kalau (dan hanya kalau) InternetService == 'No'.
#   Informasinya sudah terwakili di kolom InternetService -> kita
#   sederhanakan jadi "No" saja supaya tidak ada kategori yang redundan.

data = data.drop(columns=['customerID'])  # ID tidak informatif untuk model
# 'Unnamed: 0' biasanya sisa index pandas yang ikut tersimpan saat CSV
# sebelumnya di-export tanpa index=False -> bukan data asli, harus dibuang
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

print("--- Cek missing value eksplisit (SEBELUM masuk Pipeline) ---")
print(data.isnull().sum()[data.isnull().sum() > 0])
# TIP: kalau kamu mau tahu berapa % baris yang hilang di TotalCharges,
# di sinilah tempat yang tepat untuk print/plot-nya, seperti project kemarin.

print("\n--- Cek nilai unik tiap kolom kategorikal (cari yang 'redundan') ---")
for col in data.select_dtypes(include='object').columns:
    print(col, '->', data[col].unique())

# Kolom-kolom yang punya nilai "No internet service" / "No phone service"
cols_to_simplify = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
]
for col in cols_to_simplify:
    data[col] = data[col].replace(
        {'No internet service': 'No', 'No phone service': 'No'}
    )

# PENTING: ternyata pola True/False vs Yes/No (yang kita temukan di kolom
# Churn) ada juga di kolom kategorikal lain (misal StreamingMovies punya
# 'No' DAN 'False' sebagai kategori terpisah -> padahal maknanya sama).
# Bersihkan SECARA SISTEMATIS ke semua kolom object, bukan cuma yang
# kita duga -> ini pelajaran penting: jangan asumsikan cuma 1 kolom yang
# bermasalah kalau sumber datanya sama.
# Perluas mapping lagi: SeniorCitizen ternyata pakai '0'/'1' (string)
# bercampur dengan 'No'/'Yes' -> representasi campuran ketiga yang kita
# temukan di dataset ini (setelah True/False, kini 0/1). Pola ini cukup
# umum di dataset dunia nyata yang datanya digabung dari beberapa sumber.
bool_like_map = {
    'True': 'Yes', 'False': 'No', 'true': 'Yes', 'false': 'No',
    '1': 'Yes', '0': 'No',
}
for col in data.select_dtypes(include='object').columns:
    if col == 'Churn':
        continue  # sudah ditangani terpisah di tahap 4 (perlu di-map ke 0/1)
    data[col] = data[col].replace(bool_like_map)

print("\n--- Verifikasi: cek ulang semua kolom kategorikal setelah dibersihkan ---")
for col in data.select_dtypes(include='object').columns:
    print(col, '->', data[col].unique())

print("\n--- Setelah disederhanakan ---")
for col in cols_to_simplify:
    print(col, '->', data[col].unique())

# ======================================================================
# 4. SPLIT FITUR & TARGET, LALU TRAIN/TEST SPLIT
# ======================================================================
# PENTING: split dilakukan SEBELUM preprocessing dibuat/fit -> prinsip
# yang sama seperti project sebelumnya (cegah leakage). Bedanya, di sini
# kita tidak perlu manual pisah imputasi train/test satu-satu -> Pipeline
# yang akan menjamin itu otomatis, karena .fit() Pipeline HANYA dipanggil
# di data train (lihat langkah 8).
print("--- Cek nilai unik kolom target SEBELUM mapping ---")
print(data['Churn'].unique())
print("Jumlah NaN di Churn (asli):", data['Churn'].isnull().sum())

# .str.strip() membersihkan spasi tersembunyi, .str.capitalize() menyamakan
# 'yes'/'YES'/'Yes' jadi bentuk yang sama sebelum di-map
data['Churn'] = data['Churn'].astype(str).str.strip().str.capitalize()

X = data.drop(columns=['Churn'])
# Dataset ini ternyata campuran representasi: ada 'Yes'/'No' DAN 'True'/'False'
# -> mapping harus mencakup keduanya, bukan cuma Yes/No
churn_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
y = data['Churn'].map(churn_map)

# Safety check: kalau setelah dibersihkan masih ada NaN, artinya ada
# baris dengan nilai target yang benar-benar bukan Yes/No -> harus
# diputuskan manual (drop baris itu, atau cek lebih lanjut kenapa).
print("Jumlah NaN di y SETELAH mapping:", y.isnull().sum())
if y.isnull().sum() > 0:
    print("Baris bermasalah:")
    print(data.loc[y.isnull()])
    # contoh penanganan: buang baris yang targetnya tidak jelas
    mask = y.notnull()
    X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y tetap dipakai karena churn biasanya imbalance (~27% churn)

# ======================================================================
# 5. PISAHKAN NAMA KOLOM: NUMERIK vs KATEGORIKAL
# ======================================================================
# Ini langkah KUNCI yang tidak ada di project sebelumnya -> ColumnTransformer
# butuh tahu kolom mana diperlakukan dengan pipeline mana.
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print("Kolom numerik:", numeric_cols)
print("Kolom kategorikal:", categorical_cols)

# ======================================================================
# 6. BANGUN PIPELINE KECIL PER TIPE KOLOM
# ======================================================================
# Pipeline numerik: isi NaN dengan median, lalu scaling
# (median dipilih karena tahan outlier, sama alasannya dengan project kemarin)
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline kategorikal: isi NaN dengan kategori paling sering muncul,
# lalu ubah jadi angka lewat One-Hot Encoding
# handle_unknown='ignore' -> supaya kalau di data BARU (production) muncul
# kategori yang tidak pernah dilihat saat training, model tidak error,
# cukup dianggap semua kategori = 0 di baris itu.
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # drop='if_binary' -> buang 1 kolom HANYA untuk fitur dengan tepat 2
    # kategori (misal StreamingMovies: Yes/No), supaya tidak ada dummy
    # variable trap (2 kolom yang saling menentukan sepenuhnya, bikin
    # koefisien model jadi terpecah/tidak stabil seperti yang kita lihat
    # di top 15 sebelumnya). Kolom dengan 3+ kategori (Contract, dst)
    # tetap direpresentasikan lengkap.
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
])

# ======================================================================
# 7. GABUNGKAN LEWAT COLUMNTRANSFORMER
# ======================================================================
# ColumnTransformer menjalankan numeric_pipeline HANYA ke numeric_cols,
# dan categorical_pipeline HANYA ke categorical_cols, secara paralel,
# lalu hasilnya digabung otomatis jadi satu matriks fitur.
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
], sparse_threshold=0)  # paksa output dense -> dibutuhkan GradientBoostingClassifier
# (RandomForest & LogisticRegression sebenarnya tetap jalan dengan sparse,
# tapi kita samakan supaya perbandingan antar model konsisten)

# ======================================================================
# 8. BUNGKUS PREPROCESSING + MODEL JADI SATU PIPELINE UTUH
# ======================================================================
# Inilah objek yang akan di-fit, di-cross-validate, dan di-tuning.
# Saat .fit(X_train, y_train) dipanggil: preprocessor belajar (imputer,
# scaler, encoder) HANYA dari X_train -> otomatis bebas leakage tanpa
# perlu kode manual copy-paste seperti project sebelumnya.
# LogisticRegression dipilih sebagai model utama karena menang di
# perbandingan tahap 9b -- performa terbaik DAN paling mudah diinterpretasi
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# ======================================================================
# 9. CROSS-VALIDATION DI ATAS PIPELINE
# ======================================================================
# Karena preprocessing ikut di dalam pipeline, tiap fold CV akan
# me-refit imputer/scaler/encoder HANYA dari fold training-nya sendiri.
# Ini LEBIH KETAT mencegah leakage dibanding pendekatan manual sebelumnya,
# yang scaler-nya cuma di-fit sekali di X_train besar (di luar loop CV).
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC (RandomForest baseline): mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

# ======================================================================
# 9b. BANDINGKAN BEBERAPA CLASSIFIER (preprocessing tetap sama untuk semua)
# ======================================================================
# Karena preprocessing sudah dibungkus di 'preprocessor', membandingkan
# model di sini cuma soal ganti step 'classifier' -> jauh lebih rapi
# dibanding project sebelumnya yang harus ulang manual tiap ganti model.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

candidate_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
}

print("\n--- Perbandingan model (CV ROC-AUC, setting default) ---")
comparison_results = {}
for name, clf in candidate_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    comparison_results[name] = scores
    print(f"{name:<20} mean={scores.mean():.4f}  std={scores.std():.4f}")

# CATATAN: LogisticRegression diuntungkan oleh StandardScaler yang sudah
# ada di numeric_pipeline -- tanpa scaling, LogisticRegression biasanya
# performanya jauh lebih buruk karena sensitif terhadap skala fitur.
# RandomForest & GradientBoosting sebenarnya tidak butuh scaling sama
# sekali (tree-based model tidak sensitif skala), tapi tidak masalah
# juga kalau tetap di-scale -- tidak merugikan performanya.

# Model dengan mean CV ROC-AUC tertinggi dari perbandingan di atas yang
# akan kita lanjutkan ke GridSearchCV (tahap 10) untuk tuning lebih detail.
# Ganti 'RandomForestClassifier()' di model_pipeline (tahap 8) kalau
# ternyata model lain menang di perbandingan ini.

# ======================================================================
# 10. HYPERPARAMETER TUNING (GridSearchCV di atas Pipeline)
# ======================================================================
# Perhatikan format nama parameter: 'namastep__parameter'
# 'classifier__n_estimators' -> parameter n_estimators di step 'classifier'
# Kita juga bisa tuning parameter PREPROCESSING, misal strategi imputasi:
# 'preprocessor__num__imputer__strategy' (karena nested di ColumnTransformer)
# Parameter LogisticRegression yang paling berpengaruh:
# - C: kebalikan dari kekuatan regularisasi. C KECIL = regularisasi KUAT
#   (model lebih "menahan diri", cegah overfitting). C BESAR = regularisasi
#   lemah (model lebih bebas mengikuti data training).
# - penalty: jenis regularisasi. 'l2' standar. 'l1' bisa membuat sebagian
#   koefisien jadi persis 0 (otomatis "membuang" fitur yang tidak penting).
# - solver 'liblinear' dipakai karena mendukung baik l1 maupun l2.
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
print('Best CV ROC-AUC:', round(grid_search.best_score_, 4))

best_pipeline = grid_search.best_estimator_

# ======================================================================
# 11. EVALUASI DI TEST SET
# ======================================================================
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print('=== Evaluasi di Test Set ===')
print('Accuracy:', round(accuracy_score(y_test, y_pred), 3))
print('ROC-AUC :', round(roc_auc_score(y_test, y_proba), 3))
print()
print(classification_report(y_test, y_pred, target_names=['Tidak Churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Churn', 'Churn'],
            yticklabels=['Tidak Churn', 'Churn'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix - Best Pipeline')
plt.tight_layout()
plt.savefig('confusion_matrix_churn.png')

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'RF Pipeline (AUC={roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_churn.png')
# ======================================================================
# 12. FEATURE IMPORTANCE (perlu ambil nama fitur HASIL transformasi)
# ======================================================================
# Beda dengan project sebelumnya: OneHotEncoder memecah 1 kolom kategorikal
# jadi beberapa kolom (misal 'Contract' -> 'Contract_Month-to-month',
# 'Contract_One year', dst). Nama fitur hasil transformasi harus diambil
# dari preprocessor, tidak bisa langsung pakai nama kolom asli.
feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
# LogisticRegression pakai .coef_ (bukan .feature_importances_ seperti tree-based)
# coef_ berbentuk 2D (1, n_fitur) untuk klasifikasi biner -> ambil baris ke-0
coefficients = best_pipeline.named_steps['classifier'].coef_[0]

feat_importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})
# Urutkan berdasarkan MAGNITUDE (besar kecilnya pengaruh), tapi tampilkan
# tanda aslinya -> koefisien POSITIF artinya menaikkan odds churn,
# NEGATIF artinya menurunkan odds churn. Ini beda dengan feature_importances_
# tree-based yang cuma angka positif tanpa info arah pengaruh.
feat_importance_df['abs_coefficient'] = feat_importance_df['coefficient'].abs()
feat_importance_df = feat_importance_df.sort_values('abs_coefficient', ascending=False)

print("--- Top 15 fitur paling berpengaruh (+ = naikkan odds churn, - = turunkan) ---")
print(feat_importance_df[['feature', 'coefficient']].head(15))

plt.figure(figsize=(8, 6))
top15 = feat_importance_df.head(15)
colors = ['#d62728' if c > 0 else '#2ca02c' for c in top15['coefficient']]
sns.barplot(data=top15, x='coefficient', y='feature', palette=colors)
plt.title('Top 15 Koefisien LogisticRegression (merah=naikkan churn, hijau=turunkan)')
plt.tight_layout()
plt.savefig('feature_importance_churn.png')

# ======================================================================
# 13. PERBAIKAN RECALL - PENDEKATAN 1: class_weight='balanced'
# ======================================================================
# Memaksa model memberi bobot lebih besar ke kesalahan pada kelas minoritas
# (Churn) saat training. Bandingkan CV ROC-AUC dan classification report-nya
# dengan model sebelumnya (tanpa class_weight) untuk lihat dampaknya.
model_pipeline_balanced = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=grid_search.best_params_['classifier__C'],
        penalty=grid_search.best_params_['classifier__penalty'],
        solver='liblinear',
    ))
])
model_pipeline_balanced.fit(X_train, y_train)

y_pred_balanced = model_pipeline_balanced.predict(X_test)
y_proba_balanced = model_pipeline_balanced.predict_proba(X_test)[:, 1]

print("\n=== Evaluasi dengan class_weight='balanced' ===")
print('ROC-AUC:', round(roc_auc_score(y_test, y_proba_balanced), 3))
print(classification_report(y_test, y_pred_balanced, target_names=['Tidak Churn', 'Churn']))
# Perhatikan: ROC-AUC biasanya hampir sama, tapi recall Churn di
# classification report harusnya naik dibanding model sebelumnya.
# Trade-off: precision Churn kemungkinan turun.

# ======================================================================
# 14. PERBAIKAN RECALL - PENDEKATAN 2: THRESHOLD TUNING
# ======================================================================
# Threshold default predict() adalah 0.5. Kita coba beberapa threshold
# dan lihat trade-off precision vs recall secara eksplisit, supaya
# keputusan threshold mana yang dipakai didasari data, bukan tebakan.
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n--- Trade-off precision/recall di berbagai threshold ---")
print(f"{'Threshold':<10}{'Precision':<12}{'Recall':<10}{'F1':<10}")
for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    p = precision_score(y_test, y_pred_thresh)
    r = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"{threshold:<10}{p:<12.3f}{r:<10.3f}{f1:<10.3f}")

# CATATAN PENTING: threshold "terbaik" TIDAK selalu yang F1 tertinggi.
# Ini keputusan bisnis: kalau biaya kehilangan pelanggan (false negative)
# jauh lebih mahal daripada biaya kirim promo ke orang yang salah sasaran
# (false positive), pilih threshold yang recall-nya lebih tinggi meski
# precision turun -- misal threshold 0.35 atau 0.4, bukan otomatis 0.5.


# ======================================================================
# 15. SIMPAN MODEL
# ======================================================================
# Sebelumnya (project diabetes): joblib.dump(model, ...) DAN
# joblib.dump(scaler, ...) terpisah.
# Sekarang: preprocessing + model jadi satu objek -> cukup 1 file.
# Saat deployment nanti, cukup load file ini dan langsung .predict(data_baru_mentah),
# TIDAK perlu scaling/encoding manual lagi karena semua sudah menyatu di pipeline.
joblib.dump(best_pipeline, 'model_churn_pipeline.pkl')
print("Pipeline (preprocessing + model) berhasil disimpan dalam 1 file.")

# Kalau setelah membandingkan hasil di tahap 13-14 kamu memutuskan
# class_weight='balanced' atau threshold custom lebih sesuai kebutuhan
# bisnis, simpan juga versi itu (atau ganti yang di atas):
# joblib.dump(model_pipeline_balanced, 'model_churn_pipeline_balanced.pkl')

# ======================================================================
# LANGKAH LANJUTAN (level berikutnya lagi, kalau semua di atas sudah lancar):
# - Bandingkan beberapa classifier (LogisticRegression, GradientBoosting)
#   di dalam pipeline yang sama, seperti loop model di project sebelumnya
# - SMOTE lewat imblearn.pipeline.Pipeline (bukan sklearn biasa) sebagai
#   alternatif class_weight untuk menangani imbalance
# - Deployment: bungkus best_pipeline.pkl jadi endpoint API sederhana
#   pakai FastAPI, terima input JSON mentah, langsung .predict()
# ======================================================================