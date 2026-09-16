# Catatan Proses — Synthetic Retail Dataset (Rossmann-style, Indonesia)

## Kenapa proyek ini beda dari proyek sebelumnya

Semua project sebelumnya (CKD, Cirrhosis, Home Credit, Rossmann asli) mulai dari dataset yang sudah ada — tugasnya cleaning dan feature engineering dari data yang sumbernya tidak aku tahu persis. Di proyek ini, aku bangun datanya sendiri dari nol, jadi aku tahu persis "kebenaran" di baliknya (ground truth) — ini dipakai untuk melatih diri memverifikasi apakah proses EDA, feature engineering, dan model beneran menangkap pola yang seharusnya, bukan cuma percaya begitu saja.

## Struktur data

- 30 toko × 912 hari (Jan 2022 - Jun 2024) = 27.360 baris
- `store.csv` (atribut statis toko), `calendar.csv` (kalender + Lebaran), `train.csv` (time series harian)
- Pakai konteks Indonesia: efek Lebaran/Ramadan sebagai pengganti Christmas di Rossmann asli — window belanja H-7 s/d H-1, toko tutup pas hari H

## Kronologi proses & masalah yang ditemukan

### 1. Generate data
- Cross join store × calendar dulu baru dihitung Sales — sempat bingung kenapa harus cross join, akhirnya ngerti: setiap toko butuh baris terpisah untuk tiap tanggal, karena Sales beda-beda per toko meski tanggal sama (struktur data panel).
- Formula Sales = BaseSales × StoreIndividualFactor × TrendFactor × DowFactor × PromoFactor × LebaranFactor + noise, ditambah anomali acak 0.3% dan Customers yang sengaja dibuat leakage.

### 2. Bug kecil waktu generate (self-inflicted, ketemu sendiri lewat error)
- `set_index('Store')['Store']` — salah, kolom `Store` sudah hilang setelah jadi index. Fix: ambil dari `store_df['Store']` langsung.

### 3. Missing value — MNAR terstruktur
- `CompetitionDistance` NaN untuk 3 dari 30 toko, tapi NaN-nya per TOKO (semua 912 baris toko itu NaN), bukan acak per baris — verifikasi pakai `groupby('Store').apply(lambda x: x.isna().all() or x.isna().sum()==0)`, semua True.
- `Promo2SinceWeek/Year/PromoInterval` NaN selalu bareng `Promo2==0` — awalnya aku pikir bisa di-drop, ternyata SALAH. NaN di sini bermakna ("toko tidak ikut promo2"), bukan data hilang yang perlu dibuang. Harus dijadikan flag, bukan drop kolom.

### 4. Bug pandas klasik (lagi) — operator `and`/`or` vs `&`/`|`
- Nulis `A or B` di kondisi Series → error "truth value of Series ambiguous". Harus `&`/`|` + setiap kondisi dikurung sendiri-sendiri, karena `&` presedensinya lebih tinggi dari `>`/`==`.
- Ketemu juga bug typo spasi di nama kolom (`'Promo2SinceYear '` dengan spasi) — bikin `KeyError` diam-diam kalau tidak dicek pakai `in df.columns`.

### 5. WeekOfYear vs Year — kenapa dua-duanya perlu
- Awalnya bingung, ternyata WeekOfYear itu "posisi minggu di dalam tahun", Year itu "tahun-nya" — minggu ke-20 tahun 2022 beda total sama minggu ke-20 tahun 2023, makanya harus dicek bareng buat logika PromoStarted.

### 6. Lag/rolling feature — urutan penting
- WAJIB hitung lag/rolling SEBELUM drop `Open==0`. Kalau dibalik, gap hari libur (misal 3 hari Lebaran tutup) hilang dari struktur data, rolling average jadi salah karena baris yang harusnya berjarak beberapa hari malah dianggap bersebelahan.
- `.shift(1)` sebelum `.rolling()` — supaya rolling mean tidak ikut menghitung Sales hari itu sendiri (kalau tidak, itu leakage).
- Verifikasi manual: cek baris index 8, `Sales_lag_7` harus sama dengan Sales di index 1 (8-7=1). Cocok.

### 7. BUG BESAR — IsLebaranShoppingWindow hilang dari train.csv
Ini temuan paling penting di seluruh proyek. Ceritanya:
- Model pertama (MAE 670.74, RMSE 1195.21) — cek residual terbesar, SEMUA di sekitar Lebaran 2024.
- Awalnya nebak ini soal model kurang bagus. Ternyata setelah dicek `feature_cols`, kolom `IsLebaranShoppingWindow` TIDAK ADA — padahal kolom ini yang dipakai buat generate Sales (LebaranFactor).
- Akar masalah: waktu bikin train.csv, kolom ini dipakai buat HITUNG Sales, tapi tidak ikut disimpan ke train_df final. Jadi informasi "kenapa Sales tinggi" hilang, padahal efeknya sudah kejadi di angka Sales.
- Ini SILENT BUG — kode jalan mulus tanpa error dari awal sampai akhir, cuma ketauan lewat analisis residual (cek baris dengan error terbesar).
- Fix: merge ulang dari calendar.csv (bukan generate ulang semua data dari nol) — pilih opsi yang lebih aman karena logika Lebaran sudah pernah divalidasi di calendar.csv sebelumnya.
- Hasil setelah fix: MAE 670.74 -> 466.33, RMSE 1195.21 -> 631.49. Improvement besar.

**Pelajaran paling penting:** kode yang jalan tanpa error BUKAN jaminan pipeline-nya benar. Feature yang dipakai untuk generate data harus benar-benar ditelusuri sampai ke feature final yang dipakai model — kalau ada yang "hilang" di tengah jalan, tidak akan ada pesan error apapun, cuma performa model yang aneh di tempat-tempat spesifik.

### 8. Sisa error setelah fix — irreducible error, BUKAN bug
- Setelah fix Lebaran, error terbesar berikutnya tersebar (Feb, Mar, Apr, Mei), tidak mengelompok di tanggal tertentu.
- Awalnya nebak ini kombinasi StoreType+DayOfWeek+Promo yang jarang muncul. Ternyata SALAH — dicek StoreType-nya malah didominasi 'a' (base sales PALING TINGGI), bukan 'd' (base sales rendah). Jadi bukan kombinasi faktor negatif legitimate.
- Kesimpulan yang benar: ini anomali acak (0.3%) yang sengaja disuntik pas generate data, TIDAK berkorelasi dengan kolom apapun. Model memang seharusnya tidak bisa prediksi ini akurat — irreducible error by design.
- Beda penting: bug Lebaran BISA diperbaiki (informasi hilang, tinggal ditambahkan lagi), anomali acak TIDAK BISA diperbaiki (memang tidak ada sinyal untuk dipelajari).

### 9. SHAP — validasi akhir
- Top feature: DayOfWeek, StoreType, Promo, Sales_lag_7, IsLebaranShoppingWindow — semua match dengan formula generate.
- Fitur yang SHAP-nya nyaris nol: IsCompetitorActive, IsPromo2ActiveOnThisDate, IsMonthInPromoInterval. Awalnya bingung kenapa, padahal logikanya udah dibangun cukup rumit (perbandingan tahun+minggu, cek interval).
- Ternyata jawabannya sederhana: atribut-atribut ini (CompetitionDistance, Promo2, PromoInterval) di store.csv memang TIDAK PERNAH benar-benar dipakai untuk menghitung Sales di formula generate. Jadi wajar SHAP-nya nol — bukan salah logika perhitungan flag-nya, tapi memang dari desain awal tidak ada hubungan sebab-akibat ke target.
- Sales_lag_7 penting padahal tidak ada di formula sama sekali — karena dia proxy: Sales 7 hari lalu kena pengaruh faktor yang sama (BaseSales, TrendFactor, dan DayOfWeek yang sama persis 7 hari lalu), jadi secara tidak langsung "membawa" info itu semua.

## Insight metodologis paling penting dari proyek ini

1. **Grafik bisa menipu** — StoreType B vs D kelihatan tumpang tindih di plot time series (karena noise harian besar), padahal rata-rata keduanya beda signifikan (4309 vs 3074). Harus selalu cross-check pakai angka agregat (groupby().mean()), jangan cuma percaya mata dari grafik mentah.
2. **SHAP bisa jadi alat audit jujur** — bukan cuma buat "menjelaskan" model, tapi buat ketauan mana fitur yang secara desain memang tidak informatif (meski logika pembuatannya rumit dan kelihatan penting secara teori).
3. **Analisis residual > percaya angka metrik mentah** — MAE/RMSE turun itu bagus, tapi CARA nemuin kenapa MAE tinggi (cek baris error terbesar satu-satu) itu yang paling berharga, karena dari situ ketemu bug nyata, bukan cuma "model kurang bagus".
4. **Data leakage bisa disuntik sengaja untuk latihan** — Customers dibuat leakage-prone dengan sengaja, supaya proses feature selection beneran melatih insting soal "kolom apa yang boleh dipakai" — bukan cuma teori doang.

## Status & langkah selanjutnya

- [x] Generate data sintetis (store, calendar, train)
- [x] Missing value handling (flag-based)
- [x] Feature engineering (IsPromo2ActiveOnThisDate, IsCompetitorActive, EffectiveCompetitionDistance)
- [x] EDA visual
- [x] Lag/rolling feature
- [x] Train/test split berbasis waktu
- [x] Model LightGBM pertama
- [x] Debug bug IsLebaranShoppingWindow
- [x] SHAP validation
- [x] predict.py + save_model_artifacts.py
- [ ] Dokumentasi final (README selesai, catatan ini)
- [ ] LinkedIn post draft
- [ ] (opsional) FastAPI deployment — perlu urus API key security dulu sebelum push ke GitHub