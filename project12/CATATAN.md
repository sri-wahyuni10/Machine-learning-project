# Catatan Proses — German Credit Risk Classification

Project ini dikerjakan sebagai sesi pendalaman konsep, mirip dengan project Bank Marketing sebelumnya (mixed types, MNAR, encoding), tapi dengan tambahan konsep baru: **cost-sensitive threshold tuning** pakai cost matrix asimetris.

## Kenapa dataset ini dipilih

Ingin dataset yang mirip Bank Marketing dari sisi tantangan teknis (mixed data types, MNAR pattern), tapi beda domain. German Credit (Statlog, UCI) cocok karena:
- Ada kode kategorikal tersembunyi yang bermakna MNAR (mirip `pdays=999`)
- Dataset kecil (1000 baris) → iterasi cepat
- Punya cost matrix resmi dari UCI → kesempatan belajar konsep baru (cost-sensitive learning) yang belum pernah dicoba di project-project sebelumnya

## File yang dipakai: `german.data` vs `german.data-numeric`

Awalnya bingung ada 2 file di folder dataset. Ternyata:
- `german.data` = versi mentah, kategori masih kode simbolik (A11, A12, dst) — ini yang dipakai, karena butuh latihan encoding decision sendiri
- `german.data-numeric` = sudah dikonversi ke numerik oleh pembuat dataset, kehilangan konteks kategori asli — kalau pakai ini nggak ada lagi yang bisa diinvestigasi

Pelajaran: kalau ada pilihan versi "sudah diproses" vs "mentah", pilih mentah kalau tujuannya belajar reasoning, bukan cuma benchmark model cepat.

## Bug pertama: nggak baca data dictionary

Sempat coba tebak arti kode kategori (`A11`, `A14`, dst) cuma dari pola angka di `value_counts()` — ini salah kaprah. Data dictionary itu wajib dicari/dipakai untuk dataset publik yang punya dokumentasi resmi (`german.doc`). Kalau dataset internal yang nggak ada dictionary-nya, itu kasus lain — tapi tetap harus tanya/cari sumber definisinya, jangan nebak dari angka.

## MNAR investigation — hasil yang di luar dugaan awal

Awalnya nebak asal (`purpose`, `housing`, `personal_status` sebagai kandidat MNAR) — ternyata salah arah. Setelah cek `value_counts()` satu-satu, MNAR yang beneran ada di:
- `checking_status` → kode A14 = "no checking account"
- `savings_status` → kode A65 = "unknown/no savings account"
- `property_magnitude` → kode A124 = "unknown/no property"

**Temuan paling menarik**: arah risiko "unknown/no ..." ini **nggak konsisten** antar kolom!
- `checking_status` & `savings_status`: kategori "no account" justru bad rate paling **rendah** (11.7% dan 17.5%) — kebalikan dari dugaan awal
- `property_magnitude`: kategori "unknown/no property" justru bad rate paling **tinggi** (43.5%)

Awalnya nebak "nggak punya rekening = nggak ada masalah, tinggal buka baru" — ternyata salah kerangka mikir. Yang penting buat bank bukan "gampang/susahnya buka rekening", tapi seberapa banyak *data* yang bank punya buat menilai risiko. Insight ini baru kelihatan setelah groupby-against-target, bukan dari nebak logika awam.

Pelajaran besar: jangan asumsikan semua kolom "unknown/missing" pasti bermakna sama (entah selalu risiko tinggi atau selalu risiko rendah) — harus dicek satu-satu ke data, karena bisa berlawanan arah bahkan dalam satu dataset yang sama.

## Ordinal vs Nominal — bingung dulu sebelum ngerti

Sempat betul-betul nggak tahu bedanya nominal dan ordinal. Penjelasan yang akhirnya nempel:
- **Nominal**: kategori sejajar, nggak ada urutan yang disepakati semua orang (contoh: `housing` — rent/own/for free, nggak ada yang "lebih tinggi")
- **Ordinal**: ada tingkatan jelas yang disepakati (contoh: `employment` — makin lama kerja, makin ke atas)

Cara cepat ngecek: coba urutkan dari kecil ke besar, kalau semua orang bakal setuju urutannya → ordinal. Kalau meragukan/debatable → nominal.

Ada 2 sumber urutan ordinal yang beda:
1. **Dari data** (bad rate hasil groupby) — dipakai untuk `checking_status`, `savings_status`, `property_magnitude`, karena urutan administratifnya (nominal DM) nggak selalu match sama urutan risiko aktual
2. **Dari domain logic langsung** — dipakai untuk `employment`, `credit_history`, `job`, karena urutannya sudah jelas secara alami tanpa perlu cek data dulu

Sempat salah urutan di `credit_history_order` dan `job_order` (ketuker posisi kategori terbaik dan terburuk) — typo logika, bukan typo penulisan. Perlu baca ulang pelan-pelan arti tiap kategori sebelum nyusun urutan, jangan asal tebak dari posisi kode aslinya (A30-A34 dst nggak selalu berarti urut dari bagus ke buruk secara otomatis, harus dibaca maknanya).

## Bug ColumnTransformer: transformer ketuker + lupa instantiate

Sempat nulis `OneHotEncoder` untuk kolom numerik dan `StandardScaler` untuk kolom nominal — ketuker total. Plus lupa kasih tanda kurung `()` (nulis `StandardScaler` bukan `StandardScaler()`), jadi masih merujuk ke class, bukan objek siap pakai.

## Konsep baru: cost-sensitive threshold tuning

Ini bagian paling baru dibanding project-project sebelumnya. Biasanya threshold tuning dipakai buat naikin recall kelas minoritas berdasar F1-score atau target recall tertentu. Di sini beda — German Credit punya **cost matrix resmi**: FN (bad diprediksi good) 5x lebih mahal dari FP (good diprediksi bad).

Rumus cost:
```python
total_cost = (FN * 5) + FP
```

Sweep threshold dari 0.05 sampai 0.55, cari yang cost-nya minimum → ketemu di **threshold = 0.15** (cost = 98, vs cost = 186 di threshold default 0.5).

Di threshold 0.15, hasilnya:
- Recall bad naik jadi 90% (nangkep 54 dari 60 nasabah bad)
- Tapi precision bad anjlok ke 44%, accuracy turun ke 63%

Pelajaran paling penting sesi ini: **accuracy/F1 tinggi bukan tujuan akhir** kalau cost error-nya nggak simetris. Threshold yang "kelihatan buruk" di metrik standar bisa jadi keputusan paling tepat secara bisnis. Ini beda paradigma dari threshold tuning yang biasa dilakukan di project-project sebelumnya (Diabetes 130-US, Bank Marketing) yang optimasi ke F1/recall target tanpa cost eksplisit.

## Bug inference: lupa X_test itu sudah readable

Waktu bikin `CreditRiskPredictor` class, sempat coba `predictor.predict(X_test.iloc[[0]])` langsung — error, karena `_to_readable()` di dalam class nyoba mapping ulang value yang **sudah** readable (`'< 0 DM'`) ke dictionary yang key-nya kode asli (`'A11'`). Nggak ketemu match, jadi NaN, validasi nangkep itu dan raise error.

Ini kejadian karena lupa: `X_test` diambil dari `df_readable`, bukan `df` asli. Solusinya ambil ulang baris yang sama tapi dari `df` (pakai `X_test.index` biar barisnya konsisten), baru dikasih ke predictor.

Pelajaran: pas bikin fungsi/class inference, harus jelas dari awal fungsi itu ngarepin input **format apa** (kode asli vs readable), dan konsisten dites pakai data yang benar-benar merepresentasikan bentuk input real-world (kode asli), bukan data yang udah "dicurangi" jadi readable duluan.

## Klarifikasi konsep: "data baru" itu ambigu

Sempat tanya "gimana cara cek akurasi model di data baru" — ternyata ini pertanyaan yang perlu diperjelas dulu maksudnya:
1. **Data yang belum dilihat model tapi ada labelnya** (`X_test`/`y_test`) → ini bisa dihitung akurasinya, dan itu yang sebenarnya sudah dikerjakan dari awal
2. **Data benar-benar baru dari dunia nyata tanpa label** (misal nasabah baru hari ini) → nggak bisa dihitung akurasinya karena belum tahu jawaban sebenarnya, cuma bisa lihat prediksi + probabilitas

Ini insight penting yang kadang missed: "testing di data baru" itu selalu berarti "data yang belum dilihat model", bukan berarti otomatis "data tanpa label yang bisa dihitung akurasinya".

## Yang belum dikerjakan (next steps)

- SHAP interpretability belum diterapkan ke dataset ini
- Belum dibandingkan dengan model tree-based (Random Forest/XGBoost) — kemungkinan bisa dapat cost lebih rendah
- `personal_status` yang menggabungkan status pernikahan + gender belum ditangani secara khusus (isu fairness, dicatat tapi belum diselesaikan)

## Refleksi belajar

Sesi ini paling banyak waktunya habis di reasoning ordinal encoding (urutan dari data vs domain logic) dan cost-sensitive threshold — dua konsep yang belum pernah dipegang detail di project-project sebelumnya walau permukaannya mirip (encoding, threshold tuning) sama yang udah pernah dikerjakan. Perlu diulang lagi di project berikutnya biar makin nempel, terutama logika "kenapa urutan kategori penting buat model linear" dan "kenapa metrik standar bisa menyesatkan kalau cost error nggak simetris".