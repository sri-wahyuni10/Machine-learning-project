# Catatan Proses — Bank Marketing Classification

Project ini sengaja diambil untuk memperdalam bagian yang paling sering bikin bingung dari project telco churn kemarin: **persiapan data sebelum masuk Pipeline & ColumnTransformer**. Fokusnya bukan cuma "dapat model yang bagus", tapi paham betul *kenapa* tiap keputusan preprocessing diambil.

## Dataset

Bank Marketing (UCI) — `bank-additional-full.csv`, 41.188 baris, 20 fitur, target `y` (subscribe term deposit atau tidak). Sempat kebingungan awal soal file mana yang harus dipakai (ada `bank.csv`, `bank-full.csv`, `bank-additional.csv`, `bank-additional-full.csv` — ternyata semua itu subset/versi dari dataset yang sama, bukan data terpisah). Juga sempat lupa `sep=';'` waktu load pertama kali, hasilnya semua kolom nyatu jadi satu — pelajaran: selalu cek `.head()` dulu sebelum lanjut.

## Proses Identifikasi Kolom

Ini bagian yang paling saya latih di project ini — sebelumnya suka bingung harus mulai dari mana. Polanya ternyata:

1. **Pisahkan dtype dulu** (`select_dtypes`), tapi jangan percaya begitu saja — perlu cek manual kolom mana yang numerik secara dtype tapi sebenarnya kategorikal secara makna
2. **Cek missing value** — dan di dataset ini missing-nya tidak muncul sebagai NaN sama sekali, tapi disamarkan sebagai string `"unknown"`. Ini beda dari kasus CKD/Cirrhosis yang biasanya NaN eksplisit.
3. **Untuk tiap kolom yang punya `"unknown"`, cek apakah itu MCAR (acak) atau MNAR (informatif)** — caranya dengan `groupby` kolom tersebut vs target `y`, lihat apakah rate subscribe-nya beda jauh antar grup.

Temuan menarik: kolom `default` (status kredit macet) punya 20.87% `"unknown"` — angka yang tinggi banget. Setelah di-groupby dengan `y`, ketahuan rate subscribe untuk `unknown` (5.15%) jauh lebih rendah dari yang jawab `no` (12.88%). Ini bukti kuat missing-nya **informatif**, jadi treatment-nya: biarkan `"unknown"` sebagai kategori sendiri, jangan diimputasi jadi `no` atau `yes`. Prinsip ini persis sama dengan yang sudah dipelajari di project-project sebelumnya soal MNAR — cuma sekarang konteksnya kategorikal, bukan numerik.

## Dua Kolom yang Butuh Perhatian Khusus

**`duration`** — ternyata ini data leakage klasik. Durasi telepon terakhir baru diketahui *setelah* telepon selesai, padahal model harusnya memprediksi *sebelum* menelepon. Dibuktikan lewat `groupby('y')['duration'].describe()`: rata-rata durasi untuk `y=yes` (553 detik) 2.5x lipat dari `y=no` (221 detik), bahkan nilai minimum untuk `yes` adalah 37 detik (tidak ada satupun yang subscribe dengan telepon super singkat). Keputusan: **drop**.

**`pdays`** — ada sentinel value `999` yang artinya "belum pernah dihubungi", bukan angka hari beneran. 96.32% data isinya 999. Kalau dipakai mentah sebagai numerik, scaling akan rusak karena didominasi satu nilai ekstrem. Solusinya: dibuat flag `was_contacted_before`, kolom asli di-drop.

## Bug yang Ditemui (untuk diingat)

1. **`StandardScaler` tanpa tanda kurung** — `TypeError: Expected an estimator instance, got estimator class instead`. Lupa nulis `()`, jadi yang dikasih ke Pipeline itu class-nya, bukan instance.
2. **Kolom hilang saat split** (`ValueError: Some column names are not columns of the dataframe`) — ternyata karena urutan eksekusi cell tidak berurutan. Ada typo juga (`load_info_missing` seharusnya `loan_info_missing`) dan flag `was_contacted_before` yang sempat tidak ke-create karena `pdays` keburu di-drop sebelum flag-nya dibuat. Pelajaran lama yang kepakai lagi: restart kernel + run all kalau state sudah campur aduk.
3. **Typo nama step** (`'clasifier'` bukan `'classifier'`) — tidak bikin error langsung, tapi akan menyulitkan kalau nanti akses lewat `named_steps`.

## Hasil Model

| Model | PR-AUC | F1 (yes) |
|---|---|---|
| Logistic Regression (threshold default 0.5) | 0.466 | 0.33 |
| Logistic Regression (threshold tuned 0.203) | 0.466 | 0.516 |
| Random Forest default (`n_estimators=200`) | 0.392 | 0.47 |
| Random Forest tuned (`max_depth=10`, `min_samples_leaf=20`) + threshold 0.696 | **0.491** | **0.528** |

Temuan paling penting: **Random Forest default ternyata kalah dari Logistic Regression** (PR-AUC 0.392 vs 0.466), padahal sudah pakai `class_weight='balanced'`. Awalnya saya kira tree-based otomatis lebih unggul, tapi ternyata itu karena RF default overfit (tidak ada batas `max_depth`). Setelah dikontrol (`max_depth=10`, `min_samples_leaf=20`), baru RF menang telak dari Logistic Regression.

Pelajaran besar dari sini: **jangan pernah bandingkan model dengan parameter default begitu saja** — terutama model yang rawan overfit seperti tree-based. Perbandingan yang adil butuh tuning dulu di masing-masing model.

## Soal Threshold Tuning

Threshold optimal (berdasarkan F1) untuk tiap model ternyata **beda jauh**:
- Logistic Regression: 0.203
- Random Forest: 0.696

Ini ngajarin saya bahwa skala probabilitas antar model tidak selalu bisa dibandingkan apa adanya — threshold harus dituning ulang untuk tiap model, tidak bisa pakai satu angka yang dianggap universal.

Juga dipahami ulang prinsip lama: **threshold tuning menggeser posisi di kurva PR yang sama** (precision naik, recall turun atau sebaliknya), sedangkan **mengganti/tuning model itu mengubah bentuk kurvanya sendiri**. Dua intervensi yang beda, dan project ini jadi contoh konkret keduanya dipakai berurutan.

## Yang Masih Bisa Dikembangkan (kalau lanjut lagi nanti)

- `RandomizedSearchCV` untuk RF, biar tuning-nya sistematis, bukan coba-coba manual
- SHAP untuk RF tuned — biar tahu fitur mana yang paling berpengaruh (bagus buat pemanasan sebelum masuk fase gradient boosting + SHAP di roadmap)
- Coba LightGBM untuk pembanding tambahan
- Pertimbangkan threshold berdasarkan biaya bisnis (biaya telepon vs biaya kehilangan calon nasabah), bukan cuma F1-optimal

## Kesimpulan Pribadi

Project ini berhasil menjawab kebingungan awal soal "kolom apa saja yang perlu disiapkan sebelum ke model" — ternyata jawabannya bukan checklist tetap, tapi proses tanya-jawab per kolom: apa dtype-nya, apakah ada missing dan kenapa, apakah perlu scaling, apakah kategorikal itu nominal atau ordinal. Setelah dilatih sekali dengan proses yang jelas kayak ini, harusnya project berikutnya lebih cepat karena kerangka berpikirnya sudah terbentuk.