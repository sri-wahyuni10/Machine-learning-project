# Deteksi TBC dari Chest X-ray dengan CNN: Eksperimen Sederhana dan Pemeriksaan Jalan Pintas (Shortcut)

> **Bukan alat diagnosis.** Proyek ini adalah latihan pembelajaran mesin pada data publik yang kecil. Hasilnya tidak boleh dipakai untuk keputusan medis.

Proyek portofolio computer vision di bidang kesehatan. Fokusnya bukan mengejar akurasi setinggi mungkin, tetapi memeriksa apakah model benar-benar belajar dari paru-paru atau dari hal lain yang kebetulan berkorelasi dengan label.

## Ringkasan

- **Masalah:** pembacaan gambar medis membutuhkan tenaga ahli yang terbatas. Pertanyaan proyek: seberapa jauh CNN kecil bisa membedakan chest X-ray TBC dan Normal?
- **Model:** CNN sederhana dengan 3 blok Conv2D, total 5.921 parameter, dilatih pada 560 gambar.
- **Hasil pada data test (120 gambar, dilihat satu kali):** recall 0,66, spesifisitas 0,74, accuracy 0,70, AUC 0,78. Dengan data sekecil ini, ketidakpastiannya besar (lihat bagian Hasil).
- **Temuan utama:** data publik ini punya jalan pintas. Warna, format file, dan sumber gambar berbeda antara kelas, dan model pertama terbukti sensitif terhadap tepi gambar. Memotong bagian tepi mengurangi masalah ini, tetapi tidak menghilangkannya.

## Data

- **Sumber:** [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) (Kaggle, Tawsifur Rahman dan kolaborator). Versi yang diunduh berisi 3.500 gambar Normal dan 700 gambar TBC.
- **Catatan README dataset:** versi aslinya menyebut 3.500 gambar TBC, tetapi 2.800 di antaranya berasal dari NIAID dan butuh perjanjian berbagi data, sehingga tidak ikut di versi Kaggle.
- **Asal gambar** (dari file metadata, kolom URL):

| | TBC | Normal |
|---|---|---|
| Artikel NLM (PMC4256233) | 394 | 406 |
| Program TBC Belarus | 306 | 0 |
| Lomba RSNA (pneumonia) | 0 | 3.094 |

- **Keputusan:** hanya memakai **subset NLM (800 gambar: 394 TBC, 406 Normal)**. Alasannya, kelas hampir seimbang dan berasal dari satu sumber, sehingga dua masalah besar (ketidakseimbangan 5:1 dan perbedaan sumber) hilang sekaligus. Harganya: data jadi sangat kecil.
- **Pembagian data:** train / val / test = 560 / 120 / 120, stratified, `random_state=42`.

## Pemeriksaan data sebelum training

Gambar TBC dan Normal terlihat berbeda dengan cara yang bukan tanda penyakit:

- **Warna:** pada sampel acak 200 gambar per kelas, 0 gambar Normal berwarna, sedangkan 106 gambar TBC berwarna (selisih kanal merah dan biru di atas ambang kasar 5).
- **Format file:** semua 3.500 gambar Normal berjenis RGB, sedangkan gambar TBC terbagi 392 RGB dan 308 satu kanal (grayscale).
- **Sumber eksklusif:** gambar Belarus hanya ada di kelas TBC dan gambar RSNA hanya ada di kelas Normal.
- **Patokan sederhana tanpa CNN:** regresi logistik dari hanya 4 angka (rata-rata terang tiap pita horizontal gambar) memberi accuracy val 0,74 dan recall 0,69. Dari 4 angka rata-rata tepi saja, accuracy val 0,64. Artinya, informasi non-paru sudah cukup untuk menebak jauh di atas kebetulan (51%).

## Pra-pemrosesan

- Semua gambar diubah ke abu-abu satu kanal, ukuran 224×224, nilai piksel dibagi 255.
- Model akhir memakai **potongan tengah 80%** gambar (10% dibuang di tiap sisi) untuk mengurangi pengaruh tepi.
- Tidak ada informasi pasien di metadata TBC, sehingga **tidak bisa dipastikan** satu pasien tidak muncul di lebih dari satu split.

## Model

CNN sederhana: 3 blok Conv2D (8, 16, 32 filter, kernel 3×3, ReLU) masing-masing diikuti MaxPooling, lalu GlobalAveragePooling dan satu neuron Dense sigmoid. Total 5.921 parameter. Optimizer Adam, loss binary cross-entropy, batch 32. Model terbaik dipilih dengan `val_loss` (ModelCheckpoint). Ukuran model sengaja kecil karena datanya hanya 560 gambar.

Dua model dibandingkan:

- **Model A:** gambar penuh, 45 epoch.
- **Model B:** gambar dipotong 80%, 45 epoch (model akhir).

## Eksperimen

### 1. Hasil validasi (garis pemisah 0,5)

| | Accuracy | Recall | Spesifisitas | AUC |
|---|---|---|---|---|
| Model A (penuh) | 0,817 | 0,864 | 0,77 | 0,896 |
| Model B (dipotong) | 0,825 | 0,712 | 0,934 | 0,865 |

Data val hanya 120 gambar (59 TBC), jadi selisih sekecil ini masih dalam derau.

### 2. Uji ketahanan terhadap tepi gambar

Bingkai 10% di tepi gambar val diganti warna tertentu tanpa melatih ulang. Model yang menilai paru tidak seharusnya berubah banyak. Hasil (accuracy / recall / spesifisitas):

| Isian bingkai | Model A | Model B |
|---|---|---|
| Tanpa penutup | 0,817 / 0,864 / 0,77 | 0,825 / 0,712 / 0,934 |
| Abu-abu rata-rata | 0,617 / 0,949 / 0,295 | 0,817 / 0,661 / 0,967 |
| Hitam | 0,55 / 0,085 / 1,0 | 0,492 / 1,0 / 0,0 |
| Putih | 0,717 / 0,814 / 0,623 | 0,775 / 0,542 / 1,0 |

- Pada tes abu-abu (paling netral), Model A turun sekitar 20 poin accuracy, sedangkan Model B hampir tidak berubah. Pemotongan membantu.
- Pada bingkai hitam, Model B berubah drastis (menjawab TBC untuk semua gambar). Bingkai hitam dan putih murni juga tepi tajam yang tidak ada di data latih, sehingga tes ini lebih tepat dibaca sebagai uji kerapuhan.
- Kesimpulan: Model B lebih tahan, tetapi belum bebas dari ketergantungan pada tepi.

### 3. Grad-CAM (Model A, data val)

- Pada gambar TBC yang benar, area panas terutama di dalam paru dengan pola yang mengikuti tulang rusuk. Ini tidak otomatis berarti model melihat lesi, karena bisa juga bereaksi pada pola tulang dan tekstur paru secara umum.
- Pada gambar Normal, ada bercak panas kecil di pojok.
- Porsi panas yang jatuh di bingkai 10%: Normal 0,49, TBC 0,34, patokan jika panas merata 0,36. Angka ini kasar dan peta Grad-CAM beresolusi rendah (26×26).

### 4. Pemilihan model dan garis pemisah

Sebelum membuka data test, diputuskan:

- **Model B**, karena Model A masih ikut belajar dari tepi yang tidak berhubungan dengan penyakit.
- **Garis pemisah 0,4** (bukan 0,5), karena dalam skrining kasus TBC yang lolos (FN) dianggap lebih berbahaya daripada orang sehat yang dicurigai (FP). Pada data val, Model B di garis 0,4 memberi recall 0,75 dan spesifisitas 0,82.

## Hasil akhir (data test, dilihat satu kali)

| | TBC sebenarnya | Normal sebenarnya |
|---|---|---|
| Diprediksi TBC | 39 (TP) | 16 (FP) |
| Diprediksi Normal | 20 (FN) | 45 (TN) |

| Metrik | Nilai | Perkiraan rentang 95% |
|---|---|---|
| Recall | 0,661 | sekitar 0,53 sampai 0,77 |
| Spesifisitas | 0,738 | sekitar 0,62 sampai 0,83 |
| Accuracy | 0,70 | sekitar 0,61 sampai 0,77 |
| AUC | 0,782 | tidak dihitung |

Rentang dihitung kasar dengan metode Wilson dari jumlah gambar (59 TBC, 61 Normal). Angka test lebih rendah daripada val karena model, epoch, dan garis pemisah dipilih dengan melihat data val, sehingga angka val sedikit terlalu optimis. Data test tidak dipakai untuk memilih apa pun.

## Analisis kesalahan

Pengamatan visual awal pada gambar test yang salah (belum dihitung sistematis):

- Beberapa TBC yang lolos berukuran sangat kecil atau tidak menampilkan paru dengan jelas. Ini masalah kualitas data.
- Beberapa TBC yang lolos tampak berparu gelap bersih dengan garis putih di sisi gambar.
- Beberapa Normal yang dicurigai punya huruf penanda (misalnya "L") di pojok atau gambar yang tampak sangat terang.

Penghitungan yang perlu dilengkapi: [isi jumlah FP bertanda pojok dibanding TN acak] dan [isi jumlah FN dengan paru kecil atau tidak terlihat]. Huruf penanda ada di gambar Normal dan TBC, jadi hanya perbandingan frekuensi antara FP dan TN yang bermakna.

## Keterbatasan

- **Data sangat kecil:** hanya 800 gambar, dengan 120 gambar test. Hasil sangat berderau.
- **Tidak ada informasi pasien:** kebocoran antar-split (pasien yang sama di train dan test) tidak bisa diperiksa.
- **Satu sumber:** hasil belum diuji pada mesin atau rumah sakit lain, dan tidak ada jaminan model berlaku di luar data ini.
- **Ketergantungan pada tepi belum hilang:** pemotongan mengurangi, tetapi tes bingkai hitam menunjukkan model masih rapuh.
- **Tidak ada validasi klinis:** area merah Grad-CAM belum diperiksa oleh tenaga medis, dan label tidak diverifikasi ulang.
- **Recall sedang:** sekitar sepertiga kasus TBC pada test tidak terdeteksi, jadi model ini tidak layak dipakai sebagai alat skrining sungguhan.

## Langkah selanjutnya

- Cross-validation pada gabungan train+val untuk perbandingan model yang lebih stabil.
- Augmentasi data dan transfer learning dari model pretrained, dengan evaluasi test yang tetap tidak disentuh sampai akhir.
- Mengatasi huruf penanda dan kotak di gambar (misalnya masking) dan membandingkannya dengan pemotongan.
- Memakai dataset dengan informasi pasien dan beberapa sumber agar generalisasi bisa diuji.

## Cara menjalankan

Kode dijalankan di Google Colab. Alur: unduh dataset dari Kaggle, susun tabel `path` dan `label` dari subset NLM, bagi train/val/test, lalu latih model.



## Kredit

Dataset: Tawsifur Rahman dan kolaborator (Qatar University, University of Dhaka, Hamad Medical Corporation, dan lainnya) di Kaggle; gambar berasal dari NLM, program TBC Belarus, RSNA, dan NIAID sebagaimana tertulis di README dataset. Silakan ikuti ketentuan lisensi dataset asli bila menggunakan ulang.


