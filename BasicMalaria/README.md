# Deteksi Malaria dari Gambar Sel Darah dengan CNN

Proyek klasifikasi gambar medis: sebuah CNN (Convolutional Neural Network) yang menilai apakah satu sel darah **terinfeksi parasit malaria** atau **tidak terinfeksi**. Proyek ini juga memeriksa apa yang sebenarnya dilihat model, bukan hanya seberapa tinggi skornya.

> **Catatan penting:** proyek ini adalah latihan portofolio pada dataset publik. Hasilnya berlaku untuk klasifikasi **per sel** pada dataset ini, bukan alat diagnosis malaria.

---

## 1. Masalah dan tujuan

Malaria didiagnosis salah satunya dengan memeriksa sediaan darah di bawah mikroskop, dan pekerjaan itu membutuhkan tenaga ahli yang terlatih. Di tempat dengan tenaga ahli yang terbatas, pembacaan manual bisa lambat dan melelahkan. Proyek ini mencoba menjawab pertanyaan yang lebih kecil: **seberapa baik sebuah CNN sederhana dapat membedakan sel terinfeksi dan sel sehat dari gambar sel tunggal?**

Tujuan teknisnya:
- membangun alur kerja end-to-end (pemeriksaan data, baseline, model, evaluasi, interpretasi);
- memastikan model belajar dari parasit, bukan dari kebetulan data (warna, latar, tepi gambar);
- melaporkan hasil dan keterbatasannya dengan jujur.

## 2. Data

- **Sumber:** Malaria Cell Images Dataset (Kaggle, berasal dari U.S. National Library of Medicine). 
- **Isi:** 27.558 gambar sel darah (potongan per sel), dua kelas seimbang: `parasitized` dan `uninfected`.
- **Pembagian:**

| Bagian | Jumlah gambar | Keterangan |
|---|---|---|
| Train | 19.967 | 80% dari folder `train`, diacak (seed 42) |
| Validasi | 4.991 | 20% dari folder `train`, untuk memantau training dan memilih garis pemisah |
| Test | 2.600 | folder `test` dari dataset (1.300 per kelas), **diukur satu kali** di akhir |

- **Pemeriksaan sebelum membuat model:**
  - Ukuran gambar asli beragam (lebar 64 sampai 226 piksel, tinggi 61 sampai 217), jadi semua gambar diubah ke **96×96**. Ukuran ini cukup kecil agar training cepat dan cukup besar agar bercak parasit yang kecil tetap terlihat.
  - Setiap sel dikelilingi **latar hitam**. Proporsinya hampir sama antar kelas (0,286 terinfeksi vs 0,279 sehat), jadi bentuk latar tidak membocorkan label.
  - **Model pembanding hanya dari warna rata-rata** (regresi logistik pada rata-rata R, G, B sel) mendapat akurasi 0,635. Warna adalah petunjuk lemah, sehingga CNN yang jauh di atas angka ini berarti membaca hal lain selain warna.
  - Beberapa gambar tampak berlabel meragukan (misalnya sel tanpa bercak yang berlabel parasit, atau sel berbercak yang berlabel sehat).

## 3. Metode

**Pipeline data:** gambar dibaca bertahap dari folder (batch 64, ukuran 96×96) sehingga memori tetap aman untuk ~25 ribu gambar. Nilai piksel diubah ke rentang 0 sampai 1 di dalam model.

**CNN pertama (baseline sederhana, 25.697 parameter):**
`Rescaling` → 3 × (`Conv2D` 16/32/64 + `MaxPooling`) → `GlobalAveragePooling` → `Dense 32` → `Dropout 0,3` → `Dense 1` (sigmoid). Dilatih 15 epoch.

**CNN kedua (model akhir, 101.569 parameter):**
Augmentasi saat training (flip horizontal dan vertikal, rotasi, perubahan kecerahan kecil; rotasi diisi hitam agar latar tetap hitam) → `Rescaling` → 3 × (`Conv2D` 32/64/128 + `MaxPooling`) → `GlobalAveragePooling` → `Dense 64` → `Dropout 0,3` → `Dense 1` (sigmoid). Dilatih 30 epoch. Augmentasi dipilih karena sel darah tidak punya orientasi "atas" atau "bawah".

**Pelatihan:** optimizer Adam, loss binary cross-entropy, metrik accuracy, recall, dan AUC. Model terbaik disimpan berdasarkan `val_loss`. Dijalankan di Google Colab dengan GPU. Framework: TensorFlow/Keras, scikit-learn.

**Pemilihan garis pemisah:** kesalahan dibagi menjadi *sel terinfeksi yang lolos* (FN) dan *sel sehat yang dicurigai* (FP). Dalam konteks medis, FN lebih berbahaya, jadi garis diturunkan dari 0,5. Pilihan dibuat **hanya dari data validasi**:

| Garis | FN | FP | Recall | Spesifisitas | Presisi |
|---|---|---|---|---|---|
| 0,2 | 67 | 177 | 0,974 | 0,928 | 0,933 |
| **0,3** | **92** | **130** | **0,964** | **0,947** | **0,949** |
| 0,4 | 110 | 98 | 0,957 | 0,960 | 0,961 |
| 0,5 | 125 | 78 | 0,951 | 0,968 | 0,969 |
| 0,6 | 152 | 64 | 0,940 | 0,974 | 0,974 |

Garis **0,3** dipilih: recall tinggi dengan harga alarm palsu yang masih wajar (tiap kasus tambahan yang tertangkap menambah kurang dari 2 alarm palsu).

## 4. Hasil

**Validasi (CNN pertama vs kedua):**

| | CNN pertama | CNN kedua |
|---|---|---|
| Parameter | 25.697 | 101.569 |
| val_acc | 0,956 | 0,959 |
| val_recall (garis 0,5) | 0,946 | 0,951 |
| val_AUC | 0,988 | 0,991 |
| val_loss terbaik | 0,1346 | 0,1188 |

Selisih akurasi kecil dan masih dalam kisaran noise. Perbaikan yang lebih nyata terlihat pada loss, AUC, dan jumlah alarm palsu pada garis rendah (di garis 0,3: 233 FP pada CNN pertama vs 130 FP pada CNN kedua).

**Test akhir (CNN kedua, garis 0,3, 2.600 gambar, diukur sekali):**

| TN | FP | FN | TP |
|---|---|---|---|
| 1229 | 71 | 55 | 1245 |

| Recall | Spesifisitas | Presisi | Akurasi | AUC |
|---|---|---|---|---|
| 0,958 | 0,945 | 0,946 | 0,952 | 0,989 |

Angka test hampir sama dengan validasi. Dengan 1.300 sel per kelas, ketidakpastiannya sekitar ±1 poin (recall wajar berada di kisaran 0,947 sampai 0,969).

Gambar pendukung: `figures/kurva_training.png`, `figures/confusion_roc_test.png`.

![Kurva training](figures/kurva_training.png)
![Confusion matrix dan ROC](figures/confusion_roc_test.png)

## 5. Apa yang dilihat model (Grad-CAM)

Grad-CAM menghasilkan peta panas dari lapisan konvolusi terakhir yang menunjukkan bagian gambar yang paling mendorong keputusan "terinfeksi". Diperiksa pada 12 sel dari data test (4 benar terdeteksi, 4 lolos, 4 alarm palsu):

- **Deteksi yang yakin (prob ≈ 1,00):** area panas jatuh pada struktur berbentuk **cincin atau bulan sabit** di dalam sel, yang sesuai dengan penampilan parasit. Artinya model membaca parasitnya, bukan latar atau tepi gambar.
- **Parasit yang lolos (prob 0,01 sampai 0,27):** selnya tampak polos tanpa cincin yang jelas, dan area panasnya lemah dan berada di tepi.
- **Alarm palsu (prob 0,41 sampai 0,84):** sebagian berupa sel sehat dengan struktur menyerupai bulan sabit (kemungkinan label meragukan atau struktur mirip), sebagian lagi sel berbentuk tidak beraturan dengan panas di tepi.

![Grid Grad-CAM](figures/gradcam_grid.png)

Catatan membaca: peta dinormalkan sehingga titik terkuatnya selalu tampak merah, bahkan saat buktinya lemah. Karena itu peta selalu dibaca bersama probabilitas pada judul tiap gambar. Peta 20×20 juga hanya petunjuk kasar lokasi, bukan batas yang presisi. Pengamatan ini berasal dari 12 contoh dan dibuat dari gambar, bukan penilaian ahli.

## 6. Keterbatasan

1. **Tidak ada ID pasien.** Satu pasien menghasilkan banyak sel. Sel dari pasien yang sama bisa berada di train, validasi, dan test sekaligus, sehingga hasil kemungkinan lebih optimis dari kenyataan. Kecocokan angka test dan validasi **tidak membuktikan** tidak ada kebocoran, karena keduanya bisa sama-sama terpengaruh.
2. **Hasil per sel, bukan per pasien.** Diagnosis sebenarnya memeriksa banyak sel dalam satu sediaan darah. Proyek ini tidak mengevaluasi diagnosis tingkat pasien.
3. **Satu dataset, satu sumber.** Tidak diuji pada gambar dari mikroskop, pewarnaan, atau rumah sakit lain, sehingga kemampuan generalisasinya belum diketahui.
4. **Label tidak sempurna.** Beberapa gambar yang salah ditebak tampak seperti masalah label (sel tanpa bercak berlabel parasit, atau sel berbercak berlabel sehat). Ini dugaan dari pengamatan visual yang belum diverifikasi ahli, tetapi menunjukkan bahwa akurasi jauh di atas sekitar 96% pada dataset ini patut dicurigai.
5. **Bentuk tepi tidak biasa.** Sebagian kesalahan terjadi pada sel yang terlipat, bergerigi, atau berlekuk. Dugaan ini dari contoh terbatas dan belum diuji secara sistematis.
6. **Hanya satu spesies dan satu jenis gambar** (sel tunggal hasil potongan otomatis), bukan gambar mikroskop utuh.

**Ide pengembangan:** evaluasi pada dataset lain, deteksi sel langsung dari gambar mikroskop utuh, pemeriksaan sensitivitas terhadap bentuk tepi sel, dan peninjauan ulang label oleh ahli.



**Menjalankan:**
1. Unduh dataset dari Kaggle dan letakkan zip-nya di Google Drive.
2. Buka `notebook.ipynb` di Google Colab, aktifkan GPU (Runtime → Change runtime type), lalu jalankan sel dari atas.
3. Folder `test` hanya dipakai pada langkah evaluasi akhir.

**Pustaka:** TensorFlow/Keras, scikit-learn, NumPy, pandas, Matplotlib, Pillow.

## Ucapan terima kasih

Dataset: Malaria Cell Images Dataset (Kaggle), berasal dari U.S. National Library of Medicine. 
