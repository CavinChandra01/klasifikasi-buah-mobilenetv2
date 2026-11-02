# Klasifikasi Buah Segar dan Busuk Menggunakan CNN Transfer Learning (MobileNetV2)

Mata kuliah Pembelajaran Mendalam, Semester Ganjil TA 2026/2027.

Model mengklasifikasikan citra buah ke dalam **6 kelas**: apel segar, pisang segar, jeruk segar, apel busuk, pisang busuk, dan jeruk busuk.

Repository ini memuat dua tahap pengerjaan:

| Tahap | Isi |
|---|---|
| **Tugas 1** | MobileNetV2 + fine-tuning, fitur unik 1 (**Grad-CAM**) dan fitur unik 2 (**confidence threshold & deteksi input di luar domain**), aplikasi Streamlit |
| **Proyek 1** | Lanjutan Tugas 1 dengan dua fitur unik baru: fitur 3 (**kompresi model ke TensorFlow Lite**) dan fitur 4 (**Test-Time Augmentation**) |

## Tautan

- **Aplikasi (Streamlit Community Cloud):** https://klasifikasi-buah-segar-busuk.streamlit.app/
- **Notebook Kaggle Tugas 1:** https://www.kaggle.com/code/cavinchandra/notebookd31b9accff
- **Notebook Kaggle Proyek 1:** _tambahkan link notebook Kaggle Proyek 1 di sini_
- **Dataset:** [Fruits fresh and rotten for classification (Kaggle)](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

## Anggota kelompok dan pembagian tugas

| NIM | Nama | Peran | Rincian pekerjaan |
|---|---|---|---|
| 231111601 | Cavin Chandra | Model & aplikasi | Preprocessing, pelatihan dan evaluasi MobileNetV2, Grad-CAM, confidence threshold, kompresi TFLite, TTA, aplikasi Streamlit dan deploy |
| 231112974 | Jimmy Anggoro | EDA & laporan | Pemeriksaan data, EDA, analisis hasil, dan penyusunan laporan |

## Hasil utama (data test, 2.698 gambar)

### Tugas 1: model

| Skenario | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| A. MobileNetV2 dibekukan (feature extraction) | 98,89% | 98,88% | 98,89% | 98,88% |
| **B. + Fine-tuning (model final)** | **99,74%** | **99,76%** | **99,71%** | **99,73%** |

- Hanya 7 dari 2.698 gambar test yang salah diprediksi; 4 di antaranya jeruk busuk yang diprediksi apel busuk.
- Tidak ada indikasi overfitting: akurasi train 99,87% dan validation 99,69%.
- **Confidence threshold 0,95:** 98,85% gambar test diterima dengan akurasi 99,89%; 78% gambar di luar domain berhasil ditolak (AUROC 0,99).

### Proyek 1: kompresi TensorFlow Lite

| Model | Ukuran | Akurasi test | Prediksi sama dengan Keras | Latensi CPU |
|---|---|---|---|---|
| Keras (`.keras`) | 24,5 MB | 99,74% | 100% | – |
| TFLite float32 | 8,9 MB | 99,74% | 100% | 4,3 ms |
| **TFLite float16** | **4,5 MB** | **99,70%** | **99,96%** | **4,7 ms** |
| TFLite int8 | 2,5 MB | 99,78% | 99,96% | 23,6 ms |

### Proyek 1: Test-Time Augmentation (TTA)

| Kondisi data test | Tanpa TTA | Dengan TTA |
|---|---|---|
| Asli (bersih) | 99,74% | **99,78%** |
| Gelap (−40%) | 99,33% | **99,70%** |
| Terang (+40%) | 97,78% | **98,30%** |
| Blur | 98,81% | **99,37%** |
| Noise Gaussian | **62,86%** | 62,19% |
| Rotasi 90° | 99,67% | **99,78%** |

TTA mengurangi kesalahan pada 4 dari 5 jenis gangguan, tetapi tidak membantu pada noise Gaussian dan membuat inferensi sekitar 4 kali lebih lama, sehingga dibuat opsional di aplikasi.

## Fitur unik

1. **Grad-CAM** (Tugas 1): menampilkan area gambar yang paling memengaruhi keputusan model.
2. **Confidence threshold dan deteksi input di luar domain** (Tugas 1): prediksi dengan keyakinan di bawah 0,95 ditolak sebagai "tidak yakin / bukan buah yang dikenali".
3. **Kompresi model ke TensorFlow Lite** (Proyek 1): model 24,5 MB dikonversi menjadi TFLite float16 (4,5 MB) dan int8 (2,5 MB) dengan akurasi setara.
4. **Test-Time Augmentation** (Proyek 1): prediksi dirata-rata dari 4 versi gambar (asli, flip horizontal, flip vertikal, rotasi 180°).

Di aplikasi, fitur 3 dan 4 dipilih melalui sidebar (**Model** dan **Test-Time Augmentation**). Pengaturan bawaan (Keras, TTA nonaktif) sama dengan aplikasi Tugas 1.

## Struktur repository

```
.
├── notebook/
│   ├── klasifikasi_buah_mobilenetv2.ipynb   # Tugas 1: EDA, preprocessing, training, evaluasi, Grad-CAM, threshold
│   └── proyek1_fitur_tambahan.ipynb         # Proyek 1: kompresi TFLite dan Test-Time Augmentation
├── app/
│   ├── streamlit_app.py                     # aplikasi Streamlit
│   ├── fruit_mobilenetv2.keras              # model final (Keras)
│   ├── fruit_mobilenetv2_float16.tflite     # model TFLite float16 (Proyek 1)
│   ├── fruit_mobilenetv2_int8.tflite        # model TFLite int8 (Proyek 1)
│   ├── config.json                          # kelas, ukuran input, threshold
│   └── requirements.txt
└── README.md
```

## Ringkasan metode

- **Data:** 13.599 citra (10.901 train, 2.698 test). Folder train dibagi stratified menjadi 9.265 train dan 1.636 validation; folder test hanya dipakai untuk evaluasi akhir.
- **Preprocessing:** cek file rusak dan duplikat (MD5), resize 224×224, normalisasi piksel ke [-1, 1] di dalam model, augmentasi (flip, rotasi, zoom, kontras), class weight untuk ketidakseimbangan kelas.
- **Model:** MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout 0,3 → Dense softmax 6 (L2 1e-4). Total 2.265.670 parameter.
- **Pelatihan dua tahap:** (1) backbone dibekukan, Adam lr 1e-3; (2) fine-tuning layer ke-100 ke atas, Adam lr 1e-5. Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- **Proyek 1 (tanpa pelatihan ulang):** konversi TFLite dengan `tf.lite.TFLiteConverter` (float32, float16, int8 dynamic range), serta TTA yang diuji pada data test bersih, data test dengan 5 jenis gangguan, dan 200 input di luar domain.

## Cara menjalankan

**Notebook Tugas 1:** buka `notebook/klasifikasi_buah_mobilenetv2.ipynb` di Kaggle, tambahkan dataset di atas, aktifkan GPU dan Internet, lalu Run All.

**Notebook Proyek 1:** buka `notebook/proyek1_fitur_tambahan.ipynb` di Kaggle, tambahkan dataset di atas **dan** output notebook Tugas 1 (Add Input → Your Work → Notebooks), aktifkan GPU dan Internet, lalu Run All.

**Aplikasi secara lokal:**

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Lalu buka `http://localhost:8501` di browser.

## Teknologi

Python 3.13, TensorFlow 2.20 (termasuk TensorFlow Lite), Keras 3.13, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Pillow, Streamlit, Kaggle Notebook (GPU NVIDIA T4), Streamlit Community Cloud.
