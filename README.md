# Klasifikasi Buah Segar dan Busuk Menggunakan CNN Transfer Learning (MobileNetV2)

Proyek Ujian Akhir Semester mata kuliah Pembelajaran Mendalam, Semester Ganjil TA 2026/2027.

Model mengklasifikasikan citra buah ke dalam **6 kelas**: apel segar, pisang segar, jeruk segar, apel busuk, pisang busuk, dan jeruk busuk.

- **Aplikasi (Streamlit Community Cloud):** https://klasifikasi-buah-segar-busuk.streamlit.app/
- **Notebook Kaggle:** https://www.kaggle.com/code/cavinchandra/notebookd31b9accff
- **Dataset:** [Fruits fresh and rotten for classification (Kaggle)](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

## Anggota kelompok

| NIM | Nama | Peran |
|---|---|---|
| ... | ... | ... |

## Hasil utama (data test, 2.698 gambar)

| Skenario | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| A. MobileNetV2 dibekukan (feature extraction) | 98,89% | 98,88% | 98,89% | 98,88% |
| **B. + Fine-tuning (model final)** | **99,74%** | **99,76%** | **99,71%** | **99,73%** |

- Hanya 7 dari 2.698 gambar test yang salah diprediksi; 4 di antaranya jeruk busuk yang diprediksi apel busuk.
- Tidak ada indikasi overfitting: akurasi train 99,87% dan validation 99,69%.
- **Confidence threshold 0,95:** 98,85% gambar test diterima dengan akurasi 99,89%; 78% gambar di luar domain berhasil ditolak (AUROC 0,99).

## Dua fitur unik

1. **Grad-CAM**: menampilkan area gambar yang paling memengaruhi keputusan model, untuk memastikan model melihat bagian buah (bercak busuk, warna, tekstur), bukan latar belakang.
2. **Confidence threshold dan deteksi input di luar domain**: prediksi dengan keyakinan di bawah ambang ditolak sebagai "tidak yakin / bukan buah yang dikenali". Ambang dipilih dari data validation.

## Struktur repository

```
.
├── notebook/
│   └── klasifikasi_buah_mobilenetv2.ipynb   # EDA, preprocessing, training, evaluasi, Grad-CAM, threshold
├── app/
│   ├── streamlit_app.py                     # aplikasi Streamlit
│   ├── fruit_mobilenetv2.keras              # model final
│   ├── config.json                          # kelas, ukuran input, threshold
│   └── requirements.txt
└── README.md
```

## Ringkasan metode

- **Data:** 13.599 citra (10.901 train, 2.698 test). Folder train dibagi stratified menjadi 9.265 train dan 1.636 validation; folder test hanya dipakai untuk evaluasi akhir.
- **Preprocessing:** cek file rusak dan duplikat (MD5), resize 224×224, normalisasi piksel ke [-1, 1] di dalam model, augmentasi (flip, rotasi, zoom, kontras), class weight untuk ketidakseimbangan kelas.
- **Model:** MobileNetV2 (ImageNet) → GlobalAveragePooling → Dropout 0,3 → Dense softmax 6 (L2 1e-4). Total 2.265.670 parameter.
- **Pelatihan dua tahap:** (1) backbone dibekukan, Adam lr 1e-3; (2) fine-tuning layer ke-100 ke atas, Adam lr 1e-5. Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

## Cara menjalankan

**Notebook:** buka `notebook/klasifikasi_buah_mobilenetv2.ipynb` di Kaggle, tambahkan dataset di atas, aktifkan GPU dan Internet, lalu Run All.

**Aplikasi secara lokal:**

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Lalu buka `http://localhost:8501` di browser.

## Teknologi

Python, TensorFlow 2.20, Keras 3.13, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, Pillow, Streamlit, Kaggle Notebook (GPU NVIDIA T4), Streamlit Community Cloud.
