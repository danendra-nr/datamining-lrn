# 01. Data Preprocessing

Bagian ini membahas tahapan fundamental dalam Data Mining: **Pembersihan dan Penyiapan Data Mentah**.

## Topik Utama

1. **Handling Missing Values (Penanganan Nilai Kosong)**:
   - Identifikasi pola data hilang (MCAR, MAR, MNAR).
   - Strategi imputasi statistik (Mean, Median, Mode, Arbitrary).
   - Imputasi berbasis algoritma (KNN Imputer).

2. **Outlier Detection and Treatment (Deteksi Pencilan)**:
   - Z-Score method (asumsi distribusi normal).
   - Interquartile Range (IQR) and Boxplot fencing.
   - Trimming vs. Capping (Winsorization).

3. **Data Cleaning and Deduplication**:
   - Penghapusan duplikasi baris.
   - Standardisasi format teks/kategori dan tipe data.

## Berkas dan Implementasi

- `data_preprocessing_pipeline.ipynb`: Notebook eksperimen interaktif langkah demi langkah.
- `data_preprocessing_pipeline.py`: Skrip pipeline siap pakai untuk data cleaning.
