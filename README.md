# datamining-lrn

Kumpulan projek kecil dan eksperimen belajar Data Mining & Machine Learning. 

## Daftar Projek

1. `01-eda-house-pricing`
   - Eksplorasi data harga rumah, statistik dasar, dan korelasi antar fitur.
2. `02-regression-car-price`
   - Prediksi harga mobil bekas pakai model Regresi (Linear Regression & Ridge).
3. `03-classification-customer-churn`
   - Klasifikasi pelanggan churn / bertahan pakai Random Forest.
4. `04-unsupervised-mall-segmentation`
   - Segmentasi pelanggan mall berdasarkan pendapatan & pengeluaran pakai K-Means.
5. `05-deep-learning-image-mnist`
   - Pengenalan angka tulisan tangan dataset MNIST pakai Neural Network sederhana.
6. `06-credit-risk-assessment`
   - Prediksi risiko gagal bayar (loan default) perbankan dari dataset Excel (`.xlsx`) multi-sheet (`Data_Debitur` & `Riwayat_Pinjaman`) dengan Machine Learning & evaluasi imbalanced class.


## Cara Jalankan

1. Clone repo ini:
   ```bash
   git clone https://github.com/username/datamining-lrn.git
   cd datamining-lrn
   ```

2. Buat dan aktifkan virtual environment (venv):
   ```bash
   python -m venv venv
   
   # Di macOS / Linux:
   source venv/bin/activate
   
   # Di Windows:
   venv\Scripts\activate
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Buka Jupyter Lab / Notebook:
   ```bash
   jupyter lab
   ```
