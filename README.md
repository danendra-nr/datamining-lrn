# Data Mining and Machine Learning Repository

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/notebooks-JupyterLab-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)

Repositori terstruktur untuk pembelajaran dan eksperimen **Data Mining, Machine Learning, dan Deep Learning**, dari konsep dasar rekayasa data hingga implementasi studi kasus terapan.

---

## Daftar Topik Pembelajaran

| No | Kategori | Topik Inti | Algoritma / Metode | Implementasi / Studi Kasus |
|---|---|---|---|---|
| 01 | [Data Preprocessing](./01-data-preprocessing/) | Missing values imputation, outlier handling, cleaning | KNN Imputer, IQR method, SimpleImputer | Sintetis Data Perbankan |
| 02 | [Exploratory Data Analysis](./02-exploratory-data-analysis/) | Analisis univariat, bivariat, korelasi fitur | Matriks korelasi, distribusi statistik | [House Price Analysis](./02-exploratory-data-analysis/house-price-eda/) |
| 03 | [Feature Engineering](./03-feature-engineering/) | Encoding kategorikal, penskalaan fitur, transformasi log | One-Hot Encoding, RobustScaler, Log1p | Pipeline Rekayasa Fitur Karir & Gaji |
| 04 | [Supervised Learning](./04-supervised-learning/) | Pemodelan regresi dan klasifikasi terarah | Linear Regression, Ridge, Random Forest | [Car Price](./04-supervised-learning/regression/car-price-prediction/) & [Customer Churn](./04-supervised-learning/classification/customer-churn-prediction/) |
| 05 | [Unsupervised Learning](./05-unsupervised-learning/) | Pengelompokan data tanpa label, segmentasi | K-Means Clustering, Elbow Method | [Mall Customer Segmentation](./05-unsupervised-learning/customer-segmentation-kmeans/) |
| 06 | [Deep Learning Fundamentals](./06-deep-learning-fundamentals/) | Pengantar arsitektur jaringan saraf tiruan | Multi-Layer Perceptron (MLP) | [MNIST Handwritten Digits](./06-deep-learning-fundamentals/mnist-digit-recognition/) |
| 07 | [End-to-End Case Studies](./07-end-to-end-case-studies/) | Integrasi multi-sheet dataset, handling imbalanced data | Relational Join, Random Forest Classifier | [Credit Risk Assessment](./07-end-to-end-case-studies/credit-risk-assessment/) |

---

## Struktur Direktori

```text
datamining-lrn/
├── 01-data-preprocessing/
│   ├── data_preprocessing_pipeline.ipynb
│   ├── data_preprocessing_pipeline.py
│   └── README.md
├── 02-exploratory-data-analysis/
│   ├── house-price-eda/
│   └── README.md
├── 03-feature-engineering/
│   ├── feature_engineering_pipeline.ipynb
│   ├── feature_engineering_pipeline.py
│   └── README.md
├── 04-supervised-learning/
│   ├── regression/
│   │   └── car-price-prediction/
│   ├── classification/
│   │   └── customer-churn-prediction/
│   └── README.md
├── 05-unsupervised-learning/
│   ├── customer-segmentation-kmeans/
│   └── README.md
├── 06-deep-learning-fundamentals/
│   ├── mnist-digit-recognition/
│   └── README.md
├── 07-end-to-end-case-studies/
│   ├── credit-risk-assessment/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Panduan Menjalankan Projek

### 1. Kloning Repositori
```bash
git clone https://github.com/danendra-nr/datamining-lrn.git
cd datamining-lrn
```

### 2. Konfigurasi Virtual Environment
```bash
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Jupyter Lab
```bash
jupyter lab
```

---

## Pustaka dan Perangkat Lunak
- **Manipulasi dan Analisis Data**: `pandas`, `numpy`
- **Machine Learning & Preprocessing**: `scikit-learn`
- **Visualisasi Data**: `matplotlib`, `seaborn`
- **Penanganan Berkas Data**: `openpyxl`
- **Lingkungan Interaktif**: `jupyter`, `jupyterlab`

---

## Lisensi
Repositori ini ditujukan untuk sarana pembelajaran, riset, dan eksplorasi teknis mandiri.
