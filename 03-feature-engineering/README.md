# 03. Feature Engineering

Bagian ini membahas teknik transformasi dan rekayasa fitur agar model Machine Learning dapat mengekstrak pola secara optimal.

## Topik Utama

1. **Categorical Encoding**:
   - One-Hot Encoding (fitur nominal tanpa tingkatan).
   - Ordinal Encoding and Target Encoding (fitur bertingkat atau kardinalitas tinggi).

2. **Feature Scaling and Normalization**:
   - StandardScaler (Z-Score standardization, mean=0, std=1).
   - MinMaxScaler (Transformasi rentang 0 hingga 1).
   - RobustScaler (Tahan outlier menggunakan median dan IQR).

3. **Feature Transformation and Creation**:
   - Log Transformation (meredam skewness distribusi).
   - Interaction Features (rasio dan kombinasi antar variabel).

4. **Feature Selection**:
   - Filter methods (VarianceThreshold, Correlation Matrix).
   - Model-based selection (Feature Importance and Mutual Information).

## Berkas dan Implementasi

- `feature_engineering_pipeline.ipynb`: Eksperimen interaktif transformasi fitur.
- `feature_engineering_pipeline.py`: Skrip pipeline feature engineering modular.
