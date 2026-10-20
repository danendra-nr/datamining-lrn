"""
Modul: 01-data-preprocessing
Topik: Missing Values Handling, Outlier Detection, and Data Cleaning Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def create_sample_raw_dataset() -> pd.DataFrame:
    """Membuat synthetic raw dataset dengan missing values dan outliers."""
    np.random.seed(42)
    n = 100
    
    data = {
        "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
        "age": np.random.choice([20, 25, 30, 35, 40, np.nan, 120, -5], size=n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.025]),
        "monthly_income": np.random.choice([3000, 4500, 7000, 12000, np.nan, 250000], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.08, 0.02]),
        "education": np.random.choice(["High School", "Bachelor", "Master", "bachelor", None], size=n, p=[0.25, 0.45, 0.15, 0.1, 0.05]),
        "credit_score": np.random.normal(loc=650, scale=80, size=n).round(),
    }
    
    df = pd.DataFrame(data)
    # Tambahkan baris duplikat untuk simulasi pembersihan duplikasi
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline pembersihan dasar data."""
    df_clean = df.copy()
    
    # 1. Hapus duplikasi
    df_clean = df_clean.drop_duplicates()
    
    # 2. Standardisasi teks
    df_clean["education"] = df_clean["education"].astype(str).str.strip().str.title()
    df_clean["education"] = df_clean["education"].replace({"None": np.nan, "Nan": np.nan})
    
    # 3. Validasi rentang usia logis (18 - 85)
    df_clean.loc[(df_clean["age"] < 18) | (df_clean["age"] > 85), "age"] = np.nan
    
    # 4. Outlier capping (IQR method) pada monthly_income
    q1 = df_clean["monthly_income"].quantile(0.25)
    q3 = df_clean["monthly_income"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    df_clean.loc[df_clean["monthly_income"] > upper_bound, "monthly_income"] = upper_bound
    
    # 5. Imputasi nilai kosong
    # Imputasi kategorikal (modus)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_clean[["education"]] = cat_imputer.fit_transform(df_clean[["education"]])
    
    # Imputasi numerik dengan KNN Imputer
    num_cols = ["age", "monthly_income", "credit_score"]
    knn_imputer = KNNImputer(n_neighbors=3)
    df_clean[num_cols] = knn_imputer.fit_transform(df_clean[num_cols])
    
    return df_clean


if __name__ == "__main__":
    print("=== DATA PREPROCESSING PIPELINE DEMO ===")
    raw_df = create_sample_raw_dataset()
    print(f"Bentuk data mentah: {raw_df.shape}")
    print("Missing values sebelum dibersihkan:\n", raw_df.isnull().sum())
    
    cleaned_df = clean_raw_data(raw_df)
    print("\nBentuk data setelah dibersihkan:", cleaned_df.shape)
    print("Missing values setelah dibersihkan:\n", cleaned_df.isnull().sum())
    print("\n5 Sampel Data Bersih:\n", cleaned_df.head())
