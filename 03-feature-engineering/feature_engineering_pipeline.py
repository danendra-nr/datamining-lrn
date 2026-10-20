"""
Modul: 03-feature-engineering
Topik: Categorical Encoding, Numerical Scaling, and Feature Extraction
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def build_sample_feature_data() -> pd.DataFrame:
    """Membuat sampel data untuk demonstrasi rekayasa fitur."""
    return pd.DataFrame({
        "city": ["Jakarta", "Surabaya", "Bandung", "Jakarta", "Semarang", "Surabaya"],
        "job_level": ["Junior", "Mid", "Senior", "Junior", "Lead", "Mid"],
        "years_experience": [1.5, 3.0, 7.5, 0.8, 10.0, 4.2],
        "monthly_income": [5000, 9000, 22000, 4500, 35000, 11000],
        "portfolio_projects": [2, 5, 12, 1, 15, 6],
    })


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline transformasi dan rekayasa fitur."""
    df_transformed = df.copy()

    # 1. Feature Creation: Rasio pendapatan per tahun pengalaman
    df_transformed["income_per_year_exp"] = (
        df_transformed["monthly_income"] / (df_transformed["years_experience"] + 1)
    ).round(2)

    # 2. Log Transformation untuk fitur moneter (meredam skewness)
    df_transformed["log_monthly_income"] = np.log1p(df_transformed["monthly_income"])

    # 3. Column Transformer: One-Hot Encoding & Scaling
    nominal_cols = ["city", "job_level"]
    numerical_cols = ["years_experience", "portfolio_projects", "income_per_year_exp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(sparse_output=False, drop="first"), nominal_cols),
        ],
        remainder="passthrough",
    )

    processed_matrix = preprocessor.fit_transform(df_transformed)
    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(processed_matrix, columns=feature_names)


if __name__ == "__main__":
    print("=== FEATURE ENGINEERING DEMO ===")
    sample_df = build_sample_feature_data()
    print("Data Asli:")
    print(sample_df)

    engineered_df = apply_feature_engineering(sample_df)
    print("\nData Setelah Feature Engineering:")
    print(engineered_df.head())
