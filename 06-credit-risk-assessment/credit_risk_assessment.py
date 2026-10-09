"""
Supervised Learning: Credit Risk Assessment
Memprediksi risiko gagal bayar (loan default) nasabah menggunakan Random Forest.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score

def load_data(file_path):
    """Membaca dataset multi-sheet dari file Excel."""
    sheet_debitur = pd.read_excel(file_path, sheet_name='Data_Debitur')
    sheet_pinjaman = pd.read_excel(file_path, sheet_name='Riwayat_Pinjaman')
    
    # Merge dataset berdasarkan id_nasabah
    df_merged = pd.merge(sheet_debitur, sheet_pinjaman, on='id_nasabah')
    return df_merged

def preprocess_data(df):
    """Pembersihan data dan feature engineering sederhana."""
    df_clean = df.copy()
    
    # Imputasi missing values
    mode_pekerjaan = df_clean['status_pekerjaan'].mode()[0]
    df_clean['status_pekerjaan'] = df_clean['status_pekerjaan'].fillna(mode_pekerjaan)
    
    median_skor = df_clean['skor_kredit_internal'].median()
    df_clean['skor_kredit_internal'] = df_clean['skor_kredit_internal'].fillna(median_skor)
    
    # Feature engineering: Estimasi angsuran & rasio angsuran terhadap pendapatan
    df_clean['angsuran_estimasi'] = (df_clean['plafon_pinjaman'] / df_clean['tenor_bulan']) * 1.10
    df_clean['rasio_angsuran_pendapatan'] = np.round(df_clean['angsuran_estimasi'] / df_clean['pendapatan_bulanan'], 4)
    
    # One-hot encoding untuk variabel kategorikal
    df_encoded = pd.get_dummies(df_clean, columns=['status_pekerjaan'], drop_first=True)
    return df_encoded

def train_and_evaluate(df):
    """Pelatihan model Random Forest dan evaluasi metrik."""
    feature_cols = [
        'usia', 'pendapatan_bulanan', 'jumlah_tanggungan', 'skor_kredit_internal',
        'plafon_pinjaman', 'tenor_bulan', 'rasio_dti', 'jumlah_keterlambatan',
        'angsuran_estimasi', 'rasio_angsuran_pendapatan',
        'status_pekerjaan_Karyawan Swasta', 'status_pekerjaan_PNS', 'status_pekerjaan_Wiraswasta'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['status_gagal_bayar']
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training model dengan handle imbalanced class
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        class_weight='balanced', 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("=== Evaluasi Model ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Lancar (0)', 'Default (1)']))
    
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC Score  : {average_precision_score(y_test, y_prob):.4f}\n")
    
    print("=== Feature Importance ===")
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for col, imp in importances.head(5).items():
        print(f"{col:<30}: {imp:.4f}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "dataset_kredit.xlsx")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
        
    df = load_data(file_path)
    df_processed = preprocess_data(df)
    train_and_evaluate(df_processed)
