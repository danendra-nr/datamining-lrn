# 06 Credit Risk Assessment

Eksperimen memprediksi risiko gagal bayar (loan default) nasabah menggunakan dataset Excel multi-sheet.

## Fitur & Pembelajaran
- Import dan merge data Excel multi-sheet (`Data_Debitur` & `Riwayat_Pinjaman`).
- Imputasi missing values dan feature engineering (rasio angsuran terhadap pendapatan).
- Handling imbalanced data pakai `class_weight='balanced'`.
- Evaluasi model Random Forest dengan Confusion Matrix, ROC-AUC, dan Feature Importance.

## Berkas
- `dataset_kredit.xlsx`: Dataset sampel 2 sheet.
- `credit_risk_assessment.py`: Script Python utama.
- `credit_risk_assessment.ipynb`: Notebook eksperimen.
