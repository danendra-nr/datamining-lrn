"""
Supervised Learning: Klasifikasi Customer Churn
Memprediksi potensi churn pelanggan menggunakan Random Forest Classifier.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Bikin dataset sampel pelanggan
np.random.seed(42)
n = 300
lama_berlangganan = np.random.randint(1, 60, n)
biaya_bulanan = np.random.uniform(20, 120, n)
jumlah_keluhan = np.random.randint(0, 10, n)
prob_churn = 1 / (1 + np.exp(-(-0.05 * lama_berlangganan + 0.02 * biaya_bulanan + 0.4 * jumlah_keluhan - 1)))
churn = (np.random.rand(n) < prob_churn).astype(int)

df = pd.DataFrame({
    'lama_berlangganan': lama_berlangganan,
    'biaya_bulanan': biaya_bulanan,
    'jumlah_keluhan': jumlah_keluhan,
    'churn': churn
})

X = df[['lama_berlangganan', 'biaya_bulanan', 'jumlah_keluhan']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

prediksi = rf.predict(X_test)
probabilitas = rf.predict_proba(X_test)[:, 1]

print("Laporan Klasifikasi:")
print(classification_report(y_test, prediksi))
print(f"Skor ROC AUC: {roc_auc_score(y_test, probabilitas):.4f}")
