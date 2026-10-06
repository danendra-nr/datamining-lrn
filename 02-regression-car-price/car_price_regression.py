"""
Supervised Learning: Prediksi Harga Mobil Bekas (Regresi)
Eksperimen prediksi harga mobil bekas menggunakan Linear Regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data buatan spesifikasi mobil
np.random.seed(42)
n = 150
tahun = np.random.randint(2010, 2023, n)
jarak_tempuh = np.random.randint(10000, 150000, n)
tenaga_mesin = np.random.randint(100, 350, n)
harga = (tahun - 2000) * 1200 + tenaga_mesin * 180 - jarak_tempuh * 0.08 + np.random.normal(0, 3000, n)

df = pd.DataFrame({'tahun': tahun, 'jarak_tempuh': jarak_tempuh, 'tenaga_mesin': tenaga_mesin, 'harga': harga})

X = df[['tahun', 'jarak_tempuh', 'tenaga_mesin']]
y = df['harga']

# Bagi data train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

prediksi = model.predict(X_test)
print(f"Skor R2: {r2_score(y_test, prediksi):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, prediksi)):.2f}")
