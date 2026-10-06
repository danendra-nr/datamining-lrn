"""
Eksplorasi Data: Analisis Harga Rumah
Analisis eksplorasi data (EDA) sederhana untuk melihat faktor-faktor yang
mempengaruhi harga rumah seperti luas bangunan, jumlah kamar tidur, dan umur bangunan.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set tampilan plot
sns.set_theme(style="whitegrid")
print("Library berhasil diimpor.")

# Bikin data buatan untuk latihan EDA
np.random.seed(42)
n_samples = 200

luas_sqft = np.random.normal(2000, 500, n_samples).astype(int)
kamar_tidur = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.2, 0.5, 0.2, 0.1])
umur_tahun = np.random.randint(1, 30, n_samples)
harga = luas_sqft * 150 + kamar_tidur * 10000 - umur_tahun * 1200 + np.random.normal(0, 15000, n_samples)

df = pd.DataFrame({
    'luas_sqft': luas_sqft,
    'kamar_tidur': kamar_tidur,
    'umur_tahun': umur_tahun,
    'harga': harga.astype(int)
})

print("\n--- 5 Data Pertama ---")
print(df.head())

print("\n--- Ringkasan Statistik ---")
print(df.describe())

# Visualisasi heatmap korelasi antar variabel
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriks Korelasi Features")
plt.tight_layout()
plt.show()
