"""
Unsupervised Learning: Segmentasi Pelanggan Mall (K-Means)
Mengelompokkan pelanggan berdasarkan pendapatan tahunan dan skor pengeluaran.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Dataset sintetis 4 kelompok pelanggan
np.random.seed(42)
c1 = np.random.multivariate_normal([30, 20], [[10, 0], [0, 10]], 50)
c2 = np.random.multivariate_normal([30, 80], [[10, 0], [0, 10]], 50)
c3 = np.random.multivariate_normal([80, 20], [[10, 0], [0, 10]], 50)
c4 = np.random.multivariate_normal([80, 80], [[10, 0], [0, 10]], 50)
X = np.vstack([c1, c2, c3, c4])

df = pd.DataFrame(X, columns=['pendapatan_tahunan_k$', 'skor_pengeluaran_1-100'])

# Model K-Means dengan 4 kluster
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['kluster'] = kmeans.fit_predict(df)

print(f"Skor Silhouette: {silhouette_score(X, df['kluster']):.4f}")

# Visualisasi hasil clustering
plt.figure(figsize=(8, 5))
plt.scatter(df['pendapatan_tahunan_k$'], df['skor_pengeluaran_1-100'], c=df['kluster'], cmap='viridis')
plt.xlabel('Pendapatan Tahunan (k$)')
plt.ylabel('Skor Pengeluaran (1-100)')
plt.title('Hasil Segmentasi Pelanggan')
plt.tight_layout()
plt.show()
