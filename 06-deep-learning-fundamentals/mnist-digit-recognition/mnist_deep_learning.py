"""
Deep Learning: Klasifikasi Angka Tulisan Tangan (MNIST Sederhana)
Pengenalan angka 0-9 menggunakan Multi-Layer Perceptron (MLP) dengan Scikit-Learn.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Dataset gambar digit (8x8 pixel)
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Neural Network MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
mlp.fit(X_train, y_train)

prediksi = mlp.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

print(f"Akurasi Model Neural Network (MLP): {akurasi:.4f}")
