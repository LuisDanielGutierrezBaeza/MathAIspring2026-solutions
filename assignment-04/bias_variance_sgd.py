import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generar datos sintéticos no lineales (Función seno + Ruido)
np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 5, axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 2. Simular el Bias-Variance Tradeoff con diferentes grados polinomiales
grados = [1, 3, 15] # Grado 1 (Alto Sesgo), Grado 3 (Óptimo), Grado 15 (Alta Varianza)
X_test = np.linspace(0, 5, 100)[:, np.newaxis]

plt.figure(figsize=(12, 4))
for i, grado in enumerate(grados):
    ax = plt.subplot(1, 3, i + 1)
    pipeline = make_pipeline(PolynomialFeatures(grado), LinearRegression())
    pipeline.fit(X, y)
    
    ax.plot(X_test, pipeline.predict(X_test), label="Modelo")
    ax.plot(X_test, np.sin(X_test), label="Función Real", linestyle="--")
    ax.scatter(X, y, edgecolor='b', s=20, label="Datos + Ruido")
    ax.set_title(f"Grado {grado}\n" + 
                 ("Alto Sesgo (Underfitting)" if grado==1 else 
                  "Modelo Óptimo" if grado==3 else "Alta Varianza (Overfitting)"))
    ax.legend(loc="best")

print("=== SCRIPT GENERADO CON ÉXITO ===")
print("El gráfico muestra cómo cambia el sesgo y la varianza según la complejidad del modelo.")
plt.tight_layout()
plt.show()