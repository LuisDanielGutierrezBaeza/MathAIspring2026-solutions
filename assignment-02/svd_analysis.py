import numpy as np

# 1. Definir la matriz A 
A = np.array([
    [1, 2],
    [0, 1],
    [1, 0]
], dtype=float)

print("=== MATRIZ ORIGINAL A ===")
print(A)

# 2. Calcular la matriz S = A^T * A
S = np.dot(A.T, A)
print("\n=== MATRIZ S = A^T * A ===")
print(S)

# Verificar si es simétrica
es_simetrica = np.allclose(S, S.T)
print(f"¿La matriz S es simétrica?: {es_simetrica}")

# 3. Calcular Autovalores y Autovectores de S
autovalores, autovectores = np.linalg.eigh(S)
print("\n=== AUTOVALORES DE S ===")
print(autovalores)
print("\n=== AUTOVECTORES DE S ===")
print(autovectores)

# 4. Calcular la SVD completa usando NumPy
U, s, Vt = np.linalg.svd(A, full_matrices=True)

print("\n=== DESCOMPOSICIÓN SVD DE A ===")
print("Matriz U:\n", U)
print("\nValores Singulares (Sigma):\n", s)
print("\nMatriz V^T:\n", Vt)