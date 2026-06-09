import numpy as np

A = np.array([
    [1, 2],
    [0, 1],
    [1, 0]
], dtype=float)

print("=== MATRIZ ORIGINAL A ===")
print(A)

S = np.dot(A.T, A)
print("\n=== MATRIZ S = A^T * A ===")
print(S)

es_simetrica = np.allclose(S, S.T)
print(f"¿La matriz S es simétrica?: {es_simetrica}")

autovalores, autovectores = np.linalg.eigh(S)
print("\n=== AUTOVALORES DE S ===")
print(autovalores)
print("\n=== AUTOVECTORES DE S ===")
print(autovectores)

U, s, Vt = np.linalg.svd(A, full_matrices=True)

print("\n=== DESCOMPOSICIÓN SVD DE A ===")
print("Matriz U:\n", U)
print("\nValores Singulares (Sigma):\n", s)
print("\nMatriz V^T:\n", Vt)
