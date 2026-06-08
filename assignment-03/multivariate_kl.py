import numpy as np

# --- Task I: Simulación de Matriz de Covarianza S = A^T * A ---
A = np.array([[2, 1], 
              [1, 3]], dtype=float)

# S es simétrica y semidefinida positiva (Válida como matriz de covarianza)
S = np.dot(A.T, A)
autovalores, autovectores = np.linalg.eigh(S)

print("=== TASK I: MATRIZ DE COVARIANZA S ===")
print(S)
print("\nAutovalores (Magnitud de los ejes del elipsoide):", autovalores)
print("Dirección de los ejes (Autovectores):\n", autovectores)


# --- Task II: Evaluación de la Divergencia KL (Caso Bernoulli) ---
def kl_bernoulli(p, q):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

p_test = 0.3
kl_left = kl_bernoulli(p_test, 1 - p_test)
kl_right = kl_bernoulli(1 - p_test, p_test)

print("\n=== TASK II: VERIFICACIÓN BERNOULLI ===")
print(f"D_KL(B_p || B_(1-p)): {kl_left:.4f}")
print(f"D_KL(B_(1-p) || B_p): {kl_right:.4f}")
print(f"¿Son iguales?: {np.isclose(kl_left, kl_right)}")