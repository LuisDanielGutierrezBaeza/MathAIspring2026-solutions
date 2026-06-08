import numpy as np

# 1. Simulación de Conteo de Parámetros (Subtask III)
def calcular_parametros():
    # Capa Totalmente Conectada (Dense)
    entradas_dense = 784
    salidas_dense = 128
    pesos_dense = entradas_dense * salidas_dense
    biases_dense = salidas_dense
    total_dense = pesos_dense + biases_dense
    
    # Capa Convolucional (Conv2D) - 32 filtros de 3x3, entrada de 1 canal
    filtros = 32
    tamano_kernel = 3 * 3
    canales_entrada = 1
    pesos_conv = filtros * tamano_kernel * canales_entrada
    biases_conv = filtros
    total_conv = pesos_conv + biases_conv
    
    print("=== SUBTASK III: CONTEO DE PARÁMETROS ===")
    print(f"Total Parámetros Capa Densa (784 -> 128): {total_dense}")
    print(f"Total Parámetros Capa Convolucional (32 filtros 3x3): {total_conv}")
    print(f"¡La capa convolucional reduce drásticamente los parámetros gracias al compartir pesos!")

# 2. Simulación conceptual de la pérdida ELBO en un VAE
def simular_elbo():
    print("\n=== SUBTASK IV: SIMULACIÓN DE COMPONENTES ELBO ===")
    epochs = 5
    recon_loss = [0.5, 0.3, 0.2, 0.15, 0.12]
    kl_div = [0.01, 0.05, 0.08, 0.10, 0.11]
    
    for epoch in range(epochs):
        # ELBO = - PÉRDIDA_RECONSTRUCCIÓN - DIVERGENCIA_KL
        elbo = - recon_loss[epoch] - kl_div[epoch]
        print(f"Época {epoch+1} | Pérdida Reconstrucción: {recon_loss[epoch]:.2f} | KL Divergencia: {kl_div[epoch]:.2f} | ELBO: {elbo:.2f}")

calcular_parametros()
simular_elbo()