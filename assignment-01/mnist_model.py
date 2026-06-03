import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd

# 1. Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar las imágenes para que los píxeles estén entre 0 y 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# 2. Función para entrenar la arquitectura y medir precisión
def evaluar_red(dim_hl_1, dim_hl_2, epochs):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(dim_hl_1, activation='relu'),
        Dense(dim_hl_2, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Entrenar el modelo de forma silenciosa
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
    
    # Evaluar con los datos de prueba
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc

# 3. Correr los experimentos para tu tabla del reporte
experimentos = [
    {"dim_hl_1": 64,  "dim_hl_2": 32,  "epochs": 5},
    {"dim_hl_1": 128, "dim_hl_2": 64,  "epochs": 5},
    {"dim_hl_1": 256, "dim_hl_2": 128, "epochs": 10},
]

resultados = []
for exp in experimentos:
    print(f"Entrenando configuración: HL1={exp['dim_hl_1']}, HL2={exp['dim_hl_2']}, Épocas={exp['epochs']}...")
    acc = evaluar_red(exp['dim_hl_1'], exp['dim_hl_2'], exp['epochs'])
    resultados.append({
        "Configuración (HL1, HL2)": f"({exp['dim_hl_1']}, {exp['dim_hl_2']})",
        "Épocas": exp['epochs'],
        "Test Accuracy": f"{acc * 100:.2f}%"
    })

# Imprimir la tabla final estructurada
df_res = pd.DataFrame(resultados)
print("\n=== TABLA DE RESULTADOS PARA TU REPORTE ===")
print(df_res.to_string(index=False))