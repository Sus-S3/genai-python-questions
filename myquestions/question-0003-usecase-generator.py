import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier


def generar_caso_de_uso_seleccionar_features_importantes():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función seleccionar_features_importantes(X, y, k).
    """

    # 1. Parámetros aleatorios
    n_samples = random.randint(50, 120)
    n_features = random.randint(5, 12)
    k = random.randint(2, min(5, n_features - 1))
    n_clases = 2

    # 2. Datos de entrenamiento
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_clases, size=n_samples)

    # --- INPUT ---
    input_data = {
        'X': X,
        'y': y,
        'k': k
    }

    # --- OUTPUT esperado ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    importancias = clf.feature_importances_

    # Índices ordenados de mayor a menor importancia
    indices_ordenados = np.argsort(importancias)[::-1]
    indices_top_k = indices_ordenados[:k]
    importancias_top_k = importancias[indices_top_k]

    output_data = (indices_top_k, importancias_top_k)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_seleccionar_features_importantes()

    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"y shape: {entrada['y'].shape}")
    print(f"k: {entrada['k']}")

    print("\n=== OUTPUT ESPERADO ===")
    indices, importancias = salida_esperada
    print(f"Índices top-{entrada['k']}: {indices}")
    print(f"Importancias:              {importancias.round(4)}")
