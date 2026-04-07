import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generar_caso_de_uso_evaluar_regresion_ridge():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_regresion_ridge(X_train, y_train, X_test, y_test, alpha).
    """

    # 1. Parámetros aleatorios
    n_train = random.randint(40, 100)
    n_test = random.randint(10, 30)
    n_features = random.randint(2, 6)
    alpha = random.choice([0.1, 0.5, 1.0, 5.0, 10.0])

    # 2. Datos de entrenamiento y prueba
    X_train = np.random.randn(n_train, n_features)
    coef_real = np.random.randn(n_features)
    y_train = X_train @ coef_real + np.random.randn(n_train) * 0.5

    X_test = np.random.randn(n_test, n_features)
    y_test = X_test @ coef_real + np.random.randn(n_test) * 0.5

    # --- INPUT ---
    input_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'alpha': alpha
    }

    # --- OUTPUT esperado ---
    modelo = Ridge(alpha=alpha)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    output_data = {
        'mae':  round(mean_absolute_error(y_test, y_pred), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        'r2':   round(r2_score(y_test, y_pred), 4)
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_regresion_ridge()

    print("=== INPUT ===")
    print(f"alpha: {entrada['alpha']}")
    print(f"X_train shape: {entrada['X_train'].shape}")
    print(f"X_test shape:  {entrada['X_test'].shape}")

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
