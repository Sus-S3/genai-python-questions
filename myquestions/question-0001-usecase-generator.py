import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_limpiar_dataframe():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función limpiar_dataframe(df, umbral).
    """

    # 1. Dimensiones aleatorias
    n_rows = random.randint(8, 20)
    n_cols = random.randint(3, 6)
    col_names = [f'col_{i}' for i in range(n_cols)]

    # 2. Datos base aleatorios
    data = np.random.uniform(1, 100, size=(n_rows, n_cols)).round(2)
    df = pd.DataFrame(data, columns=col_names)

    # 3. Introducir NaNs aleatorios (~15% de los datos)
    for col in col_names:
        mask = np.random.choice([True, False], size=n_rows, p=[0.15, 0.85])
        df.loc[mask, col] = np.nan

    # 4. Introducir filas duplicadas (1 o 2)
    n_dups = random.randint(1, 2)
    dup_indices = np.random.choice(df.index, size=n_dups, replace=False)
    dup_rows = df.loc[dup_indices]
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # mezclar filas

    # 5. Umbral aleatorio entre 1 y n_cols-1
    umbral = random.randint(1, n_cols - 1)

    # --- INPUT ---
    input_data = {
        'df': df.copy(),
        'umbral': umbral
    }

    # --- OUTPUT esperado (replicamos la lógica de limpiar_dataframe) ---

    # Paso 1: eliminar duplicados
    result = df.drop_duplicates()

    # Paso 2: eliminar filas con más NaNs que el umbral
    result = result[result.isna().sum(axis=1) <= umbral]

    # Paso 3: rellenar NaNs restantes con el promedio de cada columna
    for col in result.columns:
        col_mean = result[col].mean()
        result[col] = result[col].fillna(col_mean)

    # Paso 4: reiniciar índice
    result = result.reset_index(drop=True)

    output_data = result

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_limpiar_dataframe()

    print("=== INPUT ===")
    print(f"Umbral: {entrada['umbral']}")
    print("DataFrame original:")
    print(entrada['df'])

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
