import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_resumir_por_grupo():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función resumir_por_grupo(df, col_grupo, col_valor).
    """

    # 1. Grupos aleatorios
    grupos_posibles = ['A', 'B', 'C', 'D', 'E']
    n_grupos = random.randint(2, 4)
    grupos = random.sample(grupos_posibles, n_grupos)

    # 2. Filas por grupo
    n_filas_total = random.randint(15, 35)
    categorias = np.random.choice(grupos, size=n_filas_total)

    # 3. Columna numérica aleatoria
    valores = np.random.uniform(10, 500, size=n_filas_total).round(2)

    # 4. Columnas extra (distractor)
    extra = np.random.uniform(0, 100, size=n_filas_total).round(2)

    col_grupo = 'categoria'
    col_valor = 'valor'

    df = pd.DataFrame({
        col_grupo: categorias,
        col_valor: valores,
        'extra': extra
    })

    # --- INPUT ---
    input_data = {
        'df': df.copy(),
        'col_grupo': col_grupo,
        'col_valor': col_valor
    }

    # --- OUTPUT esperado ---
    grouped = df.groupby(col_grupo)[col_valor].agg(
        media='mean',
        mediana='median',
        std='std',
        conteo='count'
    ).reset_index(drop=True)

    # Necesitamos conservar los grupos como índice primero, luego resetear
    grouped = df.groupby(col_grupo)[col_valor].agg(
        media='mean',
        mediana='median',
        std='std',
        conteo='count'
    )
    grouped = grouped.reset_index(drop=True)
    grouped = grouped.sort_values('media', ascending=False).reset_index(drop=True)

    output_data = grouped

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_resumir_por_grupo()

    print("=== INPUT ===")
    print(f"Columna grupo: {entrada['col_grupo']}")
    print(f"Columna valor: {entrada['col_valor']}")
    print("DataFrame:")
    print(entrada['df'].head(10))

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
