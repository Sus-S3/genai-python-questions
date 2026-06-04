import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clasificar_frentes_mineros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame de mediciones mineras, clasifica el nivel de riesgo
    sobre los valores originales y escala las columnas numéricas con MinMaxScaler.

    Pasos:
    1. Eliminar duplicados y filas con nulos.
    2. Clasificar nivel_riesgo sobre valores originales.
    3. Ordenar por horas_operacion original ascendente.
    4. Escalar columnas numéricas con MinMaxScaler.
    5. Reiniciar índice.
    """
    columnas = ["polvo_aire", "gases_toxicos", "temperatura_tunel", "ruido_db", "horas_operacion"]

    # 1. Limpieza
    df_clean = df.drop_duplicates().dropna().reset_index(drop=True)

    # 2. Clasificar riesgo sobre valores ORIGINALES
    condiciones = [
        (df_clean["gases_toxicos"] >= 50) | (df_clean["temperatura_tunel"] >= 38),
        (df_clean["polvo_aire"] >= 5) | (df_clean["ruido_db"] >= 90),
    ]
    niveles = ["peligroso", "precaucion"]
    df_clean["nivel_riesgo"] = np.select(condiciones, niveles, default="seguro")

    # 3. Ordenar por horas_operacion original ANTES de escalar
    df_clean = df_clean.sort_values("horas_operacion").reset_index(drop=True)

    # 4. Escalar columnas numéricas
    scaler = MinMaxScaler()
    df_clean[columnas] = scaler.fit_transform(df_clean[columnas])

    return df_clean


# --- Prueba rápida ---
if __name__ == "__main__":
    df = pd.DataFrame({
        "polvo_aire":        [3.2, 6.5, 6.5, 2.1, None, 4.8],
        "gases_toxicos":     [20,  35,  35,  55,  28,   15 ],
        "temperatura_tunel": [30,  33,  33,  40,  36,   29 ],
        "ruido_db":          [85,  95,  95,  88,  82,   91 ],
        "horas_operacion":   [4,   2,   2,   7,   5,    6  ]
    })
    resultado = clasificar_frentes_mineros(df)
    print(resultado.to_string())
