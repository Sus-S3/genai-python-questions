import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preparar_datos(df: pd.DataFrame, target_col: str):
    """
    Prepara datos operativos de un puerto para un modelo de Gradient Boosting.

    Pasos:
    1. Separa X (características) e y (objetivo).
    2. Imputa valores nulos en X con la media de cada columna (SimpleImputer).
    3. Escala X con StandardScaler (media 0, varianza 1).

    Retorna:
        (X, y): tupla de (numpy.ndarray, numpy.ndarray o pd.Series)
    """
    # 1. Separar características y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # 2. Imputar NaN con la media
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # 3. Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y


# --- Prueba rápida ---
if __name__ == "__main__":
    df = pd.DataFrame({
        "tonelaje_total":  [12000, np.nan, 9500,  11000, 8700],
        "numero_gruas":    [4,     3,      np.nan, 5,    2   ],
        "teus_capacidad":  [800,   650,    720,    np.nan, 580],
        "clima_viento":    [15,    22,     18,     10,   25  ],
        "horas_en_puerto": [8.5,   12.0,   7.0,    10.5, 14.0],
    })
    X, y = preparar_datos(df, target_col="horas_en_puerto")
    print("X shape:", X.shape)
    print("y:", y)
    print("X media (≈0):", X.mean(axis=0).round(10))
