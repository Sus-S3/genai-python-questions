import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def buscar_vecinos_textuales(df: pd.DataFrame, text_col: str, query: str, k: int = 3) -> pd.DataFrame:
    """
    Encuentra los k documentos más similares a una query usando TF-IDF + NearestNeighbors coseno.

    Pasos:
    1. Vectoriza los textos del DataFrame con TfidfVectorizer.
    2. Transforma la query con el mismo vectorizador (ya ajustado).
    3. Busca los k vecinos más cercanos con métrica coseno.
    4. Retorna las filas correspondientes del DataFrame original.
    """
    textos = df[text_col].tolist()

    # 1. Vectorizar corpus
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(textos)

    # 2. Transformar query
    query_vec = vectorizer.transform([query])

    # 3. Buscar vecinos más cercanos
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_tfidf)
    _, indices = nn.kneighbors(query_vec)

    # 4. Aplanar índices con np.ravel y retornar filas
    idx = np.ravel(indices)
    return df.iloc[idx].reset_index(drop=False)  # conserva índice original como columna opcional


# --- Prueba rápida ---
if __name__ == "__main__":
    data = {
        "id": [1, 2, 3, 4, 5],
        "texto": [
            "machine learning con redes neuronales",
            "pandas y análisis de datos en python",
            "deep learning para visión por computadora",
            "limpieza de datos con pandas y numpy",
            "clasificación con sklearn y regresión logística",
        ],
    }
    df = pd.DataFrame(data)
    resultado = buscar_vecinos_textuales(df, text_col="texto", query="análisis de datos pandas", k=2)
    print(resultado)
