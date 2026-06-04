import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def evaluar_modelo_estratificado(X, y, n_splits: int) -> dict:
    """
    Evalúa un modelo de regresión logística con validación cruzada estratificada.

    Pasos por fold:
    1. Separar train/test con StratifiedKFold.
    2. Escalar con StandardScaler (fit en train, transform en ambos).
    3. Entrenar LogisticRegression.
    4. Calcular accuracy y F1-score.

    Retorna:
        dict con 'accuracy_promedio' y 'f1_promedio'.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Escalar: fit solo sobre train
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modelo
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="binary", zero_division=0))

    return {
        "accuracy_promedio": float(np.mean(accuracies)),
        "f1_promedio": float(np.mean(f1_scores)),
    }


# --- Prueba rápida ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=5, weights=[0.8, 0.2], random_state=0
    )
    resultado = evaluar_modelo_estratificado(X, y, n_splits=5)
    print(resultado)
