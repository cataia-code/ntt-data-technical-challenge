"""Técnicas ML adicionales: detección de anomalías, clasificación de preferencia y PCA.

Complementan el forecasting para demostrar un uso más amplio de Machine Learning sobre este dataset,
siempre con métodos apropiados al tamaño (n=53 países) y reportando incertidumbre de forma honesta.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = ["cagr_full", "cagr_recent", "volatility", "trend_slope_recent", "log_level_mean"]
CLASSIFIER_FEATURES = ["log_level_mean", "cagr_full", "cagr_recent", "volatility", "share_of_global"]


# --------------------------------------------------------------------------- #
# 1. Detección de anomalías multivariante (IsolationForest)
# --------------------------------------------------------------------------- #
def detect_anomalies(feats: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    """Marca países con patrón de consumo atípico (multivariante) usando IsolationForest.

    Reemplaza el z-score univariante (que inflaba falsos positivos) por un criterio multivariante
    entrenado sobre las features de crecimiento/nivel/volatilidad.
    """
    X = StandardScaler().fit_transform(feats[ANOMALY_FEATURES].fillna(0))
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    scores = iso.score_samples(X)
    out = feats[["Country", "Coffee type"] + ANOMALY_FEATURES].copy()
    out["anomaly"] = labels == -1
    out["anomaly_score"] = scores
    return out.sort_values("anomaly_score").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Clasificador de preferencia de tipo de café dominante
# --------------------------------------------------------------------------- #
def classify_coffee_preference(feats: pd.DataFrame) -> dict:
    """Predice si un país es Arabica-dominante desde su patrón de consumo (supervisado).

    Devuelve accuracy con cross-validation (honesto sobre el n pequeño) e importancia de features.
    El valor es exploratorio: ¿el patrón de consumo lleva información sobre la preferencia de tipo?
    """
    df = feats.dropna(subset=CLASSIFIER_FEATURES).copy()
    X, y = df[CLASSIFIER_FEATURES].values, df["arabica_dominant"].values
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=CLASSIFIER_FEATURES).sort_values(ascending=False)
    baseline = max(y.mean(), 1 - y.mean())  # accuracy de la clase mayoritaria
    return {"cv_accuracy_mean": float(cv.mean()), "cv_accuracy_std": float(cv.std()),
            "majority_baseline": float(baseline), "feature_importances": importances}


# --------------------------------------------------------------------------- #
# 3. PCA de las features de país (para visualización de clusters)
# --------------------------------------------------------------------------- #
def pca_2d(feats: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Proyecta las features a 2D con PCA (previa estandarización). Devuelve PC1, PC2 y varianza."""
    X = StandardScaler().fit_transform(feats[feature_cols].fillna(feats[feature_cols].median()))
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X)
    out = feats[["Country", "Coffee type"]].copy()
    out["pc1"], out["pc2"] = comps[:, 0], comps[:, 1]
    out.attrs["explained_variance"] = pca.explained_variance_ratio_.tolist()
    return out


if __name__ == "__main__":
    from data_prep import build_long_dataset
    from features import build_and_save

    long_df = build_long_dataset()
    feats = build_and_save(long_df)

    anomalies = detect_anomalies(feats)
    print("Anomalías:", anomalies[anomalies["anomaly"]]["Country"].tolist())

    clf = classify_coffee_preference(feats)
    print(f"Clasificador tipo — CV acc: {clf['cv_accuracy_mean']:.2f} "
          f"(baseline {clf['majority_baseline']:.2f})")
    print(clf["feature_importances"].round(3).to_dict())
