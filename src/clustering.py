"""Segmentación de países en 3 vistas complementarias (consumo, preferencia de tipo, forecast)."""

import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Vista (i): patrón de consumo/crecimiento (niveles en log para no dejar que la escala domine).
CONSUMPTION_FEATURES = [
    "log_level_mean", "cagr_full", "cagr_recent", "volatility", "trend_slope_recent", "share_of_global",
]
# Vista (ii): preferencia de tipo de café + nivel/crecimiento.
PREFERENCE_FEATURES = ["arabica_dominant", "log_level_mean", "cagr_recent"]
# Vista (iii): trayectoria de forecast (requiere columnas proj_* de attach_forecast_features).
FORECAST_FEATURES = ["proj_cagr", "proj_level"]


def _matrix(feats: pd.DataFrame, cols: list):
    clean = feats.dropna(subset=cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(clean[cols])
    return X, clean


def select_k(feats: pd.DataFrame, cols: list, k_range=range(3, 7)) -> pd.DataFrame:
    X, _ = _matrix(feats, cols)
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        rows.append({"k": k, "silhouette": round(silhouette_score(X, labels), 4)})
    return pd.DataFrame(rows)


def cluster_view(feats: pd.DataFrame, cols: list, k: int = 4, method: str = "kmeans",
                 label_col: str = "cluster") -> pd.DataFrame:
    """Corre una vista de clustering y devuelve el DataFrame con la etiqueta asignada."""
    X, clean = _matrix(feats, cols)
    if method == "kmeans":
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    else:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    clean = clean.copy()
    clean[label_col] = labels
    clean.attrs["silhouette"] = round(silhouette_score(X, labels), 4)
    return clean


def linkage_matrix(feats: pd.DataFrame, cols: list = None):
    """Matriz de linkage (Ward) para dibujar el dendrograma de la vista de consumo."""
    cols = cols or CONSUMPTION_FEATURES
    X, clean = _matrix(feats, cols)
    return linkage(X, method="ward"), clean


def profile(clustered: pd.DataFrame, label_col: str, cols: list) -> pd.DataFrame:
    prof = clustered.groupby(label_col)[cols].mean().round(3)
    prof["n_paises"] = clustered.groupby(label_col).size()
    return prof


def label_consumption_clusters(prof: pd.DataFrame) -> dict:
    """Asigna 4 etiquetas de negocio DISTINTAS a los clusters de consumo (una por cluster).

    Asignación por prioridad y argmax para evitar colisiones: mayor nivel -> maduro de gran escala;
    del resto, mayor crecimiento -> emergente; del resto, mayor volatilidad -> volátil; el último ->
    estable / maduro medio.
    """
    remaining = list(prof.index)
    labels = {}

    def _assign(metric, name):
        if not remaining:
            return
        idx = prof.loc[remaining, metric].idxmax()
        labels[idx] = name
        remaining.remove(idx)

    _assign("log_level_mean", "Maduro de gran escala")
    _assign("cagr_recent", "Emergente de alto crecimiento")
    _assign("volatility", "Volátil / atípico")
    for idx in remaining:
        labels[idx] = "Estable / maduro medio"
    return labels


def label_clusters_by_profile(prof: pd.DataFrame, level_col: str, growth_col: str,
                              dominance_col: str = None) -> dict:
    """Etiquetas de negocio por tamaño/crecimiento medio del cluster (split por mediana) — reutilizable
    para cualquier vista de clustering, no solo consumo. Si se da `dominance_col` (fracción 0-1, ej.
    arabica_dominant), antepone la preferencia de tipo dominante del cluster.
    """
    lvl_med, gr_med = prof[level_col].median(), prof[growth_col].median()
    labels = {}
    for idx, row in prof.iterrows():
        size = "grande" if row[level_col] >= lvl_med else "pequeño"
        growth = "en crecimiento" if row[growth_col] >= gr_med else "estable"
        parts = [f"{size}, {growth}"]
        if dominance_col:
            parts.insert(0, "Arabica-dom." if row[dominance_col] >= 0.5 else "Robusta-dom.")
        labels[idx] = " · ".join(parts).capitalize()
    return labels


if __name__ == "__main__":
    from data_prep import build_long_dataset
    from features import build_and_save

    long_df = build_long_dataset()
    feats = build_and_save(long_df)
    print("Silhouette por k (consumo):")
    print(select_k(feats, CONSUMPTION_FEATURES).to_string(index=False))
    cl = cluster_view(feats, CONSUMPTION_FEATURES, k=4)
    prof = profile(cl, "cluster", CONSUMPTION_FEATURES)
    print(prof)
    print(label_consumption_clusters(prof))
