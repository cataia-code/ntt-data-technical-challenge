"""Modelo global 'pooled' de forecasting: un solo modelo ML entrenado sobre todas las series.

A diferencia de los modelos clásicos por-serie (statsmodels), este comparte señal entre los ~53
países apilando todas las series (~1.4k observaciones) y usando el país y el tipo de café como
features categóricas. Usa HistGradientBoostingRegressor (scikit-learn, sin dependencias extra).

Detalle clave: los árboles NO extrapolan fuera del rango visto en entrenamiento, así que predecir el
*nivel* de una serie creciente falla. Por eso el objetivo es la **tasa de crecimiento en log**
(diferencia log año-a-año, aproximadamente estacionaria); el nivel se reconstruye acumulando las
tasas predichas. Es el enfoque ML 'moderno' defendible para este dataset — sin deep learning, que
sobreajustaría con ~30 observaciones por serie.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecasting import mape, predict, select_model, DEFAULT_HORIZON

GLAGS = [1, 2, 3]
ROLL_WINDOWS = [3, 5]

FEATURE_COLS = (
    [f"glag{lag}" for lag in GLAGS]
    + [f"groll_mean{w}" for w in ROLL_WINDOWS]
    + [f"groll_std{w}" for w in ROLL_WINDOWS]
    + ["level1", "year_idx", "Country", "Coffee type"]
)
CAT_COLS = ["Country", "Coffee type"]


def _series_features(ylog: list, year_idx: int, country: str, ctype: str) -> dict:
    """Features para predecir la tasa de crecimiento log del siguiente año, dada la historia log."""
    glog = list(np.diff(ylog))
    feat = {"level1": ylog[-1], "year_idx": year_idx, "Country": country, "Coffee type": ctype}
    for lag in GLAGS:
        feat[f"glag{lag}"] = glog[-lag] if len(glog) >= lag else 0.0
    for w in ROLL_WINDOWS:
        window = glog[-w:] if len(glog) >= 1 else [0.0]
        feat[f"groll_mean{w}"] = float(np.mean(window)) if window else 0.0
        feat[f"groll_std{w}"] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    return feat


def _build_supervised(long_df: pd.DataFrame) -> pd.DataFrame:
    """Matriz supervisada: una fila por país-año con target = tasa de crecimiento log del año."""
    valid = long_df[long_df["is_valid_series"]].sort_values(["Country", "fiscal_year_start"])
    min_year = valid["fiscal_year_start"].min()

    rows = []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        ylog = np.log1p(g["consumption"].astype(float).to_numpy())
        years = g["fiscal_year_start"].to_numpy()
        ctype = g["Coffee type"].iloc[0]
        # Predecir el crecimiento del año t requiere historia hasta t-1 (mínimo 4 puntos por los lags).
        for t in range(max(GLAGS) + 1, len(ylog)):
            feat = _series_features(list(ylog[:t]), int(years[t] - min_year), country, ctype)
            feat["target_glog"] = float(ylog[t] - ylog[t - 1])
            rows.append(feat)
    return pd.DataFrame(rows)


def _fit(sup: pd.DataFrame) -> HistGradientBoostingRegressor:
    X = sup[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
        categorical_features=CAT_COLS, random_state=42,
    )
    model.fit(X, sup["target_glog"])
    return model


def _recursive_forecast(model, ylog_hist, ctype, country, min_year, next_year, horizon):
    ylog = list(ylog_hist)
    preds = []
    for h in range(horizon):
        feat = _series_features(ylog, int(next_year + h - min_year), country, ctype)
        X = pd.DataFrame([feat])[FEATURE_COLS]
        for c in CAT_COLS:
            X[c] = X[c].astype("category")
        g_hat = float(model.predict(X)[0])
        new_ylog = ylog[-1] + g_hat
        ylog.append(new_ylog)
        preds.append(np.expm1(new_ylog))
    return np.array(preds)


def evaluate_pooled(long_df: pd.DataFrame, holdout: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Backtest: entrena hasta last_year - holdout y predice recursivo el holdout por país."""
    valid = long_df[long_df["is_valid_series"]]
    min_year = valid["fiscal_year_start"].min()
    last_year = valid["fiscal_year_start"].max()
    cutoff = last_year - holdout

    sup = _build_supervised(valid[valid["fiscal_year_start"] <= cutoff])
    model = _fit(sup)

    rows = []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        hist = g[g["fiscal_year_start"] <= cutoff]["consumption"].astype(float).to_numpy()
        actual = g[g["fiscal_year_start"] > cutoff]["consumption"].astype(float).to_numpy()
        if len(hist) <= max(GLAGS) + 1 or len(actual) == 0:
            continue
        preds = _recursive_forecast(model, np.log1p(hist), g["Coffee type"].iloc[0], country,
                                    min_year, cutoff + 1, len(actual))
        rows.append({"Country": country, "pooled_mape": mape(actual, preds)})
    return pd.DataFrame(rows)


def forecast_pooled(long_df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Entrena con toda la historia y proyecta `horizon` años por país (forecast recursivo)."""
    valid = long_df[long_df["is_valid_series"]]
    min_year = valid["fiscal_year_start"].min()
    last_year = valid["fiscal_year_start"].max()
    model = _fit(_build_supervised(valid))

    rows = []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        hist = g["consumption"].astype(float).to_numpy()
        preds = _recursive_forecast(model, np.log1p(hist), g["Coffee type"].iloc[0], country,
                                    min_year, last_year + 1, horizon)
        rows.append({"Country": country, "pooled_forecast_final": float(preds[-1])})
    return pd.DataFrame(rows)


def holdout_comparison(long_df: pd.DataFrame, holdout: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Comparación JUSTA: MAPE a `holdout` años vista, mismo holdout, por país.

    Para cada país compara el modelo clásico por-serie (seleccionado por rolling backtest sobre el
    tramo de entrenamiento) contra el modelo global pooled. Ambos predicen los mismos 5 años finales.
    """
    valid = long_df[long_df["is_valid_series"]]
    min_year = valid["fiscal_year_start"].min()
    last_year = valid["fiscal_year_start"].max()
    cutoff = last_year - holdout

    model = _fit(_build_supervised(valid[valid["fiscal_year_start"] <= cutoff]))

    rows = []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        hist = g[g["fiscal_year_start"] <= cutoff]["consumption"].astype(float).to_numpy()
        actual = g[g["fiscal_year_start"] > cutoff]["consumption"].astype(float).to_numpy()
        if len(hist) <= max(GLAGS) + 1 or len(actual) == 0:
            continue
        cls_model, _ = select_model(hist, horizon=1)
        cls_pred = predict(cls_model, hist, len(actual))
        pooled_pred = _recursive_forecast(model, np.log1p(hist), g["Coffee type"].iloc[0], country,
                                          min_year, cutoff + 1, len(actual))
        rows.append({"Country": country, "classic_model": cls_model,
                     "classic_mape": mape(actual, cls_pred), "pooled_mape": mape(actual, pooled_pred)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_prep import build_long_dataset

    long_df = build_long_dataset()
    cmp = holdout_comparison(long_df)
    print(f"Holdout {5} años (comparación justa, n={len(cmp)}):")
    print(f"  Clásico por-serie — MAPE mediano: {cmp['classic_mape'].median():.2f}% "
          f"| medio: {cmp['classic_mape'].mean():.2f}%")
    print(f"  Pooled GBM global — MAPE mediano: {cmp['pooled_mape'].median():.2f}% "
          f"| medio: {cmp['pooled_mape'].mean():.2f}%")
