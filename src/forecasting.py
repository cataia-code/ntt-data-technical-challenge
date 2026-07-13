"""Forecasting de series anuales cortas (~30 obs) con backtest de origen móvil.

Mejoras de rigor sobre la v1:
- Selección de modelo por MAPE promedio en *rolling-origin* (varios orígenes), no un solo holdout.
- Modelos ajustables en escala log (crecimiento compuesto, intervalos que nunca caen bajo cero).
- Intervalos de predicción empíricos a partir de los residuos del backtest, para TODOS los modelos.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DEFAULT_HORIZON = 5
ROLLING_ORIGINS = 5          # nº de orígenes móviles para el backtest
MIN_TRAIN = 18               # mínimo de puntos de entrenamiento en el primer origen
# Grilla curada de órdenes bajos apropiados para series anuales cortas (evita la explosión de un
# grid completo: con ~25 puntos, órdenes altos sobreajustan y no aportan). Se elige por AIC.
ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 0, 0), (2, 0, 0)]

MODELS = ["naive", "linear", "holt_winters", "arima"]


# --------------------------------------------------------------------------- #
# Predictores puntuales (devuelven np.ndarray de longitud `horizon`)
# --------------------------------------------------------------------------- #
def _naive(train, horizon):
    return np.full(horizon, train[-1], dtype=float)


def _linear(train, horizon):
    x = np.arange(len(train))
    slope, intercept = np.polyfit(x, train, 1)
    fx = np.arange(len(train), len(train) + horizon)
    return slope * fx + intercept


def _holt_winters(train, horizon):
    # Ajuste en escala log (los valores son positivos y crecen de forma compuesta).
    ltrain = np.log(np.clip(train, 1.0, None))
    fit = ExponentialSmoothing(ltrain, trend="add", damped_trend=True,
                               initialization_method="estimated").fit()
    return np.exp(fit.forecast(horizon))


def _arima(train, horizon):
    ltrain = np.log(np.clip(train, 1.0, None))
    best_aic, best_fit = np.inf, None
    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(ltrain, order=order).fit()
            if fit.aic < best_aic:
                best_aic, best_fit = fit.aic, fit
        except Exception:
            continue
    if best_fit is None:
        return _naive(train, horizon)
    return np.exp(best_fit.get_forecast(horizon).predicted_mean)


PREDICTORS = {"naive": _naive, "linear": _linear, "holt_winters": _holt_winters, "arima": _arima}


def predict(model_name, train, horizon):
    return PREDICTORS[model_name](np.asarray(train, dtype=float), horizon)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


# --------------------------------------------------------------------------- #
# Backtest de origen móvil
# --------------------------------------------------------------------------- #
def rolling_backtest(values, horizon=1, origins=ROLLING_ORIGINS):
    """Evalúa cada modelo prediciendo `horizon` paso(s) desde varios orígenes móviles.

    Devuelve un DataFrame con MAPE/MAE promedio por modelo y guarda los residuos relativos
    (para construir intervalos empíricos) en df.attrs['rel_residuals'].
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    start = max(MIN_TRAIN, n - origins - horizon + 1)
    cut_points = [t for t in range(start, n - horizon + 1)]
    if not cut_points:
        cut_points = [n - horizon]

    per_model = {m: {"ape": [], "ae": [], "rel": []} for m in MODELS}
    for cut in cut_points:
        train, actual = values[:cut], values[cut:cut + horizon]
        for m in MODELS:
            try:
                pred = predict(m, train, horizon)
            except Exception:
                pred = _naive(train, horizon)
            per_model[m]["ape"].append(mape(actual, pred))
            per_model[m]["ae"].append(mae(actual, pred))
            with np.errstate(divide="ignore", invalid="ignore"):
                per_model[m]["rel"].extend(((pred - actual) / actual).tolist())

    rows, residuals = [], {}
    for m in MODELS:
        rows.append({"model": m,
                     "mape": float(np.nanmean(per_model[m]["ape"])),
                     "mae": float(np.nanmean(per_model[m]["ae"]))})
        residuals[m] = np.array(per_model[m]["rel"], dtype=float)
    out = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    out.attrs["rel_residuals"] = residuals
    return out


def select_model(values, horizon=1):
    bt = rolling_backtest(values, horizon=horizon)
    return bt.iloc[0]["model"], bt


def forecast_series(values, horizon=DEFAULT_HORIZON):
    """Selecciona modelo por rolling backtest, reentrena con todo y devuelve punto + banda empírica.

    Los intervalos usan los cuantiles de los residuos relativos del backtest del modelo ganador
    (conformal-lite, validado a 1 paso con orígenes móviles), por lo que aplican también a
    naive/linear. La banda se ensancha con `sqrt(paso)` a medida que el horizonte avanza — supuesto
    de paseo aleatorio (la varianza del error crece linealmente con el tiempo, el desvío con su raíz):
    sin esto, un forecast a 11 años mostraría la misma incertidumbre en el año 1 y en el año 11, que
    subestima el riesgo real de proyectar lejos del último dato observado.
    """
    values = np.asarray(values, dtype=float)
    model_name, bt = select_model(values, horizon=1)
    point = predict(model_name, values, horizon)
    naive_mape = float(bt.loc[bt["model"] == "naive", "mape"].iloc[0])

    rel = bt.attrs["rel_residuals"].get(model_name, np.array([]))
    rel = rel[np.isfinite(rel)]
    covered, coverage_total = 0, 0
    if rel.size >= 3:
        lo_q, hi_q = np.quantile(rel, [0.1, 0.9])
        step_scale = np.sqrt(np.arange(1, horizon + 1))
        # residual = (pred - actual)/actual  ->  actual = pred / (1 + residual)
        lower = point / (1 + hi_q * step_scale)
        upper = point / (1 + lo_q * step_scale)
        # Cobertura empírica leave-one-out del intervalo 80% nominal: por cada residuo, se reconstruye
        # el intervalo con los DEMÁS y se comprueba si el excluido cae dentro. Evita el sesgo optimista
        # de validar el intervalo con los mismos residuos que lo definieron.
        for i in range(rel.size):
            other = np.delete(rel, i)
            if other.size < 2:
                continue
            lo_i, hi_i = np.quantile(other, [0.1, 0.9])
            coverage_total += 1
            if lo_i <= rel[i] <= hi_i:
                covered += 1
    else:
        lower = upper = None

    return {"model": model_name, "point": point, "lower": lower, "upper": upper,
            "backtest": bt, "mape": float(bt.iloc[0]["mape"]), "mae": float(bt.iloc[0]["mae"]),
            "naive_mape": naive_mape, "coverage_covered": covered, "coverage_total": coverage_total}


# --------------------------------------------------------------------------- #
# Granularidades: global, por país, por tipo de café
# --------------------------------------------------------------------------- #
def _series_from(long_df, group_col, group_val):
    g = long_df[long_df[group_col] == group_val].groupby("fiscal_year_start")["consumption"].sum().sort_index()
    return g.index.to_numpy(), g.to_numpy(dtype=float)


def forecast_global(long_df, horizon=DEFAULT_HORIZON):
    valid = long_df[long_df["is_valid_series"]]
    s = valid.groupby("fiscal_year_start")["consumption"].sum().sort_index()
    res = forecast_series(s.to_numpy(), horizon)
    res["years"] = s.index.to_numpy()
    res["history"] = s.to_numpy(dtype=float)
    return res


def forecast_by_type(long_df, horizon=DEFAULT_HORIZON):
    valid = long_df[long_df["is_valid_series"]]
    out = {}
    for t in sorted(valid["Coffee type"].unique()):
        years, hist = _series_from(valid, "Coffee type", t)
        res = forecast_series(hist, horizon)
        res["years"], res["history"] = years, hist
        out[t] = res
    return out


def forecast_all_countries(long_df, horizon=DEFAULT_HORIZON):
    """Forecast por país (todas las series válidas). Devuelve dict y un resumen tabular."""
    valid = long_df[long_df["is_valid_series"]]
    results, rows = {}, []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        hist = g["consumption"].to_numpy(dtype=float)
        res = forecast_series(hist, horizon)
        res["years"] = g["fiscal_year_start"].to_numpy()
        res["history"] = hist
        results[country] = res
        rows.append({"series": country, "best_model": res["model"],
                     "backtest_mape": round(res["mape"], 2), "backtest_mae": round(res["mae"], 2),
                     "naive_mape": round(res["naive_mape"], 2),
                     "last_observed": hist[-1], "forecast_final": float(res["point"][-1])})
    summary = pd.DataFrame(rows).sort_values("last_observed", ascending=False).reset_index(drop=True)
    summary.attrs["horizon"] = horizon
    return results, summary
