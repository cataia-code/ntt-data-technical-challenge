"""Vistas analíticas de negocio para el EDA narrativo: concentración, cuadrante estratégico,
contribución al cambio reciente, resumen por tipo de café, ranking priorizado y estadísticas
descriptivas. Todo se calcula a partir de `long_df` (+ `feats` para volatilidad), sin depender del
forecasting pesado — se puede recomputar en segundos.
"""

import numpy as np
import pandas as pd

RECENT_YEARS = 10


# --------------------------------------------------------------------------- #
# 0. Estadísticas descriptivas (media, mediana) — distribución del último año
# --------------------------------------------------------------------------- #
def descriptive_stats(long_df: pd.DataFrame, year: int = None) -> dict:
    """Media, mediana y dispersión del consumo por país en un año dado."""
    valid = long_df[long_df["is_valid_series"]]
    year = year or int(valid["fiscal_year_start"].max())
    s = valid[valid["fiscal_year_start"] == year].set_index("Country")["consumption"].astype(float)

    return {
        "year": year, "n": len(s), "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std()), "min": float(s.min()), "max": float(s.max()),
        "skew_right": bool(s.mean() > s.median() * 1.5),  # asimetría fuerte: media >> mediana
    }


# --------------------------------------------------------------------------- #
# 0b. Calidad y estructura del dataset crudo — primer diagnóstico de todo EDA
# --------------------------------------------------------------------------- #
def dataset_overview(raw_df: pd.DataFrame, long_df: pd.DataFrame) -> dict:
    """Shape, tipos de columna, nulos y duplicados del dataset original (formato wide)."""
    year_cols = [c for c in raw_df.columns if "/" in c]
    null_counts = raw_df.isnull().sum()
    zero_series = sorted(long_df.loc[~long_df["is_valid_series"], "Country"].unique().tolist())
    return {
        "n_rows": len(raw_df), "n_cols": len(raw_df.columns), "n_year_cols": len(year_cols),
        "year_range": f"{year_cols[0]}–{year_cols[-1]}",
        "n_nulls_total": int(null_counts.sum()),
        "cols_with_nulls": {c: int(v) for c, v in null_counts[null_counts > 0].items()},
        "n_duplicated_rows": int(raw_df.duplicated().sum()),
        "n_countries": int(raw_df["Country"].nunique()),
        "n_coffee_types": int(raw_df["Coffee type"].nunique()),
        "zero_series_countries": zero_series,
    }


# --------------------------------------------------------------------------- #
# 0c. Crecimiento global y quiebre interanual reciente
# --------------------------------------------------------------------------- #
def global_growth_summary(long_df: pd.DataFrame) -> dict:
    """CAGR global de la serie completa y variación del último año vs. el anterior (para detectar
    un quiebre de tendencia reciente)."""
    valid = long_df[long_df["is_valid_series"]]
    global_hist = valid.groupby("fiscal_year_start")["consumption"].sum().sort_index()
    last_val, prev_val = float(global_hist.iloc[-1]), float(global_hist.iloc[-2])
    n_years = int(global_hist.index[-1] - global_hist.index[0])
    cagr = (global_hist.iloc[-1] / global_hist.iloc[0]) ** (1 / n_years) - 1
    return {
        "cagr_global": float(cagr),
        "last_year_val": last_val,
        "prev_year_val": prev_val,
        "drop_pct": (last_val - prev_val) / prev_val * 100,
    }


# --------------------------------------------------------------------------- #
# 0d. Tamaño estructural por continente (mediana y # de mercados, no solo el país más grande)
# --------------------------------------------------------------------------- #
def continent_size_comparison(long_df: pd.DataFrame, year: int) -> dict:
    """Mediana de consumo y número de países con consumo > 0 por continente en `year` — compara
    tamaño estructural entre continentes más allá del país individual más grande."""
    valid = long_df[long_df["is_valid_series"]]
    d = valid[(valid["fiscal_year_start"] == year) & (valid["consumption"] > 0)]
    return {
        "median_by_continent": d.groupby("continent")["consumption"].median().sort_values(ascending=False),
        "count_by_continent": d.groupby("continent")["consumption"].count(),
    }


# --------------------------------------------------------------------------- #
# 0e. CAGR reciente de un país puntual (lookup con nombre de negocio, no un .loc suelto)
# --------------------------------------------------------------------------- #
def country_cagr_recent(feats: pd.DataFrame, country: str) -> float:
    return float(feats.loc[feats["Country"] == country, "cagr_recent"].iloc[0])


# --------------------------------------------------------------------------- #
# 1. Concentración: Pareto de países
# --------------------------------------------------------------------------- #
def pareto_table(long_df: pd.DataFrame) -> pd.DataFrame:
    valid = long_df[long_df["is_valid_series"]]
    totals = valid.groupby("Country")["consumption"].sum().sort_values(ascending=False)
    total_sum = totals.sum()
    df = totals.reset_index()
    df.columns = ["Country", "total"]
    df["share_pct"] = df["total"] / total_sum * 100
    df["cum_share_pct"] = df["share_pct"].cumsum()
    df.attrs["top5_share"] = float(df["share_pct"].head(5).sum())
    df.attrs["top10_share"] = float(df["share_pct"].head(10).sum())
    df.attrs["brazil_share"] = float(df.loc[df["Country"] == "Brazil", "share_pct"].iloc[0])
    return df


# --------------------------------------------------------------------------- #
# 2. Cuadrante estratégico (tamaño × crecimiento) con clasificación BCG-style
# --------------------------------------------------------------------------- #
def classify_quadrant(bubble_df: pd.DataFrame, x_col: str = "cagr_recent",
                      y_col: str = "level_last5_mean") -> pd.DataFrame:
    """Clasifica cada país en 4 cuadrantes por mediana de tamaño (y) y crecimiento reciente (x)."""
    df = bubble_df.dropna(subset=[x_col, y_col]).copy()
    x_med, y_med = df[x_col].median(), df[y_col].median()

    def _label(row):
        big, fast = row[y_col] >= y_med, row[x_col] >= x_med
        if big and fast:
            return "Priorizar"
        if big and not fast:
            return "Defender"
        if not big and fast:
            return "Explorar"
        return "Baja prioridad"

    df["quadrant"] = df.apply(_label, axis=1)
    df.attrs["x_median"] = float(x_med)
    df.attrs["y_median"] = float(y_med)
    return df


# --------------------------------------------------------------------------- #
# 3. Contribución al cambio 2018/19 -> 2019/20 (waterfall)
# --------------------------------------------------------------------------- #
def waterfall_contribution(long_df: pd.DataFrame, top_n: int = 8) -> dict:
    valid = long_df[long_df["is_valid_series"]]
    years = sorted(valid["fiscal_year_start"].unique())
    y0, y1 = years[-2], years[-1]
    piv = valid.pivot(index="Country", columns="fiscal_year_start", values="consumption")
    delta = (piv[y1] - piv[y0]).sort_values(key=lambda s: s.abs(), ascending=False)

    top = delta.head(top_n)
    rest_sum = float(delta.iloc[top_n:].sum())
    items = [(c, float(v)) for c, v in top.items()]
    if abs(rest_sum) > 0:
        items.append((f"Otros {len(delta) - top_n} países", rest_sum))

    return {
        "y0": int(y0), "y1": int(y1), "items": items,
        "total_delta": float(delta.sum()), "total_y0": float(piv[y0].sum()), "total_y1": float(piv[y1].sum()),
    }


# --------------------------------------------------------------------------- #
# 4. Resumen por tipo de café: participación, CAGR, # países, país dominante
# --------------------------------------------------------------------------- #
def type_summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    valid = long_df[long_df["is_valid_series"]]
    total_all = valid["consumption"].sum()
    rows = []
    for t, g in valid.groupby("Coffee type"):
        total_t = g["consumption"].sum()
        s = g.groupby("fiscal_year_start")["consumption"].sum().sort_index()
        last_year = s.index.max()
        recent = s[s.index >= last_year - RECENT_YEARS]
        if recent.iloc[0] > 0 and len(recent) > 1:
            cagr = (recent.iloc[-1] / recent.iloc[0]) ** (1 / (recent.index[-1] - recent.index[0])) - 1
        else:
            cagr = np.nan
        dominant = g.groupby("Country")["consumption"].sum().idxmax()
        rows.append({
            "Coffee type": t, "share_pct": total_t / total_all * 100, "cagr_recent": cagr,
            "n_countries": g["Country"].nunique(), "dominant_country": dominant,
        })
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Ranking priorizado (tabla final accionable)
# --------------------------------------------------------------------------- #
_QUADRANT_ORDER = {"Priorizar": 0, "Explorar": 1, "Defender": 2, "Baja prioridad": 3}


def priority_ranking_table(quad_df: pd.DataFrame, feats: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    df = quad_df.merge(feats[["Country", "volatility"]], on="Country", how="left")
    df["q_order"] = df["quadrant"].map(_QUADRANT_ORDER)
    df = df.sort_values(["q_order", "cagr_recent"], ascending=[True, False])
    cols = ["Country", "Coffee type", "level_last5_mean", "cagr_recent", "volatility", "quadrant"]
    return df[cols].head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Volatilidad / riesgo (para el bloque de anomalías del EDA)
# --------------------------------------------------------------------------- #
def volatility_ranking(feats: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    return feats.sort_values("volatility", ascending=False)[
        ["Country", "Coffee type", "volatility", "cagr_recent"]
    ].head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Mercados de alto potencial: crecimiento × tamaño, excluyendo el mega-mercado maduro
# --------------------------------------------------------------------------- #
def priority_markets_table(feats: pd.DataFrame, forecast_summary: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Ranking de mercados emergentes de alto potencial (crecimiento × tamaño, no saturados).

    `forecast_summary` debe tener columnas: series (=Country), last_observed, forecast_final
    (formato de `forecasting.forecast_all_countries`).
    """
    df = feats.merge(
        forecast_summary[["series", "forecast_final", "last_observed"]].rename(columns={"series": "Country"}),
        on="Country", how="left",
    )
    df["incremental_forecast"] = df["forecast_final"] - df["last_observed"]
    # Excluye el mega-mercado maduro (top 5% de nivel) y los de crecimiento negativo.
    level_cap = df["level_mean"].quantile(0.95)
    cand = df[(df["cagr_recent"] > 0) & (df["level_mean"] < level_cap)].copy()
    cand["potential_score"] = cand["cagr_recent"] * np.log1p(cand["level_mean"])
    cols = ["Country", "Coffee type", "level_mean", "cagr_recent", "incremental_forecast", "potential_score"]
    return cand.sort_values("potential_score", ascending=False)[cols].head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 8. Gobierno de modelos: qué serie, qué modelo ganó, contra qué baseline, con qué riesgo
# --------------------------------------------------------------------------- #
N_MODELS_EVALUATED = 4  # naive/linear/holt_winters/arima — ver forecasting.MODELS


def _mape_risk(mape: float) -> str:
    if mape < 2.0:
        return "Bajo"
    if mape < 5.0:
        return "Medio"
    return "Alto"


def model_governance_table(ml: dict, forecast_summary: pd.DataFrame, top_n: int = 8,
                           anomalous_countries: set = frozenset()) -> pd.DataFrame:
    """Tabla de gobierno de modelos: serie, modelo ganador, MAPE vs. baseline naive, motivo de
    selección y riesgo — el resumen que un revisor técnico necesita antes de confiar en un forecast.

    Cubre Global, los 4 tipos de café y los `top_n` países de mayor consumo. `ml` es el dict de
    ml_summary.json (con `global_backtest`/`type_backtests`); `forecast_summary` es el resumen por
    país de `forecasting.forecast_all_countries` (columnas: series, best_model, backtest_mape,
    naive_mape).
    """
    def _row(series_name, model, mape, naive_mape):
        return {
            "Serie": series_name, "Modelo": model,
            "MAPE backtest": round(float(mape), 2), "Baseline (naive)": round(float(naive_mape), 2),
            "Motivo de selección": f"Menor MAPE rolling-origin de {N_MODELS_EVALUATED} modelos evaluados",
            "Riesgo": _mape_risk(mape),
        }

    rows = []
    global_bt = pd.DataFrame(ml["global_backtest"])
    winner, naive = global_bt.iloc[0], global_bt.loc[global_bt["model"] == "naive"].iloc[0]
    rows.append(_row("Global", winner["model"], winner["mape"], naive["mape"]))

    for t, bt_records in ml["type_backtests"].items():
        bt = pd.DataFrame(bt_records)
        winner, naive = bt.iloc[0], bt.loc[bt["model"] == "naive"].iloc[0]
        rows.append(_row(f"Tipo: {t}", winner["model"], winner["mape"], naive["mape"]))

    for _, r in forecast_summary.head(top_n).iterrows():
        row = _row(r["series"], r["best_model"], r["backtest_mape"], r["naive_mape"])
        if r["series"] in anomalous_countries:
            row["Riesgo"] = row["Riesgo"] + " + atípico"
        rows.append(row)

    return pd.DataFrame(rows)


def best_worst_fit_countries(forecast_summary: pd.DataFrame) -> dict:
    """País con mejor y peor ajuste de backtest (MAPE) — ejemplos concretos de buen/mal ajuste."""
    best = forecast_summary.loc[forecast_summary["backtest_mape"].idxmin()]
    worst = forecast_summary.loc[forecast_summary["backtest_mape"].idxmax()]
    return {"best": best.to_dict(), "worst": worst.to_dict()}


# --------------------------------------------------------------------------- #
# 9. Segmentación accionable: tamaño, comportamiento y riesgo operacional por cluster
# --------------------------------------------------------------------------- #
_CLUSTER_RECOMMENDATIONS = {
    "Maduro de gran escala": "Defender participación, no expandir — el motor de crecimiento está en otro cluster.",
    "Emergente de alto crecimiento": "Priorizar inversión comercial; cruzar contra volatilidad antes de comprometer presupuesto grande.",
    "Volátil / atípico": "Tratar como alto riesgo: validar cualquier señal de crecimiento contra su historial antes de actuar.",
    "Estable / maduro medio": "Mantenimiento de bajo costo; candidato secundario si los mercados prioritarios se saturan.",
}


def cluster_profile_summary(feats: pd.DataFrame, clusters: pd.DataFrame, label_col: str,
                            cluster_names: dict) -> pd.DataFrame:
    """Perfil de negocio por cluster: tamaño, comportamiento medio, riesgo operacional (volatilidad)
    y recomendación comercial — convierte el clustering en una pieza accionable, no solo en PCA.

    Recalculado en el momento a partir de `feats` (features por país, con volatility/cagr_recent/
    level_last5_mean) y `clusters` (etiquetas ya asignadas por el pipeline); no requiere persistir
    el perfil en el bundle ML.
    """
    merged = clusters[["Country", label_col]].merge(feats, on="Country", how="left")
    overall_median_vol = merged["volatility"].median()
    rows = []
    for label, g in merged.groupby(label_col):
        vol_mean = float(g["volatility"].mean())
        name = cluster_names.get(label, f"Cluster {label}")
        riesgo = "Alto" if vol_mean > overall_median_vol * 1.5 else ("Medio" if vol_mean > overall_median_vol else "Bajo")
        rows.append({
            "cluster": label, "nombre": name, "n_paises": int(len(g)),
            "nivel_medio": float(g["level_last5_mean"].mean()),
            "cagr_medio": float(g["cagr_recent"].mean()),
            "volatilidad_media": vol_mean, "riesgo_operacional": riesgo,
            "recomendacion": _CLUSTER_RECOMMENDATIONS.get(name, "Evaluar caso a caso."),
        })
    return pd.DataFrame(rows).sort_values("nivel_medio", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_prep import build_long_dataset
    from features import build_and_save

    long_df = build_long_dataset()
    feats = build_and_save(long_df)

    print("Descriptive stats:", descriptive_stats(long_df))
    p = pareto_table(long_df)
    print(f"Top5 share: {p.attrs['top5_share']:.1f}% | Top10: {p.attrs['top10_share']:.1f}% | "
          f"Brasil: {p.attrs['brazil_share']:.1f}%")
