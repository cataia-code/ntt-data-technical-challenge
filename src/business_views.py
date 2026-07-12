"""Vistas analíticas de negocio para el EDA narrativo: concentración, cuadrante estratégico,
contribución al cambio reciente, resumen por tipo de café, ranking priorizado y estadísticas
descriptivas. Todo se calcula a partir de `long_df` (+ `feats` para volatilidad), sin depender del
forecasting pesado — se puede recomputar en segundos.
"""

import numpy as np
import pandas as pd

RECENT_YEARS = 10


# --------------------------------------------------------------------------- #
# 0. Estadísticas descriptivas (media, mediana, moda) — distribución del último año
# --------------------------------------------------------------------------- #
def descriptive_stats(long_df: pd.DataFrame, year: int = None) -> dict:
    """Media, mediana y clase modal (histograma) del consumo por país en un año dado.

    La moda no está bien definida para una variable continua (cada valor es casi único); se reporta
    la CLASE MODAL de un histograma de 10 bins, no un valor puntual — y se documenta como tal.
    """
    valid = long_df[long_df["is_valid_series"]]
    year = year or int(valid["fiscal_year_start"].max())
    s = valid[valid["fiscal_year_start"] == year].set_index("Country")["consumption"].astype(float)

    # Clase modal en espacio log10 (consistente con el histograma de la página, que también usa
    # bins logarítmicos dado lo sesgada que está la distribución).
    s_pos = s[s > 0]
    log_counts, log_edges = np.histogram(np.log10(s_pos), bins=10)
    modal_idx = log_counts.argmax()
    modal_range = (float(10 ** log_edges[modal_idx]), float(10 ** log_edges[modal_idx + 1]))
    modal_count = int(log_counts[modal_idx])

    return {
        "year": year, "n": len(s), "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std()), "min": float(s.min()), "max": float(s.max()),
        "modal_range": modal_range, "modal_count": modal_count,
        "skew_right": bool(s.mean() > s.median() * 1.5),  # asimetría fuerte: media >> mediana
    }


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


if __name__ == "__main__":
    from data_prep import build_long_dataset
    from features import build_and_save

    long_df = build_long_dataset()
    feats = build_and_save(long_df)

    print("Descriptive stats:", descriptive_stats(long_df))
    p = pareto_table(long_df)
    print(f"Top5 share: {p.attrs['top5_share']:.1f}% | Top10: {p.attrs['top10_share']:.1f}% | "
          f"Brasil: {p.attrs['brazil_share']:.1f}%")
