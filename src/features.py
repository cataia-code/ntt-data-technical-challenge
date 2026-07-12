"""Construcción de features por país a partir del dataset long, para clustering y ML."""

from pathlib import Path

import numpy as np
import pandas as pd

RECENT_YEARS = 10
LAST_N_YEARS = 5

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "country_features.parquet"


def _cagr_robust(series: pd.Series) -> float:
    """CAGR calculado desde el primer año con valor > 0 (evita NaN por arranques en cero)."""
    s = series[series > 0]
    if len(s) < 2:
        return np.nan
    first, last = s.iloc[0], s.iloc[-1]
    periods = s.index[-1] - s.index[0]
    if first <= 0 or periods <= 0:
        return np.nan
    return (last / first) ** (1 / periods) - 1


def build_country_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Deriva features de nivel, crecimiento, volatilidad y preferencia por país.

    Excluye países marcados como is_valid_series=False (series en cero).
    Usa CAGR robusto (desde primer año no-cero) para no perder países con arranque tardío.
    """
    valid = long_df[long_df["is_valid_series"]].copy()
    global_total = valid.groupby("fiscal_year_start")["consumption"].sum()

    rows = []
    for country, g in valid.groupby("Country"):
        g = g.sort_values("fiscal_year_start")
        s = g.set_index("fiscal_year_start")["consumption"].astype(float)
        years = s.index.to_numpy()
        vals = s.to_numpy()
        last_year = years[-1]

        cagr_full = _cagr_robust(s)
        cagr_recent = _cagr_robust(s[s.index >= last_year - RECENT_YEARS])

        pct_change = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        volatility = pct_change.std() if len(pct_change) > 1 else np.nan

        recent = s[s.index >= last_year - RECENT_YEARS]
        trend_slope_recent = np.polyfit(recent.index.to_numpy(), recent.to_numpy(), 1)[0] if len(recent) > 1 else np.nan

        level_mean = vals.mean()
        level_last5_mean = s[s.index >= last_year - LAST_N_YEARS].mean()
        share_of_global = (vals / global_total.reindex(years).to_numpy()).mean()

        coffee_type = g["Coffee type"].iloc[0]
        # Preferencia binaria: tipo dominante (primer componente del "A/B") es Arabica o Robusta.
        arabica_dominant = int(coffee_type.split("/")[0].strip().lower().startswith("arabica"))

        rows.append({
            "Country": country,
            "iso3": g["iso3"].iloc[0],
            "continent": g["continent"].iloc[0],
            "Coffee type": coffee_type,
            "arabica_dominant": arabica_dominant,
            "level_mean": level_mean,
            "level_last5_mean": level_last5_mean,
            "log_level_mean": np.log1p(level_mean),
            "cagr_full": cagr_full,
            "cagr_recent": cagr_recent,
            "volatility": volatility,
            "trend_slope_recent": trend_slope_recent,
            "share_of_global": share_of_global,
        })

    feats = pd.DataFrame(rows)
    # Imputa CAGR faltante (series muy cortas) con la mediana para no perder al país en clustering.
    for col in ["cagr_full", "cagr_recent", "volatility"]:
        feats[col] = feats[col].fillna(feats[col].median())
    return feats


def attach_forecast_features(feats: pd.DataFrame, forecast_summary: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivadas del forecast por país (nivel y CAGR proyectados).

    `forecast_summary` debe tener columnas: series (=Country), last_observed, forecast_final.
    """
    fc = forecast_summary.rename(columns={"series": "Country"}).copy()
    horizon = fc.attrs.get("horizon", 5)
    fc["proj_cagr"] = (fc["forecast_final"] / fc["last_observed"]) ** (1 / horizon) - 1
    fc["proj_level"] = fc["forecast_final"]
    merged = feats.merge(fc[["Country", "proj_cagr", "proj_level"]], on="Country", how="left")
    return merged


def build_and_save(long_df: pd.DataFrame, save_to: Path = FEATURES_PATH) -> pd.DataFrame:
    feats = build_country_features(long_df)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(save_to, index=False)
    return feats


if __name__ == "__main__":
    from data_prep import build_long_dataset

    long_df = build_long_dataset()
    feats = build_and_save(long_df)
    print(feats.shape, "NaNs:", feats.isna().sum().sum())
    print(feats[["Country", "cagr_recent", "log_level_mean", "arabica_dominant"]].head())
