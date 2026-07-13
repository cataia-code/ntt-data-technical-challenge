# ruff: noqa: E402

"""Pipeline end-to-end: computa forecasting (3 granularidades), ML y clustering, y persiste un
bundle de resultados en reports/data/ que consume el informe web (scripts/build_report.py).

Uso: py -3 scripts/run_pipeline.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_prep import build_long_dataset, load_raw, validate_coherence
from features import build_and_save, attach_forecast_features
from forecasting import forecast_global, forecast_by_type, forecast_all_countries, DEFAULT_HORIZON
from pooled_model import holdout_comparison
from clustering import (cluster_view, profile, label_consumption_clusters, select_k,
                        CONSUMPTION_FEATURES, PREFERENCE_FEATURES, FORECAST_FEATURES)
from ml_extra import detect_anomalies, classify_coffee_preference, pca_2d

DATA_DIR = ROOT / "reports" / "data"
REPORTS_DIR = ROOT / "reports"
TOP_CHART_COUNTRIES = 12


def _res_to_dict(res: dict) -> dict:
    return {
        "model": res["model"], "mape": res["mape"],
        "years": np.asarray(res["years"]).tolist(),
        "history": np.asarray(res["history"]).tolist(),
        "point": np.asarray(res["point"]).tolist(),
        "lower": None if res["lower"] is None else np.asarray(res["lower"]).tolist(),
        "upper": None if res["upper"] is None else np.asarray(res["upper"]).tolist(),
    }


def build_priority_markets(feats: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Ranking de mercados emergentes de alto potencial (crecimiento × tamaño, no saturados)."""
    df = feats.merge(summary[["series", "forecast_final", "last_observed"]].rename(columns={"series": "Country"}),
                     on="Country", how="left")
    df["incremental_2024_25"] = df["forecast_final"] - df["last_observed"]
    # Excluye el mega-mercado maduro (top 5% de nivel) y los de crecimiento negativo.
    level_cap = df["level_mean"].quantile(0.95)
    cand = df[(df["cagr_recent"] > 0) & (df["level_mean"] < level_cap)].copy()
    cand["potential_score"] = cand["cagr_recent"] * np.log1p(cand["level_mean"])
    cols = ["Country", "Coffee type", "level_mean", "cagr_recent", "incremental_2024_25", "potential_score"]
    return cand.sort_values("potential_score", ascending=False)[cols].head(10).reset_index(drop=True)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1/8 Cargando y validando coherencia...")
    raw = load_raw()
    coherence = validate_coherence(raw)
    long_df = build_long_dataset()
    print(f"    coherencia OK: {coherence}")

    print("2/8 Features por país...")
    feats = build_and_save(long_df)

    print("3/8 Forecasting global y por tipo de café...")
    g = forecast_global(long_df)
    types = forecast_by_type(long_df)

    print("4/8 Forecasting por país (rolling-origin, ~3 min)...")
    country_results, summary = forecast_all_countries(long_df)
    summary.to_csv(REPORTS_DIR / "forecast_summary.csv", index=False)

    # Coherencia jerárquica: forecast global directo vs. suma de forecasts por país.
    sum_countries = float(sum(r["point"][-1] for r in country_results.values()))
    global_direct = float(g["point"][-1])
    coherence_gap = (sum_countries - global_direct) / global_direct
    print(f"    coherencia jerárquica: global={global_direct:,.0f} vs suma países={sum_countries:,.0f} "
          f"(gap {coherence_gap:+.1%})")

    print("5/8 Modelo ML global pooled (HistGradientBoosting) — comparación justa a 5 años...")
    cmp = holdout_comparison(long_df)
    model_cmp = {
        "horizon": DEFAULT_HORIZON, "n": int(len(cmp)),
        "classic_mape_median": float(cmp["classic_mape"].median()),
        "classic_mape_mean": float(cmp["classic_mape"].mean()),
        "pooled_mape_median": float(cmp["pooled_mape"].median()),
        "pooled_mape_mean": float(cmp["pooled_mape"].mean()),
    }
    print(f"    holdout 5 años (mediana): clásico {model_cmp['classic_mape_median']:.2f}% "
          f"vs pooled {model_cmp['pooled_mape_median']:.2f}%")

    print("6/8 Clustering (3 vistas)...")
    feats_fc = attach_forecast_features(feats, summary)
    cl_cons = cluster_view(feats, CONSUMPTION_FEATURES, k=4, label_col="cluster_consumo")
    prof_cons = profile(cl_cons, "cluster_consumo", CONSUMPTION_FEATURES)
    names_cons = label_consumption_clusters(prof_cons)
    cl_pref = cluster_view(feats, PREFERENCE_FEATURES, k=4, label_col="cluster_preferencia")
    cl_fc = cluster_view(feats_fc.dropna(subset=FORECAST_FEATURES), FORECAST_FEATURES, k=4, label_col="cluster_forecast")
    pca_df = pca_2d(feats, CONSUMPTION_FEATURES)

    clusters = pca_df.merge(cl_cons[["Country", "cluster_consumo"]], on="Country", how="left")
    clusters = clusters.merge(cl_pref[["Country", "cluster_preferencia"]], on="Country", how="left")
    clusters = clusters.merge(cl_fc[["Country", "cluster_forecast"]], on="Country", how="left")
    clusters["cluster_consumo_label"] = clusters["cluster_consumo"].map(names_cons)
    clusters.to_parquet(DATA_DIR / "clusters.parquet", index=False)

    print("7/8 ML extra: anomalías + clasificador de preferencia...")
    anomalies = detect_anomalies(feats)
    anomalies.to_csv(REPORTS_DIR / "anomalies.csv", index=False)
    clf = classify_coffee_preference(feats)

    priority = build_priority_markets(feats, summary)
    priority.to_csv(REPORTS_DIR / "priority_markets.csv", index=False)

    print("8/8 Persistiendo bundle para el informe web...")
    # Bubble map: total histórico + forecast + tipo + iso3.
    valid = long_df[long_df["is_valid_series"]]
    totals = valid.groupby(["Country", "Coffee type", "iso3"])["consumption"].sum().reset_index(name="total")
    bubble = totals.merge(summary[["series", "forecast_final"]].rename(columns={"series": "Country"}),
                          on="Country", how="left")
    bubble = bubble.merge(feats[["Country", "cagr_recent", "level_last5_mean"]], on="Country", how="left")
    bubble.to_parquet(DATA_DIR / "bubble.parquet", index=False)

    forecasts = {"GLOBAL": _res_to_dict(g)}
    forecasts.update({f"TYPE::{t}": _res_to_dict(r) for t, r in types.items()})
    top_countries = summary.head(TOP_CHART_COUNTRIES)["series"].tolist()
    forecasts.update({c: _res_to_dict(country_results[c]) for c in top_countries})
    (DATA_DIR / "forecasts.json").write_text(json.dumps(forecasts), encoding="utf-8")

    type_backtests = {t: r["backtest"][["model", "mape"]].to_dict("records") for t, r in types.items()}
    global_backtest = g["backtest"][["model", "mape"]].to_dict("records")

    ml_summary = {
        "coherence": coherence,
        "coherence_gap_hierarchical": coherence_gap,
        "global_forecast_2024_25": global_direct,
        "model_comparison": model_cmp,
        "rolling_mape_mean": float(summary["backtest_mape"].mean()),
        "classifier": {"cv_accuracy": clf["cv_accuracy_mean"], "cv_std": clf["cv_accuracy_std"],
                       "baseline": clf["majority_baseline"],
                       "importances": clf["feature_importances"].round(3).to_dict()},
        "pca_explained_variance": pca_df.attrs.get("explained_variance", [0, 0]),
        "silhouette_consumo": select_k(feats, CONSUMPTION_FEATURES).to_dict("records"),
        "cluster_consumo_names": {str(k): v for k, v in names_cons.items()},
        "global_backtest": global_backtest,
        "type_backtests": type_backtests,
        "n_countries": int(valid["Country"].nunique()),
        "top_chart_countries": top_countries,
    }
    (DATA_DIR / "ml_summary.json").write_text(json.dumps(ml_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPipeline completo. Bundle en reports/data/. Ejecuta: py -3 scripts/build_report.py")


if __name__ == "__main__":
    main()
