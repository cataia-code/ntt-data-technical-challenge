"""Tests de src/report_sections.py: contenido HTML de cada página del informe.

Construye un ReportBundle sintético (mismo contrato que report_data.load_bundle, sin depender de
haber corrido scripts/run_pipeline.py) para poder probar la generación de páginas de forma aislada
y rápida, en vez de un test de integración que requiera el bundle real en reports/data/.

`test_build_eda_page_never_touches_disk` es la guarda concreta de la separación de capas: si alguien
vuelve a colar una llamada a load_raw()/build_long_dataset() dentro de report_sections.py, ese test
falla de inmediato, en vez de depender de una revisión manual del código para detectarlo.
"""

import numpy as np
import pandas as pd
import pytest

import business_views as bv
import report_sections as rs
from clustering import CONSUMPTION_FEATURES, cluster_view
from report_data import ReportBundle


def _fake_forecast_result(hist: np.ndarray, horizon: int = 5) -> dict:
    point = np.full(horizon, hist[-1])
    return {"model": "naive", "mape": 1.0, "years": list(range(1990, 1990 + len(hist))),
            "history": hist.tolist(), "point": point.tolist(), "lower": None, "upper": None}


def _fake_backtest() -> list:
    """Mismo contrato que `forecasting.rolling_backtest(...)[["model","mape"]].to_dict("records")`:
    ordenado ascendente por MAPE, incluye siempre el baseline "naive"."""
    return [{"model": "arima", "mape": 1.0}, {"model": "naive", "mape": 1.5},
            {"model": "linear", "mape": 1.8}, {"model": "holt_winters", "mape": 2.0}]


@pytest.fixture(scope="module")
def bundle(raw, long_df, feats):
    valid = long_df[long_df["is_valid_series"]]
    n = len(feats)
    mapes = np.linspace(0.5, 6.0, n)

    forecast_summary = pd.DataFrame({
        "series": feats["Country"].to_numpy(),
        "best_model": "naive",
        "backtest_mape": mapes,
        "naive_mape": mapes + 1.0,
        "last_observed": feats["level_last5_mean"].to_numpy(),
        "forecast_final": (feats["level_last5_mean"] * 1.05).to_numpy(),
    })

    bubble = feats[["Country", "Coffee type", "iso3", "cagr_recent", "level_last5_mean"]].copy()
    bubble["total"] = feats["level_mean"] * 30
    bubble = bubble.merge(
        forecast_summary[["series", "forecast_final"]].rename(columns={"series": "Country"}), on="Country")

    priority = bv.priority_markets_table(feats, forecast_summary, top_n=10)

    cl_cons = cluster_view(feats, CONSUMPTION_FEATURES, k=4, label_col="cluster_consumo")
    clusters = cl_cons[["Country", "cluster_consumo"]].copy()
    clusters["cluster_preferencia"] = clusters["cluster_consumo"]
    clusters["cluster_forecast"] = clusters["cluster_consumo"]
    clusters["pc1"], clusters["pc2"] = 0.0, 0.0

    anomalies = feats[["Country", "Coffee type"]].copy()
    anomalies["anomaly"] = False
    anomalies.loc[anomalies.index[:2], "anomaly"] = True

    global_hist = valid.groupby("fiscal_year_start")["consumption"].sum().to_numpy(dtype=float)
    forecasts = {"GLOBAL": _fake_forecast_result(global_hist, horizon=11)}
    for t in ["Arabica", "Arabica/Robusta", "Robusta", "Robusta/Arabica"]:
        forecasts[f"TYPE::{t}"] = _fake_forecast_result(global_hist, horizon=11)
    for c in forecast_summary["series"]:  # las 53 series, no una muestra
        forecasts[c] = _fake_forecast_result(global_hist, horizon=11)

    ml = {
        "n_countries": len(feats),
        "global_forecast_final": float(global_hist[-1]),
        "forecast_target_year": 2030,
        "forecast_horizon_years": 11,
        "validation_horizon_years": 5,
        "last_observed_year": 2019,
        "model_comparison": {"classic_mape_median": 1.0, "pooled_mape_median": 2.0, "horizon": 5, "n": n},
        "rolling_mape_mean": float(forecast_summary["backtest_mape"].mean()),
        "rolling_mape_median": float(forecast_summary["backtest_mape"].median()),
        "naive_mape_mean": float(forecast_summary["naive_mape"].mean()),
        "classifier": {"cv_accuracy": 0.5, "baseline": 0.55, "importances": {"log_level_mean": 0.4}},
        "coherence_gap_hierarchical": 0.01,
        "cluster_consumo_names": {"0": "Maduro de gran escala", "1": "Emergente de alto crecimiento",
                                  "2": "Volátil / atípico", "3": "Estable / maduro medio"},
        "cluster_preferencia_names": {"0": "Arabica-dom. · grande, en crecimiento", "1": "Robusta-dom. · pequeño, estable",
                                      "2": "Arabica-dom. · pequeño, estable", "3": "Robusta-dom. · grande, en crecimiento"},
        "cluster_forecast_names": {"0": "Grande, en crecimiento", "1": "Chico, estable",
                                   "2": "Chico, en crecimiento", "3": "Grande, estable"},
        "pca_explained_variance": [0.5, 0.3],
        "silhouette_consumo": [{"k": 3, "silhouette": 0.31}, {"k": 4, "silhouette": 0.35},
                               {"k": 5, "silhouette": 0.29}, {"k": 6, "silhouette": 0.27}],
        "silhouette_k4": {"consumo": 0.35, "preferencia": 0.40, "forecast": 0.33},
        "interval_coverage": {"covered": 210, "total": 260, "pct": 80.8, "target_pct": 80},
        "global_backtest": _fake_backtest(),
        "type_backtests": {t: _fake_backtest() for t in ["Arabica", "Arabica/Robusta", "Robusta", "Robusta/Arabica"]},
    }

    return ReportBundle(raw=raw, long_df=long_df, feats=feats, ml=ml, forecasts=forecasts,
                        bubble=bubble, clusters=clusters, forecast_summary=forecast_summary,
                        priority=priority, anomalies=anomalies)


def test_build_eda_page_renders_all_blocks(bundle):
    html = rs.build_eda_page(bundle)
    assert "Data audit y alcance" in html
    assert "Concentración y distribución" in html
    assert "Evolución y quiebre reciente" in html
    assert "Insights clave" in html
    assert "Limitaciones del análisis" in html
    assert "Recomendaciones" in html


def test_build_eda_page_key_insights_do_not_repeat_block_insights(bundle):
    """El cierre 'Insights clave' debe aportar hallazgos nuevos, no repetir lo ya dicho en los
    insight_box de cada bloque (bloque 2: Brasil/top10; bloque 3: CAGR global/quiebre 2019/20)."""
    html = rs.build_eda_page(bundle)
    key_insights_start = html.index("<h3 class=\"section-title\"")
    key_insights_html = html[key_insights_start:html.index("Limitaciones del análisis")]
    assert "cuadrante" in key_insights_html.lower()  # hallazgo cruzado nuevo: volatilidad × cuadrante
    assert "no es representativo" in key_insights_html.lower()  # hallazgo nuevo: media vs. mediana


def test_build_eda_page_never_touches_disk(bundle, monkeypatch):
    """Guarda de arquitectura: la composición de páginas no debe volver a tocar disco por su cuenta.

    Se parchea el primitivo de I/O (pd.read_parquet/read_csv, Path.read_text), no un nombre
    importado en report_sections — así la guarda detecta CUALQUIER forma de reintroducir I/O
    (`import data_prep; data_prep.load_raw()` o `from data_prep import load_raw`), no solo la que
    causó la violación original.
    """
    from pathlib import Path

    def _fail(*args, **kwargs):
        raise AssertionError("la composición de páginas no debe leer archivos: todo debe venir del ReportBundle")

    monkeypatch.setattr(pd, "read_parquet", _fail)
    monkeypatch.setattr(pd, "read_csv", _fail)
    monkeypatch.setattr(Path, "read_text", _fail)
    rs.build_eda_page(bundle)
    rs.build_forecasting_page(bundle)


def test_build_forecasting_page_renders_four_blocks(bundle):
    html = rs.build_forecasting_page(bundle)
    assert "Forecast validation" in html
    assert "Model comparison" in html
    assert "Risk / anomaly governance" in html
    assert "Segmentation &amp; business action" in html
    assert "Model governance" in html
    assert "Qué cambiaría para producción" in html
    # La matriz tamaño×crecimiento vive solo en EDA; aquí debe referenciarse, no repetirse.
    assert "eda_quad" not in html
    assert "bloque 4" in html


def test_build_forecasting_page_uses_target_horizon(bundle):
    html = rs.build_forecasting_page(bundle)
    assert "2030" in html
    assert "+11 años" in html


def test_build_forecasting_page_cluster_profile_has_recommendation(bundle):
    html = rs.build_forecasting_page(bundle)
    assert "Perfil de negocio por cluster" in html
    assert "Recomendación" in html
    assert "Riesgo operacional" in html


def test_build_forecasting_page_forecast_selector_covers_all_countries(bundle, feats):
    html = rs.build_forecasting_page(bundle)
    assert 'class="tab-select"' in html
    for country in feats["Country"].head(10):
        assert "<option value=" in html and country in html


def test_build_forecasting_page_shows_interval_calibration(bundle):
    """La banda de incertidumbre debe validarse contra datos reales (cobertura leave-one-out), no solo
    declararse por fórmula (√paso)."""
    html = rs.build_forecasting_page(bundle)
    assert "Calibración empírica" in html
    assert "80%" in html


def test_build_forecasting_page_justifies_k4(bundle):
    """La elección de k=4 clusters debe mostrar evidencia (silhouette), no ser arbitraria."""
    html = rs.build_forecasting_page(bundle)
    assert "¿Por qué k=4?" in html
    assert "silhouette" in html.lower()


def test_build_forecasting_page_cluster_labels_are_not_generic(bundle):
    """Las 3 vistas de clustering deben tener nombres de negocio, no "Cluster 0/1/2/3" — esa era la
    queja original: no se entendía el propósito de cada cluster en preferencia/forecast."""
    html = rs.build_forecasting_page(bundle)
    assert "Cluster 0" not in html and "Cluster 1" not in html
    assert "Arabica-dom." in html or "Robusta-dom." in html


def test_build_genai_page_renders_without_data():
    """Página estática (propuesta de arquitectura, no resultado del pipeline): no debe requerir bundle."""
    html = rs.build_genai_page()
    assert "Motor de estrategia de precio y mercado" in html
    assert "Estudio creativo generativo" in html
    assert "Chatbot agéntico de insights" in html
    assert html.count('class="pipeline"') == 3
    assert "Human-in-the-loop" in html
