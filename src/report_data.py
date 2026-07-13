"""Capa de acceso a datos del informe web: el ÚNICO módulo con permiso de tocar disco para el
reporte (parquet/CSV/JSON del bundle de resultados, y el dataset base crudo/long/features).

`report_sections.py` compone HTML a partir de un `ReportBundle` ya cargado — nunca vuelve a leer un
archivo. Si el pipeline cambia de formato de persistencia, o el dataset base cambia de fuente, solo
este módulo cambia.
"""

import json
from dataclasses import dataclass

import pandas as pd

from config import BUNDLE_DIR, FORECAST_SUMMARY_CSV, PRIORITY_MARKETS_CSV, ANOMALIES_CSV
from data_prep import build_long_dataset, load_raw
from features import build_country_features


@dataclass
class ReportBundle:
    """Todo el input de datos que necesita el informe web, ya cargado en memoria: el dataset base
    (raw/long/features) y el bundle de resultados ML persistido por scripts/run_pipeline.py.
    """
    # Dataset base
    raw: pd.DataFrame
    long_df: pd.DataFrame
    feats: pd.DataFrame
    # Bundle de resultados del pipeline (reports/data/ + reports/*.csv)
    ml: dict
    forecasts: dict
    bubble: pd.DataFrame
    clusters: pd.DataFrame
    forecast_summary: pd.DataFrame
    priority: pd.DataFrame
    anomalies: pd.DataFrame


def load_bundle(bundle_dir=BUNDLE_DIR) -> ReportBundle:
    """Carga el dataset base y el bundle completo de resultados que consume el informe web.

    `build_long_dataset(save_to=None)`: el informe solo LEE el dataset, no debe volver a escribir
    `data/processed/coffee_long.parquet` como efecto secundario (eso es responsabilidad exclusiva
    de scripts/run_pipeline.py).
    """
    raw = load_raw()
    long_df = build_long_dataset(save_to=None, raw_df=raw)
    feats = build_country_features(long_df)
    return ReportBundle(
        raw=raw,
        long_df=long_df,
        feats=feats,
        ml=json.loads((bundle_dir / "ml_summary.json").read_text(encoding="utf-8")),
        forecasts=json.loads((bundle_dir / "forecasts.json").read_text(encoding="utf-8")),
        bubble=pd.read_parquet(bundle_dir / "bubble.parquet"),
        clusters=pd.read_parquet(bundle_dir / "clusters.parquet"),
        forecast_summary=pd.read_csv(FORECAST_SUMMARY_CSV),
        priority=pd.read_csv(PRIORITY_MARKETS_CSV),
        anomalies=pd.read_csv(ANOMALIES_CSV),
    )
