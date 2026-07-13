"""Rutas y constantes compartidas del proyecto.

Única fuente de verdad para ubicaciones en disco: antes cada script/test recomputaba
`Path(__file__).resolve().parents[N]` de forma independiente (configuración dispersa y frágil ante
cualquier reorganización de carpetas). Todo consumidor de rutas importa de aquí.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Datos --- #
RAW_PATH = ROOT / "data" / "raw" / "coffee_db.parquet"
LONG_PATH = ROOT / "data" / "processed" / "coffee_long.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "country_features.parquet"

# --- Reportes --- #
REPORTS_DIR = ROOT / "reports"
BUNDLE_DIR = REPORTS_DIR / "data"          # bundle intermedio persistido por scripts/run_pipeline.py
WEB_DIR = REPORTS_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"

FORECAST_SUMMARY_CSV = REPORTS_DIR / "forecast_summary.csv"
PRIORITY_MARKETS_CSV = REPORTS_DIR / "priority_markets.csv"
ANOMALIES_CSV = REPORTS_DIR / "anomalies.csv"

# --- Forecasting --- #
# Horizonte del forecast PUBLICADO (proyección final mostrada en el informe). Distinto del horizonte
# de la comparación justa clásico-vs-pooled (forecasting.DEFAULT_HORIZON, fijo en 5 años) — ver
# scripts/run_pipeline.py y la sección de metodología del informe.
TARGET_FORECAST_YEAR = 2030
