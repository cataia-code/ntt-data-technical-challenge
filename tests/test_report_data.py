"""Tests de src/report_data.py: la única capa con permiso de tocar disco para el informe web.

`load_bundle()` depende de los artefactos que persiste scripts/run_pipeline.py en reports/data/ y
reports/*.csv — archivos gitignored, no versionados, que NO existen en un checkout limpio de CI (el
job de test no corre el pipeline completo, solo pytest). Por eso este test se salta si el bundle no
está presente en disco: es un test de integración real del límite de I/O, no un smoke test con datos
falsos — se ejecuta cuando corresponde (localmente tras `run_pipeline.py`, o en el workflow de
deploy), y se omite explícitamente en vez de fallar de forma engañosa en la CI rápida.
"""

import pandas as pd
import pytest

from config import BUNDLE_DIR, FORECAST_SUMMARY_CSV
from report_data import ReportBundle, load_bundle

_BUNDLE_AVAILABLE = (BUNDLE_DIR / "ml_summary.json").exists() and FORECAST_SUMMARY_CSV.exists()

pytestmark = pytest.mark.skipif(
    not _BUNDLE_AVAILABLE,
    reason="requiere haber corrido scripts/run_pipeline.py (bundle en reports/data/ gitignored)",
)


def test_load_bundle_returns_fully_populated_bundle():
    bundle = load_bundle()
    assert isinstance(bundle, ReportBundle)
    for field in ("raw", "long_df", "feats", "bubble", "clusters", "forecast_summary", "priority", "anomalies"):
        value = getattr(bundle, field)
        assert isinstance(value, pd.DataFrame)
        assert len(value) > 0
    assert isinstance(bundle.ml, dict) and "n_countries" in bundle.ml
    assert isinstance(bundle.forecasts, dict) and "GLOBAL" in bundle.forecasts


def test_load_bundle_long_df_matches_raw_countries():
    bundle = load_bundle()
    assert bundle.raw["Country"].nunique() == bundle.long_df["Country"].nunique()


def test_load_bundle_does_not_persist_long_dataset():
    """load_bundle() solo LEE — no debe reescribir data/processed/coffee_long.parquet como efecto
    secundario (esa escritura es responsabilidad exclusiva de scripts/run_pipeline.py)."""
    from config import LONG_PATH

    mtime_before = LONG_PATH.stat().st_mtime if LONG_PATH.exists() else None
    load_bundle()
    mtime_after = LONG_PATH.stat().st_mtime if LONG_PATH.exists() else None
    assert mtime_before == mtime_after
