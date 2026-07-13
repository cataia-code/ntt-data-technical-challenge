"""Tests de src/forecasting.py: backtest de origen móvil y forecast por serie."""

import numpy as np

from forecasting import MODELS, forecast_by_type, forecast_series, rolling_backtest


def test_rolling_backtest_all_models(long_df):
    brazil = (long_df[long_df["Country"] == "Brazil"]
              .sort_values("fiscal_year_start")["consumption"].to_numpy(dtype=float))
    bt = rolling_backtest(brazil, horizon=1)
    assert set(bt["model"]) == set(MODELS)
    assert bt["mape"].notna().all()
    assert (bt["mape"] >= 0).all()


def test_forecast_series_positive_with_band(long_df):
    brazil = (long_df[long_df["Country"] == "Brazil"]
              .sort_values("fiscal_year_start")["consumption"].to_numpy(dtype=float))
    res = forecast_series(brazil, horizon=5)
    assert len(res["point"]) == 5
    assert (np.asarray(res["point"]) > 0).all()
    if res["lower"] is not None:
        assert (np.asarray(res["lower"]) <= np.asarray(res["upper"])).all()


def test_forecast_series_reports_empirical_coverage(long_df):
    """La banda de incertidumbre debe venir con evidencia de cobertura leave-one-out, no solo la
    fórmula que la construye — si hay banda (rel.size>=3), debe haber al menos un punto evaluado."""
    brazil = (long_df[long_df["Country"] == "Brazil"]
              .sort_values("fiscal_year_start")["consumption"].to_numpy(dtype=float))
    res = forecast_series(brazil, horizon=5)
    assert "coverage_covered" in res and "coverage_total" in res
    if res["lower"] is not None:
        assert res["coverage_total"] > 0
        assert 0 <= res["coverage_covered"] <= res["coverage_total"]


def test_forecast_by_type_covers_four_types(long_df):
    out = forecast_by_type(long_df)
    assert len(out) == 4
    for r in out.values():
        assert (np.asarray(r["point"]) > 0).all()
