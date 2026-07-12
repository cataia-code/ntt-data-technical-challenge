"""Tests de calidad: coherencia de datos y smoke tests de modelos y clustering."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import load_raw, build_long_dataset, validate_coherence, COUNTRY_ISO3, COUNTRY_CONTINENT
from features import build_country_features
from forecasting import rolling_backtest, forecast_series, forecast_by_type, MODELS
from clustering import cluster_view, CONSUMPTION_FEATURES, PREFERENCE_FEATURES
from ml_extra import detect_anomalies, classify_coffee_preference
import business_views as bv


@pytest.fixture(scope="module")
def long_df():
    return build_long_dataset(save_to=None)


@pytest.fixture(scope="module")
def feats(long_df):
    return build_country_features(long_df)


# --- Coherencia de datos --------------------------------------------------- #
def test_total_equals_yearly_sum():
    raw = load_raw()
    result = validate_coherence(raw)
    assert result["max_abs_diff"] == 0
    assert result["has_negatives"] is False


def test_no_negative_consumption(long_df):
    assert (long_df["consumption"] >= 0).all()


def test_all_countries_have_iso3(long_df):
    assert long_df["iso3"].isna().sum() == 0
    assert len(COUNTRY_ISO3) == long_df["Country"].nunique()


def test_all_countries_have_continent(long_df):
    assert long_df["continent"].isna().sum() == 0
    assert len(COUNTRY_CONTINENT) == long_df["Country"].nunique()
    assert set(long_df["continent"].unique()) == {"África", "América", "Asia", "Oceanía"}


def test_long_shape(long_df):
    # 55 países × 30 años.
    assert long_df.shape[0] == 55 * 30
    assert {"Country", "Coffee type", "consumption", "fiscal_year_start", "iso3", "continent",
            "is_valid_series"} <= set(long_df.columns)


def test_zero_series_flagged(long_df):
    invalid = set(long_df.loc[~long_df["is_valid_series"], "Country"].unique())
    assert invalid == {"Equatorial Guinea", "Nepal"}


def test_cote_divoire_encoding_fixed(long_df):
    assert "Côte d'Ivoire" in set(long_df["Country"])
    assert not any("�" in c for c in long_df["Country"].unique())


# --- Features -------------------------------------------------------------- #
def test_features_no_nans(feats):
    assert feats.isna().sum().sum() == 0
    assert len(feats) == 53  # 55 - 2 series en cero


# --- Forecasting ----------------------------------------------------------- #
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


def test_forecast_by_type_covers_four_types(long_df):
    out = forecast_by_type(long_df)
    assert len(out) == 4
    for r in out.values():
        assert (np.asarray(r["point"]) > 0).all()


# --- Clustering ------------------------------------------------------------ #
def test_cluster_views_assign_all(feats):
    cl = cluster_view(feats, CONSUMPTION_FEATURES, k=4, label_col="c")
    assert cl["c"].nunique() == 4
    assert len(cl) == len(feats)
    cl2 = cluster_view(feats, PREFERENCE_FEATURES, k=4, label_col="c2")
    assert cl2["c2"].nunique() == 4


# --- ML extra -------------------------------------------------------------- #
def test_anomaly_detection(feats):
    anom = detect_anomalies(feats)
    assert "anomaly" in anom.columns
    assert 0 < anom["anomaly"].sum() < len(feats)


def test_classifier_runs(feats):
    clf = classify_coffee_preference(feats)
    assert 0.0 <= clf["cv_accuracy_mean"] <= 1.0


# --- Business views (EDA narrativo) ----------------------------------------- #
def test_descriptive_stats(long_df):
    stats = bv.descriptive_stats(long_df)
    assert stats["mean"] > stats["median"] > 0  # distribución fuertemente sesgada a la derecha
    assert stats["skew_right"] is True


def test_pareto_sums_to_100(long_df):
    p = bv.pareto_table(long_df)
    assert abs(p["cum_share_pct"].iloc[-1] - 100) < 1e-6
    assert p.attrs["brazil_share"] > p.attrs["top5_share"] / 2  # Brasil domina el top 5


def test_quadrant_classification_covers_all_four(feats):
    bubble_like = feats.rename(columns={"level_mean": "total"})
    quad = bv.classify_quadrant(bubble_like)
    assert set(quad["quadrant"].unique()) <= {"Priorizar", "Defender", "Explorar", "Baja prioridad"}
    assert len(quad) > 0


def test_waterfall_reconciles_to_total_delta(long_df):
    wf = bv.waterfall_contribution(long_df)
    assert abs(sum(v for _, v in wf["items"]) - wf["total_delta"]) < 1.0
    assert abs((wf["total_y0"] + wf["total_delta"]) - wf["total_y1"]) < 1.0


def test_type_summary_covers_four_types(long_df):
    t = bv.type_summary_table(long_df)
    assert len(t) == 4
    assert abs(t["share_pct"].sum() - 100) < 1e-6
