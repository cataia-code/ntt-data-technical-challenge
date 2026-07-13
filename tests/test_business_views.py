"""Tests de src/business_views.py: vistas de negocio del EDA narrativo."""

import pandas as pd
import pytest

import business_views as bv


def test_descriptive_stats(long_df):
    stats = bv.descriptive_stats(long_df)
    assert stats["mean"] > stats["median"] > 0  # distribución fuertemente sesgada a la derecha
    assert stats["skew_right"] is True


def test_dataset_overview_flags_zero_series(raw, long_df):
    overview = bv.dataset_overview(raw, long_df)
    assert overview["n_rows"] == 55
    assert overview["n_nulls_total"] == 0
    assert overview["n_duplicated_rows"] == 0
    assert overview["zero_series_countries"] == ["Equatorial Guinea", "Nepal"]


def test_global_growth_summary_shows_recent_break(long_df):
    growth = bv.global_growth_summary(long_df)
    assert growth["cagr_global"] > 0  # el consumo global creció en 30 años
    assert growth["last_year_val"] < growth["prev_year_val"]  # caída en el último año (2019/20)
    assert growth["drop_pct"] < 0


def test_continent_size_comparison_orders_by_median(long_df):
    cont = bv.continent_size_comparison(long_df, year=2019)
    median_by_continent = cont["median_by_continent"]
    assert list(median_by_continent.index) == sorted(
        median_by_continent.index, key=lambda c: -median_by_continent[c])
    assert set(cont["count_by_continent"].index) == set(median_by_continent.index)


def test_country_cagr_recent_matches_feats_column(feats):
    expected = float(feats.loc[feats["Country"] == "Brazil", "cagr_recent"].iloc[0])
    assert bv.country_cagr_recent(feats, "Brazil") == expected


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


@pytest.fixture
def fake_forecast_summary(feats):
    # Sustituto ligero de forecasting.forecast_all_countries(): mismo contrato (series/last_observed/
    # forecast_final) sin pagar el costo del backtest real.
    return pd.DataFrame({
        "series": feats["Country"],
        "last_observed": feats["level_last5_mean"],
        "forecast_final": feats["level_last5_mean"] * 1.05,
    })


def test_priority_markets_table_excludes_saturated_and_declining(feats, fake_forecast_summary):
    priority = bv.priority_markets_table(feats, fake_forecast_summary, top_n=10)
    assert {"Country", "Coffee type", "cagr_recent", "potential_score"} <= set(priority.columns)
    assert (priority["cagr_recent"] > 0).all()  # excluye crecimiento negativo
    assert priority["potential_score"].is_monotonic_decreasing
    level_cap = feats["level_mean"].quantile(0.95)
    assert (feats.set_index("Country").loc[priority["Country"], "level_mean"] < level_cap).all()


def test_model_governance_table_covers_global_types_and_countries(fake_forecast_summary):
    fs = fake_forecast_summary.assign(best_model="arima", backtest_mape=1.0, naive_mape=1.5)
    ml = {
        "global_backtest": [{"model": "arima", "mape": 1.0}, {"model": "naive", "mape": 1.5}],
        "type_backtests": {"Arabica": [{"model": "arima", "mape": 1.0}, {"model": "naive", "mape": 1.5}]},
    }
    gov = bv.model_governance_table(ml, fs, top_n=5)
    assert len(gov) == 1 + 1 + 5  # Global + 1 tipo + 5 países
    assert set(gov["Riesgo"]) <= {"Bajo", "Medio", "Alto"}
    assert (gov["Baseline (naive)"] >= gov["MAPE backtest"]).all()


def test_model_governance_table_flags_anomalous_countries(fake_forecast_summary):
    fs = fake_forecast_summary.assign(best_model="arima", backtest_mape=1.0, naive_mape=1.5)
    ml = {"global_backtest": [{"model": "arima", "mape": 1.0}, {"model": "naive", "mape": 1.5}], "type_backtests": {}}
    flagged_country = fs["series"].iloc[0]
    gov = bv.model_governance_table(ml, fs, top_n=3, anomalous_countries={flagged_country})
    row = gov[gov["Serie"] == flagged_country].iloc[0]
    assert "atípico" in row["Riesgo"]


def test_best_worst_fit_countries_picks_extremes(fake_forecast_summary):
    fs = fake_forecast_summary.assign(backtest_mape=range(len(fake_forecast_summary)))
    bw = bv.best_worst_fit_countries(fs)
    assert bw["best"]["backtest_mape"] == 0
    assert bw["worst"]["backtest_mape"] == len(fs) - 1


def test_cluster_profile_summary_has_size_and_recommendation(feats):
    clusters = feats[["Country"]].copy()
    clusters["cluster_consumo"] = [i % 4 for i in range(len(feats))]
    names = {0: "Maduro de gran escala", 1: "Emergente de alto crecimiento",
             2: "Volátil / atípico", 3: "Estable / maduro medio"}
    profile = bv.cluster_profile_summary(feats, clusters, "cluster_consumo", names)
    assert len(profile) == 4
    assert profile["n_paises"].sum() == len(feats)
    assert set(profile["riesgo_operacional"]) <= {"Bajo", "Medio", "Alto"}
    assert (profile["recomendacion"] != "Evaluar caso a caso.").all()
