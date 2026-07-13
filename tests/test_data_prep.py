"""Tests de src/data_prep.py: carga, coherencia y transformacion wide -> long."""

import pytest

from data_prep import (
    COUNTRY_CONTINENT,
    COUNTRY_ISO3,
    build_long_dataset,
    load_raw,
    validate_coherence,
)


def test_total_equals_yearly_sum():
    raw = load_raw()
    result = validate_coherence(raw)
    assert result["max_abs_diff"] == 0
    assert result["has_negatives"] is False


def test_validate_coherence_raises_on_incoherent_data():
    """La validacion es un limite del sistema: debe fallar con una excepcion explicita."""
    raw = load_raw().copy()
    raw.loc[0, "Total_domestic_consumption"] += 1
    with pytest.raises(ValueError):
        validate_coherence(raw)


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
    # 55 paises x 30 anos.
    assert long_df.shape[0] == 55 * 30
    assert {
        "Country",
        "Coffee type",
        "consumption",
        "fiscal_year_start",
        "iso3",
        "continent",
        "is_valid_series",
    } <= set(long_df.columns)


def test_zero_series_flagged(long_df):
    invalid = set(long_df.loc[~long_df["is_valid_series"], "Country"].unique())
    assert invalid == {"Equatorial Guinea", "Nepal"}


def test_cote_divoire_encoding_fixed(long_df):
    assert "Côte d'Ivoire" in set(long_df["Country"])
    assert not any("�" in c for c in long_df["Country"].unique())


def test_build_long_dataset_reuses_provided_raw(raw, monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("load_raw() no debe ejecutarse cuando raw_df ya fue provisto")

    monkeypatch.setattr("data_prep.load_raw", _fail)
    long_df = build_long_dataset(save_to=None, raw_df=raw)
    assert long_df.shape[0] == 55 * 30
