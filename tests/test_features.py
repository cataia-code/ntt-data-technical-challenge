"""Tests de src/features.py: construcción de features por país."""


def test_features_no_nans(feats):
    assert feats.isna().sum().sum() == 0
    assert len(feats) == 53  # 55 - 2 series en cero
