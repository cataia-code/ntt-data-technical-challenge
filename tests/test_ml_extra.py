"""Tests de src/ml_extra.py: anomalías (IsolationForest) y clasificador de preferencia."""

from ml_extra import classify_coffee_preference, detect_anomalies


def test_anomaly_detection(feats):
    anom = detect_anomalies(feats)
    assert "anomaly" in anom.columns
    assert 0 < anom["anomaly"].sum() < len(feats)


def test_classifier_runs(feats):
    clf = classify_coffee_preference(feats)
    assert 0.0 <= clf["cv_accuracy_mean"] <= 1.0
