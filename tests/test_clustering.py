"""Tests de src/clustering.py: las 3 vistas de segmentación."""

from clustering import CONSUMPTION_FEATURES, PREFERENCE_FEATURES, cluster_view, label_clusters_by_profile, profile


def test_cluster_views_assign_all(feats):
    cl = cluster_view(feats, CONSUMPTION_FEATURES, k=4, label_col="c")
    assert cl["c"].nunique() == 4
    assert len(cl) == len(feats)
    cl2 = cluster_view(feats, PREFERENCE_FEATURES, k=4, label_col="c2")
    assert cl2["c2"].nunique() == 4


def test_label_clusters_by_profile_avoids_generic_names(feats):
    """La queja original: sin nombres, las vistas de preferencia/forecast se leían como
    "Cluster 0/1/2/3" sin significado de negocio."""
    cl = cluster_view(feats, PREFERENCE_FEATURES, k=4, label_col="cluster_preferencia")
    prof = profile(cl, "cluster_preferencia", PREFERENCE_FEATURES)
    labels = label_clusters_by_profile(prof, "log_level_mean", "cagr_recent", dominance_col="arabica_dominant")
    assert len(labels) == 4
    assert all(not v.lower().startswith("cluster ") for v in labels.values())
    assert any("dom." in v for v in labels.values())
