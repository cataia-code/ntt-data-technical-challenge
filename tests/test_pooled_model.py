"""Tests de src/pooled_model.py: modelo global pooled (HistGradientBoosting) y comparación justa
contra los modelos clásicos por-serie. Sin cobertura previa en el proyecto.
"""

from pooled_model import holdout_comparison


def test_holdout_comparison_covers_most_countries(long_df):
    cmp = holdout_comparison(long_df, holdout=5)
    assert len(cmp) > 40  # la mayoría de los 53 países válidos tienen suficiente historia
    assert {"Country", "classic_model", "classic_mape", "pooled_mape"} <= set(cmp.columns)
    # mape() devuelve NaN si algún año del holdout tiene consumo real 0 (ver forecasting.mape) —
    # legítimo para un puñado de series, pero no debería dominar el resultado.
    assert cmp["classic_mape"].notna().mean() > 0.8
    assert (cmp["classic_mape"].dropna() >= 0).all()
    assert (cmp["pooled_mape"].dropna() >= 0).all()
