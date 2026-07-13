"""Tests de src/report_components.py: 'UI kit' HTML puro, sin datos ni lógica de negocio."""

import report_components as rc


def test_human_formats_by_magnitude():
    assert rc.human(1_500_000_000) == "1.50 B"
    assert rc.human(3_015_000) == "3 M"
    assert rc.human(120) == "120"


def test_stat_renders_value_label_and_delta():
    html = rc.stat("42", "%", "Crecimiento", "+5 pts", "up")
    assert "42" in html and "%" in html and "Crecimiento" in html
    assert 'class="delta up"' in html


def test_insight_box_has_three_rows():
    html = rc.insight_box("hallazgo", "implicación", "acción")
    assert html.count("insight-row") == 3
    assert "hallazgo" in html and "implicación" in html and "acción" in html


def test_pill_applies_color():
    assert 'background:#C77E12' in rc.pill("Arabica", "#C77E12")


def test_pipeline_numbers_steps_and_connects_with_arrows():
    html = rc.pipeline([("Scraping", "desc a", "Ingesta"), ("Limpieza", "desc b", "ETL")])
    assert html.count('class="pipeline-step"') == 2
    assert html.count('class="pipeline-arrow"') == 1  # N pasos -> N-1 flechas, no una al final
    assert "Scraping" in html and "Limpieza" in html and "ETL" in html


def test_shell_marks_active_nav_link():
    html = rc.shell("EDA", "EDA", "<p>body</p>")
    assert '<a href="index.html" class="active">EDA</a>' in html
    assert "<p>body</p>" in html
