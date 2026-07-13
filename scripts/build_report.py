# ruff: noqa: E402

"""Genera el informe web interactivo (reports/web/) a partir del bundle de reports/data/.

Orquestador delgado: carga los datos (report_data), delega TODO el contenido de cada página a
report_sections, y escribe los HTML resultantes envueltos en el shell común (report_components).
No contiene narrativa de negocio ni cálculo — eso vive en business_views/report_sections.

Ejecutar DESPUÉS de scripts/run_pipeline.py.

Uso: py -3 scripts/build_report.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import plotly

from config import ASSETS_DIR, WEB_DIR
from report_data import load_bundle
import report_components as rc
import report_sections as rs


def build():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", ASSETS_DIR / "plotly.min.js")

    bundle = load_bundle()

    eda_body = rs.build_eda_page(bundle)
    fc_body = rs.build_forecasting_page(bundle)
    genai_body = rs.build_genai_page()

    # Cache-bust assets/styles.css: sin esto, un navegador ya abierto sigue sirviendo la hoja de
    # estilos vieja desde caché aunque el archivo en disco cambie entre builds sucesivos.
    css_version = str(int((ASSETS_DIR / "styles.css").stat().st_mtime))

    (WEB_DIR / "index.html").write_text(rc.shell("EDA", "EDA", eda_body, css_version), encoding="utf-8")
    (WEB_DIR / "forecasting.html").write_text(
        rc.shell("Forecasting & ML", "Forecasting & ML", fc_body, css_version), encoding="utf-8")
    (WEB_DIR / "genai.html").write_text(rc.shell("GenAI", "GenAI", genai_body, css_version), encoding="utf-8")
    print("Informe web generado en reports/web/ (abre index.html en el navegador).")


if __name__ == "__main__":
    build()
