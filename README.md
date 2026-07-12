# Reto Técnico ML — High Garden Coffee

Solución analítica y de Machine Learning sobre el dataset de consumo doméstico de café
(1990–2020, 55 países) para el reto técnico de NTT DATA. Incluye un **informe web interactivo con
estilo café** como entregable de presentación.

## Qué resuelve

El dataset (`data/raw/coffee_db.parquet`) contiene consumo doméstico (en tazas) por país y año, **sin
columna de precio** — el pedido de "rangos de precios futuros" se reformula como forecasting de
**consumo** con intervalos de confianza (limitación documentada). Se verificó además que
`Total_domestic_consumption` == suma de las columnas anuales en las 55 filas (coherencia perfecta).

**Componentes:**
1. **EDA narrativo y accionable** (`reports/web/index.html`) — 8 secciones con cierre
   Insight/Implicación/Acción: alcance de datos, tendencia global interactiva (selector Global/tipo/país
   dentro del gráfico), concentración (Pareto + distribución por continente/tipo + histograma
   media/mediana/moda), matriz tamaño×crecimiento con cuadrantes de negocio (Priorizar/Defender/
   Explorar/Baja prioridad), waterfall de contribución al cambio 2018/19→2019/20, tabla por tipo de
   café, riesgos de volatilidad, y ranking final de mercados prioritarios.
2. **Forecasting a 3 granularidades** — global, por tipo de café (4 series) y por país (53 series), con
   selección de modelo por *rolling-origin backtest*, escala log e intervalos empíricos.
3. **Machine Learning complementario** — modelo global *pooled* (HistGradientBoosting) comparado de
   forma justa contra los clásicos; clasificador de preferencia de tipo de café (RandomForest);
   detección de anomalías (IsolationForest); PCA.
4. **Segmentación en 3 vistas** — por consumo/crecimiento, por preferencia de tipo, y por trayectoria
   de forecast.
5. **Informe web interactivo** (`reports/web/`) con estilo café: páginas de EDA, Forecasting & ML, y un
   placeholder de GenAI.

## Estructura

```
data/raw/                  Dataset original
data/processed/            Long dataset + features por país
src/
  data_prep.py             wide→long, ISO-3, continente, validación de coherencia
  features.py              CAGR robusto, log, features de forecast
  business_views.py        Pareto, cuadrante BCG, waterfall, tabla por tipo, ranking, stats descriptivas
  forecasting.py           rolling-origin backtest, log, intervalos, 3 granularidades
  pooled_model.py          HistGradientBoosting global + comparación justa
  ml_extra.py              IsolationForest, clasificador de tipo, PCA
  clustering.py            3 vistas de clustering
  plotly_charts.py         figuras Plotly (tema café validado, selectores interactivos)
  viz.py                   plotting matplotlib para notebooks
notebooks/                 01_eda · 02_forecasting · 03_clustering
scripts/
  run_pipeline.py          computa todo y persiste el bundle en reports/data/
  build_report.py          genera el informe web desde el bundle
reports/
  web/                     informe interactivo (abrir index.html)
  data/                    bundle de resultados (JSON/parquet)
  *.csv                    forecast_summary, priority_markets, anomalies
  executive_summary.md     resumen de negocio + propuesta GenAI
tests/test_pipeline.py     coherencia + smoke tests
```

## Cómo correr

```bash
py -3 -m pip install -r requirements.txt

# 1) Pipeline: computa forecasting/ML/clustering y persiste el bundle (~5 min por el rolling backtest)
py -3 scripts/run_pipeline.py

# 2) Informe web: genera reports/web/*.html desde el bundle (rápido)
py -3 scripts/build_report.py
# luego abrir reports/web/index.html en el navegador

# Tests
py -3 -m pytest tests/
```

## Entregable principal

`reports/web/index.html` — informe interactivo. Para el resumen ejecutivo de negocio, ver
`reports/executive_summary.md`.

## Notas de diseño

- **Sin deep learning**: con ~30 observaciones anuales por serie sobreajustaría; los modelos clásicos +
  un GBM global son lo apropiado a esta escala. La elección del modelo simple es *por evidencia* (ver la
  comparación justa a 5 años en el informe).
- **Paleta de tipos de café validada** con el validador de accesibilidad (CVD-safe en light/dark).
- **Sin dependencias pesadas**: pandas, scikit-learn, statsmodels y plotly; sin MLflow/Docker/LightGBM,
  fuera de alcance para un dataset de 55 filas.
