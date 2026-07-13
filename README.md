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
1. **EDA & Insights** (`reports/web/index.html`) — 4 bloques, cada uno respondiendo una pregunta de
   negocio, con cierre Insight/Implicación/Acción: (1) *Data audit y alcance* — calidad del dataset
   (nulos, duplicados, coherencia) y qué puede/no puede responder; (2) *Concentración y distribución* —
   Pareto y distribución del consumo por continente; (3) *Evolución y quiebre reciente* — tendencia
   global interactiva y los top movers del cambio 2018/19→2019/20; (4) *Segmentación e insights clave* —
   matriz tamaño×crecimiento (Priorizar/Defender/Explorar/Baja prioridad), tabla por tipo de café, y una
   síntesis de 2 insights cruzados clave. Cierra con limitaciones del análisis y una sección de recomendaciones de
   negocio explícitamente separada del EDA.
2. **Forecasting & ML** (`reports/web/forecasting.html`) — a 3 granularidades (global, 4 tipos de café,
   53 países), horizonte publicado **hasta 2030** (+11 años desde 2019/20), en 4 bloques técnicos:
   (1) *Forecast validation* — metodología (rolling-origin backtest, baseline naive, cómo se evita
   leakage) + selector de forecast + dispersión de error + ejemplos de buen/mal ajuste + tabla de
   *model governance* por serie; (2) *Model comparison* — pooled HistGradientBoosting vs. modelos
   clásicos por-serie, holdout justo a 5 años; (3) *Risk / anomaly governance* — IsolationForest +
   clasificador de tipo, con peso visual proporcional a si conecta con una acción; (4) *Segmentation &
   business action* — clustering (3 vistas) con perfil de negocio por cluster (tamaño, riesgo
   operacional, recomendación) y la traducción explícita de cada resultado a una decisión. Cierra con
   qué cambiaría para producción (drift, reentrenamiento, CI de modelo).
3. **Segmentación en 3 vistas** — por consumo/crecimiento, por preferencia de tipo, y por trayectoria
   de forecast.
4. **Informe web interactivo** (`reports/web/`) con estilo café: páginas de EDA, Forecasting & ML, y un
   placeholder de GenAI.

## Estructura

Servicios internos con fronteras claras: **datos → features → analítica/ML → reporte**, cada capa
solo conoce la anterior (el reporte nunca recalcula lo que ya calculó la analítica; la analítica
nunca sabe cómo se renderiza en HTML). `config.py` es la única fuente de verdad para rutas.

```
data/raw/                  Dataset original
data/processed/            Long dataset + features por país
src/
  config.py                Rutas + TARGET_FORECAST_YEAR centralizados (sin config dispersa)
  data_prep.py             wide→long, ISO-3, continente, validación de coherencia (raise, no assert)
  features.py              CAGR robusto, log, features de forecast
  business_views.py        dataset_overview (data audit), Pareto, cuadrante BCG, top movers, tabla por
                            tipo, ranking, mercados prioritarios, model governance, perfil de cluster,
                            stats descriptivas
  forecasting.py           rolling-origin backtest, log, bandas que se ensanchan con el horizonte,
                            baseline naive, 3 granularidades
  pooled_model.py          HistGradientBoosting global + comparación justa
  ml_extra.py              IsolationForest, clasificador de tipo, PCA
  clustering.py            3 vistas de clustering
  plotly_charts.py         figuras Plotly (tema café validado, selectores interactivos, dispersión de
                            error de backtest)
  report_data.py           ReportBundle: única capa de I/O del informe — carga el dataset base
                            (raw/long/features, reutilizando el raw en memoria) y el bundle de
                            resultados del pipeline, ya tipados
  report_components.py     'UI kit' HTML puro (shell, stat, insight_box...), sin lógica de negocio
  report_sections.py       contenido de cada página (EDA, Forecasting & ML, GenAI): recibe el
                            ReportBundle ya cargado, llama business_views y formatea a HTML — sin I/O
  viz.py                   plotting matplotlib para notebooks
notebooks/                 01_eda · 02_forecasting · 03_clustering
scripts/
  run_pipeline.py          orquesta forecasting/ML/clustering y persiste el bundle en reports/data/
  build_report.py          orquestador delgado: carga el bundle y delega el HTML a report_sections
reports/
  web/                     informe interactivo (abrir index.html)
  data/                    bundle de resultados (JSON/parquet)
  *.csv                    forecast_summary, priority_markets, anomalies
  executive_summary.md     resumen de negocio + propuesta GenAI
tests/                     un módulo de test por servicio (conftest.py con fixtures compartidas);
                            report_sections se prueba con un ReportBundle sintético, sin depender
                            de haber corrido el pipeline completo
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
- **Paleta de tipos de café separada por matiz + luminosidad**, no solo por matiz (Arabica/Robusta
  compartían tono naranja-caramelo y eran casi indistinguibles a tamaño de burbuja pequeño — corregido).
  No hay una herramienta de validación CVD automatizada corriendo en CI; la separación se verificó a ojo
  sobre la matriz tamaño×crecimiento real.
- **Sin dependencias pesadas**: pandas, scikit-learn, statsmodels y plotly; sin MLflow/Docker/LightGBM,
  fuera de alcance para un dataset de 55 filas.
