# Resumen Ejecutivo — Analítica de Consumo Doméstico de Café

**Para:** Área de Innovación, High Garden Coffee
**Fuente:** `coffee_db.parquet` (55 países, 1990/91–2019/20)
**Entregable interactivo:** `reports/web/index.html`

## 1. Alcance y limitaciones de los datos

El dataset contiene **consumo doméstico** por país y año, no precios. La petición de "rangos de precios
futuros" se reformula como **rangos de consumo futuro** (con intervalos), que es lo que los datos
permiten responder con rigor. Incorporar precio requeriría una fuente externa (ICO Composite Price,
futuros de arábica/robusta en ICE) — recomendación de roadmap.

Se verificó la **coherencia** del dato: `Total_domestic_consumption` es exactamente igual a la suma de
las 30 columnas anuales en las 55 filas (diferencia máxima 0, sin negativos). Las unidades son **tazas**,
tal como indica el enunciado. Dos países (Equatorial Guinea, Nepal) tienen serie en cero y se excluyen
del modelado (quedan 53 series válidas).

## 2. Insights clave

1. **Crecimiento sostenido con quiebre reciente.** El consumo global casi se triplicó (1990/91: 1,171M →
   2018/19: 3,015M tazas) pero **cayó 0.5%** en 2019/20 (2,999M) — primer retroceso interanual relevante,
   a vigilar. La caída no es generalizada: Venezuela y Brasil la explican en gran parte, pero Colombia,
   Etiopía y Vietnam crecieron ese mismo año y compensaron parte de la baja.
2. **Concentración extrema.** Brasil concentra **45%** del consumo doméstico acumulado; los 5 países más
   grandes explican **71%** y los 10 más grandes, **87%**. La media (57M) es >12x la mediana (5M) —
   evidencia directa de la asimetría.
3. **Un grupo de mercados medianos crece muy por encima del promedio**: Vietnam (8.2%/año), Tailandia
   (7.2%), Filipinas (6.0%) y Colombia (4.7%) — el mejor balance tamaño/momentum, distinto de Brasil (gran
   tamaño, crecimiento moderado ~1.8%/año).
4. **El tipo Robusta (puro) es el único en declive** (CAGR reciente −0.7%, solo 2.3% del consumo); el
   segmento Robusta/Arabica crece más rápido (4.7%/año) pese a no ser el de mayor volumen.
5. **La preferencia de tipo de café NO se explica por el patrón de consumo** (ver §4) — responde a
   geografía/agronomía, no a la dinámica de demanda.

## 3. Forecasting de consumo (proyección 2024/25)

Selección de modelo por **backtest de origen móvil** (rolling-origin), ajuste en escala log e intervalos
empíricos P10–P90. Tres granularidades: global, por tipo de café y por país.

| Serie | Modelo | MAPE backtest | 2019/20 | Proyección 2024/25 |
|---|---|---|---|---|
| **Global** | ARIMA | 1.1% | 2,999M | **~3,079M** |
| Brasil | ARIMA | 1.5% | 1,320M | ~1,283M |
| Indonesia | naive | 1.7% | 288M | ~288M |
| Etiopía | Holt-Winters | 0.5% | 227M | ~253M |
| Filipinas | ARIMA | 2.9% | 195M | ~223M |
| Vietnam | ARIMA | 1.3% | 159M | ~169M |

**Coherencia jerárquica:** el forecast global directo (3,079M) y la suma de forecasts por país (3,020M)
difieren en −1.9% — dentro de lo esperado al modelar cada nivel por separado.

## 4. Machine Learning aplicado (más allá del forecasting)

- **Modelo global "pooled" (HistGradientBoosting).** Una sola máquina entrenada sobre las 53 series
  apiladas, compartiendo señal entre países y prediciendo la **tasa de crecimiento** (los árboles no
  extrapolan niveles). En una comparación **justa** sobre el mismo holdout de 5 años obtiene 4.9% de MAPE
  mediano frente a **0.9%** de los modelos clásicos por-serie. Conclusión basada en evidencia: para estas
  series maduras y suaves, los modelos clásicos ganan; el modelo simple se elige tras evaluar el enfoque
  ML moderno, no por defecto.
- **Clasificador de preferencia de tipo (RandomForest).** Intenta predecir si un mercado es
  Arabica-dominante desde su patrón de consumo: 52% de accuracy (CV) vs. 55% del baseline — **no supera
  al azar**. Resultado negativo informativo: la preferencia de tipo no se infiere de la dinámica de
  consumo.
- **Detección de anomalías (IsolationForest).** Mercados con patrón multivariante atípico: Brasil (escala
  extrema), Tanzania, Togo, Vietnam, Côte d'Ivoire y Gabón — revisar antes de confiar en su forecast.
- **PCA + clustering** para segmentación (§5).

Se descartó **deep learning** por diseño: con ~30 observaciones anuales por serie sobreajustaría.

## 5. Segmentación de mercados (3 vistas)

Clustering sobre los 53 países en tres vistas complementarias: por **consumo/crecimiento**, por
**preferencia de tipo**, y por **trayectoria de forecast**. La vista de consumo produce 4 segmentos de
negocio: *Maduro de gran escala* (Brasil), *Emergente de alto crecimiento*, *Estable / maduro medio* y
*Volátil / atípico*.

## 6. Recomendaciones de negocio

1. **Priorizar los mercados emergentes de alto potencial**: Vietnam, Tailandia, Filipinas, Colombia,
   Tanzania (crecimiento reciente alto, tamaño no saturado). Varios son **Robusta/Arabica** — alinear la
   oferta de exportación a ese perfil.
2. **Estrategia por tipo de café**: el segmento Arabica/Robusta concentra el mayor volumen agregado y
   crece de forma sostenida; los mercados emergentes se inclinan a mezclas con Robusta. Segmentar la
   oferta por región según el tipo dominante.
3. **Vigilar la caída de 2019/20** en la próxima campaña del ICO para confirmar si es puntual o un cambio
   de tendencia global.
4. **Validar las 6 anomalías detectadas** contra la fuente antes de basar decisiones en sus años atípicos.
5. **Para abordar el pedido original de "precio"**, incorporar una fuente externa de precios y repetir la
   metodología de forecasting sobre esa variable (fase 2).

## 7. Bonus — Integración de IA generativa / LLM (propuesta)

Página placeholder en el informe web (`reports/web/genai.html`), en evaluación. Dos propuestas concretas:
- **(a) Narrativas automáticas de insights**: un LLM (Claude) redacta insights y recomendaciones a partir
  de los resultados ya calculados por el pipeline (rankings, perfiles de cluster, forecast). Bajo riesgo:
  resume resultados validados, no interpreta datos crudos.
- **(b) Asistente text-to-pandas**: responde preguntas de negocio en lenguaje natural generando y
  ejecutando código pandas sobre los datos y el forecast, con la tabla/gráfico de soporte.

Requisitos si se implementa: SDK `anthropic`, `ANTHROPIC_API_KEY` vía `.env` (no versionada).

**Productivización real** (fuera de alcance): con datos actualizándose cada campaña, se movería a un
pipeline con tracking de experimentos (MLflow), reentrenamiento programado por crop-year y una API ligera
(FastAPI) para servir el forecast y el asistente.

---
*Detalle técnico y visualizaciones interactivas en `reports/web/` y en los notebooks
`notebooks/01_eda.ipynb`, `02_forecasting.ipynb`, `03_clustering.ipynb`.*
