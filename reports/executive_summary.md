# EDA & Insights — Analítica de Consumo Doméstico de Café

**Para:** Área de Innovación, High Garden Coffee
**Fuente:** `coffee_db.parquet` (55 países, 1990/91–2019/20)
**Entregable interactivo:** `reports/web/index.html` — EDA en 4 bloques (data audit, concentración,
evolución, segmentación), cada uno con hallazgo/interpretación/recomendación separados, cerrando en una
síntesis de "Insights clave" y una sección de Recomendaciones aparte del análisis. La página de
Forecasting & ML (`reports/web/forecasting.html`) sigue el mismo criterio en 4 bloques técnicos:
validación del forecast (metodología + model governance por serie), comparación de modelos, gobierno de
riesgo/anomalías y segmentación accionable — cierra con qué cambiaría para llevar esto a producción.

## 1. Data audit y alcance

**Calidad del dato (auditoría inicial):** 55 filas × 33 columnas (2 texto + 31 numéricas int64), **0
valores nulos**, **0 filas duplicadas**, coherencia exacta (`Total_domestic_consumption` = suma de las
30 columnas anuales en las 55 filas, diferencia máxima 0, sin negativos). Es un dataset limpio, sin
pasos de limpieza previos necesarios. Dos países (Equatorial Guinea, Nepal) tienen serie completa en
cero y se excluyen del modelado (quedan 53 series válidas).

**Alcance:** el dataset contiene **consumo doméstico** por país y año, no precios. La petición de
"rangos de precios futuros" se reformula como **rangos de consumo futuro** (con intervalos), que es lo
que los datos permiten responder con rigor. Incorporar precio requeriría una fuente externa (ICO
Composite Price, futuros de arábica/robusta en ICE) — recomendación de roadmap. El dataset tampoco
explica *causalidad* (por qué cambia el consumo), solo *magnitud* (cuánto cambia).

## 2. Insights clave

1. **Crecimiento sostenido con quiebre reciente.** El consumo global casi se triplicó (1990/91: 1,171M →
   2018/19: 3,015M tazas, CAGR 3.3%/año) pero **cayó 0.5%** en 2019/20 (2,999M) — primer retroceso
   interanual relevante de la serie. No es generalizado: Colombia (+14M) y Venezuela (−16M) son los
   mayores movimientos individuales, con Brasil, Philippines, Honduras e India también a la baja, pero
   Ethiopia y Viet Nam compensando parte de la caída.
2. **Concentración extrema, no un 80/20 típico.** Brasil concentra **45%** del consumo doméstico
   acumulado; los 10 países más grandes explican **87%**. La media (57M) es **~12x** la mediana (5M) —
   evidencia directa de la asimetría.
3. **Asia: menos mercados, pero más grandes.** La mediana de consumo por país en Asia (84M, 9 países)
   supera a la de América (16M, 20 países) pese a tener menos de la mitad de los mercados — la
   concentración no es solo por país (Brasil), también es estructural por continente.
4. **El crecimiento reciente no siempre es confiable.** Côte d'Ivoire tiene la mayor volatilidad
   interanual del dataset (CV=0.99); varios países de alto CAGR reciente (ej. Tanzania) son también los
   más volátiles — un año atípico puede simular "alto crecimiento" donde hay inestabilidad.
5. **El tipo dominante no es el de mayor momentum.** Arabica/Robusta lidera en volumen (53.5%), pero
   Robusta/Arabica crece más rápido (4.7%/año); Robusta (puro) es el único tipo en declive (CAGR
   reciente −0.7%, solo 2.3% del consumo).

**Limitaciones del análisis:** sin variables explicativas (clima, PIB, tipo de cambio) no es posible
establecer causalidad; Oceanía está representada por un solo país (Papúa Nueva Guinea, N=1, no
generalizable); las anomalías detectadas (§4) no fueron validadas contra una fuente externa.

## 3. Forecasting de consumo (proyección a 2030)

**Metodología (validación, no solo resultado):** selección de modelo por **backtest de origen móvil**
(rolling-origin) — 5 orígenes, 1 paso adelante por origen, mínimo 18 observaciones de entrenamiento en
el primer origen (evita fugas de información: cada evaluación solo ve datos anteriores al punto que
predice). Métrica principal: MAPE, contra un **baseline naive** (persistencia del último valor). Ajuste
en escala log, intervalos empíricos P10–P90 que se **ensanchan con la raíz del horizonte** (paseo
aleatorio) en vez de mantenerse planos. El forecast publicado se proyecta a **2030** (+11 años desde el
último dato observado, 2019/20) reentrenando con todo el histórico — más allá de la ventana
directamente validada a 1 paso, límite documentado explícitamente en el informe web. Series cortas o
degeneradas caen a naive automáticamente o se excluyen del modelado.

| Serie | Modelo | MAPE backtest | Baseline (naive) | 2019/20 | Proyección 2030 |
|---|---|---|---|---|---|
| **Global** | ARIMA | 1.08% | 1.65% | 2,999M | **~3,156M** |
| Brasil | ARIMA | 1.48% | 1.91% | 1,320M | ~1,264M |
| Indonesia | naive | 1.67% | 1.67% | 288M | ~288M |
| Etiopía | Holt-Winters | 0.48% | 1.91% | 227M | ~283M |
| Filipinas | ARIMA | 2.85% | 3.51% | 195M | ~250M |
| Viet Nam | ARIMA | 1.32% | 3.65% | 159M | ~174M |

MAPE medio de backtest entre las 53 series de país: 1.45% (mediana 0.67%) vs. 1.77% del baseline naive —
el modelo seleccionado gana, pero por un margen modesto en la mayoría de las series maduras.

**Coherencia jerárquica:** el forecast global directo (~3,156M) y la suma de forecasts por país
difieren en −2.2% — dentro de lo esperado al modelar cada nivel por separado, y una señal de riesgo
más a monitorear que un error a corregir.

## 4. Machine Learning aplicado (más allá del forecasting)

- **Modelo global "pooled" (HistGradientBoosting).** Una sola máquina entrenada sobre las 53 series
  apiladas, compartiendo señal entre países y prediciendo la **tasa de crecimiento** (los árboles no
  extrapolan niveles). En una comparación **justa** sobre el mismo holdout de 5 años (esta comparación se
  mantiene a 5 años aunque el forecast publicado llegue a 2030 — es la ventana ya validada) obtiene 4.9%
  de MAPE mediano frente a **0.8%** de los modelos clásicos por-serie. Conclusión basada en evidencia:
  para estas series maduras y suaves, los modelos clásicos ganan; el modelo simple se elige tras evaluar
  el enfoque ML moderno, no por defecto.
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

*(Síntesis de negocio, separada del EDA — cada recomendación se apoya en un hallazgo de §1–§5, no en
juicio no sustentado.)*

1. **Priorizar mercados medianos de alto crecimiento y baja volatilidad**: Vietnam (+8.2%/año), Tailandia
   (+7.2%/año), Filipinas (+6.0%/año), Colombia (+4.7%/año) — tamaño no saturado y señal confiable.
   Varios son **Robusta/Arabica** — alinear la oferta de exportación a ese perfil.
2. **Retener a Brasil como mercado ancla, no como motor de crecimiento**: 45% del consumo mundial con
   crecimiento modesto (+1.8%/año) — la estrategia comercial ahí es defender, no expandir.
3. **Validar antes de comprometer presupuesto**: cruzar cualquier mercado candidato contra su volatilidad
   histórica (insight 4, §2) antes de asignar inversión basada solo en su CAGR reciente.
4. **Vigilar la caída de 2019/20** en la próxima campaña del ICO (2020/21) para confirmar si es puntual o
   un cambio de tendencia global.
5. **Validar las 6 anomalías detectadas** (Brasil, Tanzania, Togo, Vietnam, Côte d'Ivoire, Gabón) contra
   la fuente antes de basar decisiones en sus años atípicos.
6. **Para abordar el pedido original de "precio"**, incorporar una fuente externa de precios y repetir la
   metodología de forecasting sobre esa variable (fase 2).

## 7. Bonus — Integración de IA generativa / LLM (propuesta)

Página placeholder en el informe web (`reports/web/genai.html`), en evaluación. Dos propuestas concretas:
- **(a) Narrativas automáticas de insights**: un LLM (Claude) redacta insights y recomendaciones a partir
  de los resultados ya calculados por el pipeline (rankings, perfiles de cluster, forecast). Bajo riesgo:
  resume resultados validados, no interpreta datos crudos.
- **(b) Asistente text-to-pandas**: responde preguntas de negocio en lenguaje natural generando y
  ejecutando código pandas sobre los datos y el forecast, con la tabla/gráfico de soporte.

Requisitos si se implementa: SDK `anthropic`, `ANTHROPIC_API_KEY` vía `.env` (no versionada).

**Productivización real** (fuera de alcance de este entregable): monitoreo de drift (alertar cuando el
error real se aleje del MAPE de backtest, no solo reentrenar en fecha fija), reentrenamiento por
crop-year, tracking de experimentos (MLflow), validación temporal continua (repetir el rolling-origin
backtest en cada campaña nueva), revisión humana obligatoria para mercados anómalos antes de usar su
forecast en una decisión de presupuesto, y empaquetar `src/` como paquete instalable con un job de CI que
falle si el MAPE de backtest empeora entre versiones — regresión de modelo, no solo de código. Una API
ligera (FastAPI) serviría el forecast y el asistente de IA generativa.

---
*Detalle técnico y visualizaciones interactivas en `reports/web/` y en los notebooks
`notebooks/01_eda.ipynb`, `02_forecasting.ipynb`, `03_clustering.ipynb`.*
