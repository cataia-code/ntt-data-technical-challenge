# Diseño: Arquitectura de servicios, MLOps ligero y CI/CD gratuito

**Fecha:** 2026-07-12
**Estado:** Aprobado, pendiente de plan de implementación

## Contexto

El reto técnico ML de NTT DATA (High Garden Coffee) ya tiene una solución funcional: EDA, forecasting
a 3 granularidades, ML complementario (pooled model, clasificador, anomalías, PCA), clustering, y un
informe web interactivo (`reports/web/`), todo generado por `scripts/run_pipeline.py` +
`scripts/build_report.py` sobre módulos en `src/`. El código funciona y los resultados ya están
documentados en `reports/executive_summary.md`.

El repositorio ya está inicializado localmente y el usuario pide evolucionar esto a un sistema
automatizado, modular, con buenas prácticas de MLOps, tests unitarios/integración, y un repositorio
GitHub con CI/CD que despliegue gratuitamente el sitio generado.

## Objetivo

Refactorizar el pipeline existente (sin rehacer la lógica de cálculo, que ya está validada) hacia una
arquitectura de **servicios internos modulares** orquestados por un pipeline/CLI, con testing en capas,
validación de datos como paso explícito, artefactos de métricas versionables, y un flujo CI/CD en GitHub
Actions que valida cada cambio y despliega el sitio a GitHub Pages de forma gratuita.

Explícitamente **fuera de alcance**: exponer los servicios como API HTTP/microservicios, MLflow u otras
plataformas de tracking con servidor, contenedores/Docker, y cualquier infraestructura de pago. El
dataset es pequeño (55 países × 30 años) y el entregable final es un sitio estático — la complejidad debe
mantenerse proporcional a eso.

## Arquitectura de servicios

Paquete instalable `coffee_analytics/` (src-layout), reemplaza el `src/` actual:

```
coffee_analytics/
  __init__.py
  config.py                  # PipelineConfig: rutas, hiperparámetros, semillas, fast_mode
  services/
    __init__.py
    data_service.py           # load_raw, build_long_dataset, validate_coherence (de data_prep.py)
    eda_service.py             # business_views.py + features.py → EDAResult
    forecasting_service.py     # forecasting.py + pooled_model.py → ForecastResult (soporta fast_mode)
    ml_service.py               # ml_extra.py (anomalías, clasificador) + clustering.py → MLResult
    report_service.py           # plotly_charts.py + build_report.py → escribe reports/web/*.html
  pipeline.py                  # orquesta data → eda → forecasting → ml → report; produce el bundle
  cli.py                        # entrypoint `coffee-analytics run [--fast]` / `coffee-analytics report`
```

**Contrato de cada servicio:** función `run(input: XInput) -> XResult` (dataclasses tipadas), sin efectos
secundarios ocultos — solo `report_service` escribe a disco. Esto es lo que permite testear cada servicio
en aislamiento y lo que hace "modular" el sistema (servicios internos con interfaces claras, **no**
servidores HTTP — decisión explícita del usuario dado el tamaño del dataset y que el entregable es
estático).

La lógica interna de cada módulo (`data_prep.py`, `features.py`, `business_views.py`, `forecasting.py`,
`pooled_model.py`, `ml_extra.py`, `clustering.py`, `plotly_charts.py`) se **mueve y envuelve**, no se
reescribe — minimiza el riesgo de alterar resultados ya validados y documentados en
`executive_summary.md`.

## Validación de datos, configuración y MLOps ligero

- **`config.py`**: dataclass `PipelineConfig` centraliza rutas, año de corte, hiperparámetros de
  forecasting, semillas aleatorias (hoy dispersas en `pooled_model.py`/`ml_extra.py`), y un flag
  `fast_mode: bool` que reduce el rolling-backtest (menos orígenes de ventana / menos series) para uso en
  CI de PRs.
- **Validación de datos como paso explícito y bloqueante**: `data_service.validate()` reutiliza
  `validate_coherence` (Total == suma de columnas anuales, sin negativos) pero **falla el pipeline** con
  una excepción clara si la coherencia se rompe — hoy esto solo se verifica en tests, no bloquea una
  ejecución real del pipeline.
- **Artefactos de métricas versionables**: `reports/data/metrics.json` con MAPE de backtest por serie,
  comparación pooled-vs-clásico, accuracy del clasificador vs. baseline, `pipeline_version` (hash corto de
  git) y `generated_at`. Da trazabilidad de cómo evolucionan las métricas sin necesitar MLflow ni servidor
  de tracking.
- **Reproducibilidad**: semillas fijas centralizadas en `config.py`; `pyproject.toml` con dependencias
  ancladas (versión exacta) en vez de rangos abiertos.
- **Versionado de datos**: `data/raw/coffee_db.parquet` se versiona vía **Git LFS** (`.gitattributes`
  con `data/raw/*.parquet filter=lfs`), no como blob normal — es el único insumo inmutable del reto, sin
  fuente externa de la que descargarse, y sacarlo de git rompería la reproducibilidad de CI/clones. LFS
  evita inflar el historial de git con binarios (la práctica que se busca evitar) sin añadir fricción
  operativa. `data/processed/` (long dataset + features) se **ignora por completo**: es 100% derivable de
  `data/raw/` + código, no debe versionarse.

## Testing

```
tests/
  unit/                        # por servicio, fixtures sintéticas pequeñas, <10s en total
    test_data_service.py
    test_eda_service.py
    test_forecasting_service.py
    test_ml_service.py
    test_report_service.py
  integration/
    test_pipeline_fast.py      # pipeline end-to-end, fast_mode=True, datos reales — corre en CI de PR
    test_pipeline_full.py      # @pytest.mark.slow — pipeline completo real — solo en push a main
```

`tests/test_pipeline.py` (coherencia + smoke, ya existente) se reparte entre `unit/test_data_service.py`
(tests de coherencia) e `integration/test_pipeline_full.py` (smoke tests de modelos/clustering).

## CI/CD y despliegue

`.github/workflows/ci.yml` — en cada push y PR a `develop` (rama de integración) y en PRs hacia `master`:
1. `ruff check .` + `ruff format --check .`
2. `pytest tests/unit tests/integration/test_pipeline_fast.py`
3. Bloquea el merge si algo falla

`.github/workflows/deploy.yml` — en push a `master` (rama de release/producción), después de que `ci.yml`
haya pasado en el PR develop→master:
1. `pytest -m slow` (pipeline completo real, ~3-5 min) como gate final
2. `coffee-analytics run` (pipeline completo) → `coffee-analytics report` (genera `reports/web/*.html`)
3. Publica `reports/web/` a **GitHub Pages** vía `actions/upload-pages-artifact` +
   `actions/deploy-pages` (oficial, gratis en repos públicos)

Los artefactos generados (`reports/web/*.html`, `reports/data/`, CSVs) **no se commitean de vuelta** a
ninguna rama — quedan solo como artefacto de build de esa ejecución y se regeneran siempre desde
código+datos fuente. El repo Git permanece limpio: solo versiona código, `data/raw/coffee_db.parquet`
(vía LFS), notebooks, `docs/` y `reports/executive_summary.md` (documento narrativo curado a mano).

**Buenas prácticas DevOps adicionales incluidas:**
- **Caché de pip** en `actions/setup-python` (`cache: pip`) — acelera cada corrida de CI reutilizando
  dependencias entre ejecuciones, sin infraestructura extra.
- **Pre-commit hooks** (`.pre-commit-config.yaml`): `ruff check --fix` + `ruff format` en cada commit
  local — detecta problemas de estilo antes de llegar a CI, mismo linter que ya usa `ci.yml` (una sola
  herramienta, sin duplicar configuración).
- **Dependabot** (`.github/dependabot.yml`): actualizaciones automáticas semanales de dependencias de
  `pip` y de las Actions usadas en los workflows — gratis, nativo de GitHub, sin servidor propio.
- **Escaneo de secretos en CI**: paso con `gitleaks` (Action open-source, gratis) en `ci.yml` para evitar
  que credenciales (p.ej. una `ANTHROPIC_API_KEY` si se retoma la página GenAI) se cuelen en un commit.

## Empaquetado y migración

- `pyproject.toml` (setuptools, src-layout) reemplaza `requirements.txt`: define el paquete
  `coffee-analytics`, dependencias ancladas, entrypoint de consola `coffee-analytics`, y configuración de
  `ruff` y `pytest` (marker `slow`).
- `.gitignore` se amplía: excluye `.claude/` (herramientas del agente/IDE, no forman parte del proyecto),
  `data/processed/` (derivado, regenerable), `reports/web/*.html`, `reports/data/`, `reports/*.csv`.
- `docs/` agrupa documentación no-código: el PDF del enunciado (`docs/Reto Tecnico ML.pdf`) y las specs
  de diseño (`docs/superpowers/specs/`).
- `scripts/run_pipeline.py` y `scripts/build_report.py` se retiran a favor de `coffee_analytics/cli.py`.
- `reports/web/genai.html` (placeholder GenAI) queda sin cambios — fuera de alcance de este diseño.

## Repositorio Git y GitHub

Repo: [github.com/cataia-code/ntt-data-technical-challenge](https://github.com/cataia-code/ntt-data-technical-challenge)
(creado vacío y público por el usuario, para que GitHub Pages sea gratuito).

**Modelo de ramas (GitFlow simplificado):**
- **`develop`** — rama de integración y **rama por defecto** del repositorio en GitHub. Todo el trabajo
  (features, fixes, este mismo refactor) se hace en ramas cortas y se fusiona a `develop` vía PR. `ci.yml`
  corre en cada push/PR contra `develop`.
- **`master`** — rama de release/producción, protegida. Solo recibe merges desde `develop` (vía PR) cuando
  el trabajo está listo para publicarse. Un push a `master` dispara `deploy.yml`: pipeline completo +
  publicación a GitHub Pages. `master` siempre refleja lo que está desplegado.

Flujo de trabajo:
1. Se inicializa git localmente (ya hecho), rama de trabajo `develop`.
2. Se añade el remoto `origin` apuntando al repo de GitHub y se hace el push inicial de `develop`, con
   confirmación del usuario antes de cualquier push.
3. En GitHub: se configura `develop` como rama por defecto, se crea `master` a partir de `develop` cuando
   el refactor esté listo para el primer release, y se protegen ambas ramas (`master` requiere PR +
   CI en verde; `develop` requiere CI en verde).
4. El usuario habilita GitHub Pages con fuente "GitHub Actions" en la configuración del repo (paso manual
   de un clic, documentado en el plan de implementación).

## Fuera de alcance (explícito)

- API HTTP / microservicios por etapa.
- MLflow, Docker, o cualquier infraestructura con servidor.
- Commit automático de resultados generados de vuelta a `main`.
- Cambios a `reports/web/genai.html`.
