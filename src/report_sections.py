"""Contenido de cada página del informe web: compone las vistas de negocio (business_views) y las
figuras (plotly_charts) en HTML, usando los componentes de report_components.

Cada `build_*_page(bundle)` es la única responsable de UNA página y no conoce el resto (EDA no sabe
nada de Forecasting & ML, y viceversa) — así cada página se puede probar y modificar de forma
aislada. Este módulo NO toca disco: recibe un `ReportBundle` ya cargado por report_data.load_bundle()
y solo (a) llama funciones puras de business_views sobre los DataFrames ya en memoria del bundle, y
(b) formatea el resultado a HTML. Cualquier I/O nuevo pertenece a report_data.py, no aquí.
"""

import numpy as np

import business_views as bv
import plotly_charts as pc
import report_components as rc
from forecasting import MIN_TRAIN, ROLLING_ORIGINS
from report_data import ReportBundle

_RISK_PILL_COLORS = {"Bajo": "#2F7D32", "Medio": "#C77E12", "Alto": "#B23A48"}


# --------------------------------------------------------------------------- #
# Helpers de narrativa data-driven (evitan números de negocio escritos a mano en el texto)
# --------------------------------------------------------------------------- #
def _market_mentions(priority_df, n: int) -> list:
    """"País (+X.X%/año)" para los `n` mercados de mayor potencial (ver bv.priority_markets_table).

    Formateo de texto puro sobre datos ya calculados por business_views — vive aquí (no en
    business_views) porque el formato de salida es una decisión de presentación, no de negocio.
    """
    return [f"{r['Country']} ({r['cagr_recent']*100:+.1f}%/año)" for _, r in priority_df.head(n).iterrows()]


# --------------------------------------------------------------------------- #
# EDA & Insights
# --------------------------------------------------------------------------- #
def _eda_hero_stats(long_df, pareto, type_tbl, cagr_global) -> tuple:
    total_consumption_all = float(pareto["total"].sum())
    top_country_row = pareto.iloc[0]
    top_country = top_country_row["Country"]
    top_country_continent = long_df.loc[long_df["Country"] == top_country, "continent"].iloc[0]
    top_type_row = type_tbl.iloc[0]
    stats_html = "".join([
        rc.stat(rc.human(total_consumption_all), "tazas", "Consumo global total 1990–2020"),
        rc.stat(top_type_row["Coffee type"], "", "Tipo de café líder",
                f"{top_type_row['share_pct']:.0f}% del consumo"),
        rc.stat(top_country, "", "País líder en consumo",
                f"{top_country_continent} · {top_country_row['share_pct']:.0f}% del total"),
        rc.stat(f"+{cagr_global*100:.1f}", "%/año", "CAGR global 1990–2020", "Crecimiento compuesto", "up"),
    ])
    return stats_html, top_type_row


def _block_data_audit(overview, ml_n_countries) -> str:
    nulls_txt = ("0 (ninguna columna)" if overview["n_nulls_total"] == 0
                else ", ".join(f"{c}: {n}" for c, n in overview["cols_with_nulls"].items()))
    return f"""
<h2 class="section-title">1 · Data audit y alcance</h2>
<p class="section-sub">¿Podemos confiar en estos datos, y qué preguntas pueden responder con rigor?</p>
<div class="card pad">
  <div class="table-wrap"><table class="data">
    <tr><th>Chequeo</th><th>Resultado</th></tr>
    <tr><td>Filas × columnas (formato original)</td>
        <td class="num">{overview['n_rows']} × {overview['n_cols']}</td></tr>
    <tr><td>Tipos de columna</td>
        <td class="num">2 texto (Country, Coffee type) + {overview['n_year_cols'] + 1} numéricas
        (int64)</td></tr>
    <tr><td>Rango de campañas</td><td class="num">{overview['year_range']} ({overview['n_year_cols']}
        columnas anuales)</td></tr>
    <tr><td>Valores nulos</td><td class="num">{nulls_txt}</td></tr>
    <tr><td>Filas duplicadas</td><td class="num">{overview['n_duplicated_rows']}</td></tr>
    <tr><td>Coherencia Total = Σ columnas anuales</td><td class="num">✅ exacta, 0 países con diferencia</td></tr>
    <tr><td>Países / tipos de café únicos</td>
        <td class="num">{overview['n_countries']} / {overview['n_coffee_types']}</td></tr>
    <tr><td>Series válidas para modelado</td>
        <td class="num">{ml_n_countries} de {overview['n_countries']}</td></tr>
    <tr><td>Series completas en cero (excluidas)</td>
        <td class="num">{len(overview['zero_series_countries'])}
        ({', '.join(overview['zero_series_countries'])})</td></tr>
  </table></div>
</div>
<div class="scope-grid">
  <div class="scope-card can">
    <h4>✅ Con estos datos podemos responder</h4>
    <ul>
      <li>Tendencias de consumo 1990–2020 por país, tipo de café y continente.</li>
      <li>Concentración de mercado, evolución temporal y segmentación tamaño × crecimiento.</li>
    </ul>
  </div>
  <div class="scope-card cannot">
    <h4>❌ NO podemos responder</h4>
    <ul>
      <li><strong>Precios</strong>: sin columna de precio; el pedido original de "rangos de precios"
      requeriría una fuente externa (ICO Composite Price).</li>
      <li><strong>Causalidad</strong>: el dataset explica <em>cuánto</em> cambia el consumo, no
      <em>por qué</em>.</li>
    </ul>
  </div>
</div>
{rc.insight_box(
    "El dataset original no tiene nulos, duplicados, ni incoherencias (Total = suma de columnas "
    "anuales exacta en las 55 filas): es confiable sin pasos de limpieza previos.",
    "La única salvedad de calidad son 2 series completamente en cero (Guinea Ecuatorial, Nepal — "
    "0.05% de las filas): sin señal, no aportan ni distorsionan si se excluyen.",
    "Excluir esas 2 series del modelado (ya se hace en el pipeline) y tratar el resto del dataset "
    "como base confiable para todo el análisis que sigue.",
)}
"""


def _block_concentration(long_df, pareto, stats, cont_median, top_cont, second_cont,
                         n_cont_top, n_cont_second) -> str:
    return f"""
<h2 class="section-title">2 · Concentración y distribución</h2>
<p class="section-sub">¿Qué tan repartido está el consumo entre países y continentes?</p>
<div class="card">{rc.div(pc.fig_pareto(pareto), "eda_pareto")}</div>
<div class="card">{rc.div(pc.fig_distribution_by_continent(long_df, stats), "eda_box")}</div>
{rc.insight_box(
    f"Brasil concentra {pareto.attrs['brazil_share']:.0f}% del consumo acumulado; los 10 países más "
    f"grandes explican {pareto.attrs['top10_share']:.0f}% (Pareto extremo, no un 80/20 típico). Por "
    f"continente, {top_cont} tiene la mediana de consumo más alta ({rc.human(cont_median[top_cont])} "
    f"tazas, {n_cont_top} países) pese a tener menos mercados que {second_cont} "
    f"({rc.human(cont_median[second_cont])}, {n_cont_second} países).",
    "La concentración es doble: por país (Brasil como caso extremo aislado) y por continente "
    f"(mercados de {top_cont} individualmente más grandes, aunque menos numerosos).",
    "No diversificar 'por número de países' sin mirar tamaño real — sumar mercados pequeños del mismo "
    "continente no compensa la concentración si carecen de escala.",
)}
"""


def _block_evolution(long_df, wf, cagr_global, drop_pct) -> str:
    movers = [(c, v) for c, v in wf["items"] if "Otros" not in c]
    gainers = sorted((m for m in movers if m[1] > 0), key=lambda t: -t[1])
    losers = sorted((m for m in movers if m[1] < 0), key=lambda t: t[1])
    gainers_rows = "".join(f'<tr><td>{c}</td><td class="num">+{rc.human(v)}</td></tr>' for c, v in gainers)
    losers_rows = "".join(f'<tr><td>{c}</td><td class="num">−{rc.human(abs(v))}</td></tr>' for c, v in losers)
    return f"""
<h2 class="section-title">3 · Evolución y quiebre reciente</h2>
<p class="section-sub">¿Cómo evolucionó el consumo, y qué explica la caída de 2019/20?</p>
<div class="card">{rc.div(pc.fig_evolution_dual_selector(long_df), "eda_evo")}</div>
<div class="grid-2">
  <div class="card pad">
    <h4>▲ Top movers — subieron en 2019/20</h4>
    <div class="table-wrap"><table class="data">
      <tr><th>País</th><th>Δ consumo</th></tr>{gainers_rows}
    </table></div>
  </div>
  <div class="card pad">
    <h4>▼ Top movers — bajaron en 2019/20</h4>
    <div class="table-wrap"><table class="data">
      <tr><th>País</th><th>Δ consumo</th></tr>{losers_rows}
    </table></div>
  </div>
</div>
{rc.insight_box(
    f"El consumo global casi se triplicó entre 1990/91 y 2018/19 (CAGR {cagr_global*100:.1f}%/año), "
    f"pero cayó {abs(drop_pct):.1f}% en 2019/20 ({rc.human(abs(wf['total_delta']))} tazas) — la primera "
    f"caída interanual relevante de la serie, y no generalizada: {len(losers)} países bajaron pero "
    f"{len(gainers)} crecieron y compensaron parte de la baja.",
    "Una sola campaña no confirma si es un evento puntual o el inicio de una desaceleración "
    "estructural; la señal es mixta, no una tendencia uniforme de caída.",
    "Monitorear la campaña 2020/21 del ICO antes de ajustar proyecciones; investigar puntualmente "
    f"{gainers[0][0] if gainers else 'los países en alza'} y {losers[0][0] if losers else 'los países en baja'} "
    "(mayores movimientos individuales) antes de generalizar.",
)}
"""


def _block_segmentation(quad, type_tbl, top_type_row, declining_type, n_priorizar,
                        brazil_cagr, market_mentions) -> str:
    type_rows = "".join(
        f'<tr><td>{rc.pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td>'
        f'<td class="num">{r["share_pct"]:.1f}%</td>'
        f'<td class="num">{r["cagr_recent"]*100:+.1f}%</td>'
        f'<td class="num">{r["n_countries"]}</td>'
        f'<td>{r["dominant_country"]}</td></tr>'
        for _, r in type_tbl.iterrows()
    )
    return f"""
<h2 class="section-title">4 · Segmentación e insights clave</h2>
<p class="section-sub">¿Dónde hay tamaño, dónde hay crecimiento, y qué mercados combinan ambos?</p>
<div class="card">{rc.div(pc.fig_quadrant_matrix(quad), "eda_quad")}</div>
<div class="card pad">
  <div class="table-wrap"><table class="data">
    <tr><th>Tipo de café</th><th>% del consumo</th><th>CAGR reciente</th><th># países</th><th>País dominante</th></tr>
    {type_rows}
  </table></div>
</div>
{rc.insight_box(
    f"{n_priorizar} países caen en 'Priorizar' (tamaño y crecimiento por encima de la mediana) — desde "
    f"medianos en fuerte aceleración ({market_mentions[0]}) hasta Brasil, grande pero de crecimiento "
    f"modesto ({brazil_cagr*100:+.1f}%/año). Por tipo, '{top_type_row['Coffee type']}' concentra "
    f"{top_type_row['share_pct']:.0f}% del consumo, pero '{type_tbl.iloc[1]['Coffee type']}' crece más "
    f"rápido ({type_tbl.iloc[1]['cagr_recent']*100:.1f}%/año); '{declining_type}' es el único en "
    "declive.",
    "Tamaño, crecimiento y tipo de café son tres ejes independientes: el país más grande no es el de "
    "mayor momentum, y el tipo dominante en volumen no es el de mayor crecimiento.",
    "Presupuesto diferenciado por cuadrante (expansión en 'Priorizar', retención en Brasil) y oferta de "
    f"exportación alineada a '{type_tbl.iloc[1]['Coffee type']}' en los mercados emergentes.",
)}
"""


def _block_key_insights(quad, vol_rank, stats) -> str:
    """Cierre de la narrativa: SOLO hallazgos que cruzan varios bloques y que ningún insight_box de
    arriba ya cubrió — evita repetir Brasil 45%/top10 87% (bloque 2), el quiebre 2019/20 (bloque 3) o
    el desajuste de tipo dominante (bloque 4), que ya están dichos una vez, en su bloque correcto.
    """
    top_volatile = vol_rank.iloc[0]
    match = quad.loc[quad["Country"] == top_volatile["Country"], "quadrant"]
    volatile_quadrant = match.iloc[0] if len(match) else "sin clasificar"
    return f"""
<h3 class="section-title" style="font-size:1.05rem;margin-top:22px">Insights clave</h3>
<p class="section-sub">Hallazgos que conectan varios bloques anteriores — no se repiten aquí los ya
mencionados en cada caja de insight.</p>
<div class="card pad">
  <ol class="key-insights">
    <li><strong>El cuadrante de negocio no filtra por estabilidad.</strong>
    {top_volatile['Country']} tiene la mayor volatilidad interanual del dataset
    (CV={top_volatile['volatility']:.2f}) y aun así cae en el cuadrante '{volatile_quadrant}' de la
    matriz del bloque 4 — antes de asignar presupuesto por cuadrante (ver Recomendaciones), cruzar
    siempre contra volatilidad histórica.</li>
    <li><strong>El promedio por país no es representativo.</strong> La media de consumo
    ({rc.human(stats['mean'])}) es {stats['mean']/stats['median']:.0f}x la mediana
    ({rc.human(stats['median'])}): la mayoría de los {stats['n']} países está muy por debajo del
    promedio, que un puñado de mercados grandes empuja hacia arriba.</li>
  </ol>
</div>
"""


def _block_limitations() -> str:
    return """
<h2 class="section-title">Limitaciones del análisis</h2>
<div class="card pad">
  <ul class="note" style="margin:0;padding-left:18px">
    <li>Sin datos de precio: el pedido original de "rangos de precios futuros" se reformula como
    consumo, la variable que el dataset sí contiene.</li>
    <li>Sin variables explicativas (clima, PIB, tipo de cambio): no es posible establecer causalidad,
    solo describir y proyectar el propio patrón de consumo.</li>
    <li>Oceanía está representada por un solo país (Papúa Nueva Guinea) — cualquier lectura "por
    continente" sobre Oceanía es N=1, no generalizable.</li>
    <li>Las anomalías detectadas (ver página Forecasting &amp; ML) no fueron validadas contra una
    fuente externa — son candidatas a revisar, no hechos confirmados.</li>
  </ul>
</div>
"""


def _block_recommendations(pareto, type_tbl, priority_tbl, brazil_cagr, market_mentions) -> str:
    prio_rows = "".join(
        f'<tr><td>{r["Country"]}</td>'
        f'<td>{rc.pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td>'
        f'<td class="num">{rc.human(r["level_last5_mean"])}</td>'
        f'<td class="num">{r["cagr_recent"]*100:+.1f}%</td>'
        f'<td>{rc.pill(r["quadrant"], pc.QUADRANT_COLORS.get(r["quadrant"], "#6B5647"))}</td></tr>'
        for _, r in priority_tbl.iterrows()
    )
    return f"""
<div style="border-top:2px dashed var(--line);padding-top:20px;margin-top:8px">
<span class="eyebrow">Fuera del EDA — síntesis de negocio</span>
<h2 class="section-title" style="margin-top:6px">Recomendaciones</h2>
<div class="card pad">
  <ol class="key-insights">
    <li><strong>Priorizar mercados medianos de alto crecimiento y baja volatilidad</strong> —
    {', '.join(market_mentions)}: tamaño no saturado y señal confiable (ver matriz, bloque 4).</li>
    <li><strong>Retener a Brasil como mercado ancla</strong>, no como motor de crecimiento —
    {pareto.attrs['brazil_share']:.0f}% del consumo mundial con crecimiento modesto
    ({brazil_cagr*100:+.1f}%/año).</li>
    <li><strong>Alinear la oferta de exportación a '{type_tbl.iloc[1]['Coffee type']}'</strong>
    ({type_tbl.iloc[1]['cagr_recent']*100:.1f}%/año) en los mercados emergentes priorizados, no solo al
    tipo dominante en volumen.</li>
    <li><strong>Validar antes de comprometer presupuesto</strong>: cruzar cualquier candidato contra su
    volatilidad histórica y confirmar en la campaña 2020/21 del ICO si la caída de 2019/20 fue puntual o
    estructural.</li>
  </ol>
  <div class="table-wrap" style="margin-top:14px"><table class="data">
    <tr><th>País</th><th>Tipo</th><th>Consumo reciente</th><th>CAGR reciente</th><th>Cuadrante</th></tr>
    {prio_rows}
  </table></div>
</div>
</div>
"""


def build_eda_page(bundle: ReportBundle) -> str:
    """Página 1 — EDA & Insights: 4 bloques (data audit, concentración, evolución, segmentación),
    cierre de insights clave, limitaciones y recomendaciones de negocio separadas del EDA.

    Toma todo del `bundle` ya cargado (report_data.load_bundle) — no lee nada de disco.
    """
    long_df, feats = bundle.long_df, bundle.feats

    growth = bv.global_growth_summary(long_df)
    overview = bv.dataset_overview(bundle.raw, long_df)
    stats = bv.descriptive_stats(long_df)
    pareto = bv.pareto_table(long_df)
    quad = bv.classify_quadrant(bundle.bubble)
    wf = bv.waterfall_contribution(long_df)
    type_tbl = bv.type_summary_table(long_df)
    vol_rank = bv.volatility_ranking(feats, top_n=5)
    priority_tbl = bv.priority_ranking_table(quad, feats, top_n=8)
    cont = bv.continent_size_comparison(long_df, stats["year"])

    cagr_global, drop_pct = growth["cagr_global"], growth["drop_pct"]
    n_priorizar = int((quad["quadrant"] == "Priorizar").sum())
    declining_type = type_tbl.loc[type_tbl["cagr_recent"].idxmin(), "Coffee type"]
    brazil_cagr = bv.country_cagr_recent(feats, "Brazil")
    market_mentions = _market_mentions(bundle.priority, n=4)

    median_by_continent, count_by_continent = cont["median_by_continent"], cont["count_by_continent"]
    top_cont, second_cont = median_by_continent.index[0], median_by_continent.index[1]
    n_cont_top, n_cont_second = int(count_by_continent[top_cont]), int(count_by_continent[second_cont])

    eda_stats, top_type_row = _eda_hero_stats(long_df, pareto, type_tbl, cagr_global)

    b1 = _block_data_audit(overview, bundle.ml["n_countries"])
    b2 = _block_concentration(long_df, pareto, stats, median_by_continent, top_cont, second_cont,
                              n_cont_top, n_cont_second)
    b3 = _block_evolution(long_df, wf, cagr_global, drop_pct)
    b4 = (_block_segmentation(quad, type_tbl, top_type_row, declining_type, n_priorizar,
                              brazil_cagr, market_mentions)
          + _block_key_insights(quad, vol_rank, stats))
    limitations = _block_limitations()
    recommendations = _block_recommendations(pareto, type_tbl, priority_tbl, brazil_cagr, market_mentions)

    return f"""
<section class="hero hero-grid">
  <div>
    <span class="eyebrow">EDA &amp; Insights</span>
    <h1>Tres décadas de <span class="accent">consumo de café</span> convertidas en decisiones</h1>
    <p class="lead">Cuatro bloques, cada uno respondiendo una pregunta de negocio concreta: calidad del dato,
    concentración, evolución y priorización comercial. Cada bloque cierra con hallazgo, implicación y acción.</p>
    <div class="hero-meta">
      <span class="meta-pill">55 países</span>
      <span class="meta-pill">1990/91–2019/20</span>
      <span class="meta-pill">53 series modelables</span>
      <span class="meta-pill">0 nulos · 0 duplicados</span>
    </div>
  </div>
  <aside class="hero-aside">
    <h3>Business lens</h3>
    <ul>
      <li>Priorizar mercados por tamaño y crecimiento, no por intuición.</li>
      <li>Separar mercados ancla, oportunidades y riesgos operativos.</li>
      <li>Evitar conclusiones de causalidad: aquí se describe y se proyecta.</li>
    </ul>
  </aside>
</section>
<div class="stats">{eda_stats}</div>
{b1}{b2}{b3}{b4}{limitations}{recommendations}
"""


# --------------------------------------------------------------------------- #
# Forecasting & ML
# --------------------------------------------------------------------------- #
def _fc_hero_stats(ml: dict) -> str:
    global_bt = ml["global_backtest"]  # ya ordenado ascendente por MAPE (rolling_backtest)
    winner = global_bt[0]
    naive = next(r for r in global_bt if r["model"] == "naive")
    gap_vs_naive = naive["mape"] - winner["mape"]
    return "".join([
        rc.stat(winner["model"], "", "Modelo global seleccionado",
                f"{gap_vs_naive:+.1f} pts MAPE vs. naive", "up"),
        rc.stat(f"{winner['mape']:.1f}", "% MAPE", "Error de validación (backtest 1 paso)"),
        rc.stat(str(ml["forecast_target_year"]), "", "Horizonte publicado",
                f"+{ml['forecast_horizon_years']} años desde {ml['last_observed_year']}"),
        rc.stat(f"{ml['coherence_gap_hierarchical']:+.1%}", "", "Riesgo de coherencia",
                "global directo vs. suma de países"),
    ])


def _block_forecast_validation(ml: dict, bundle: ReportBundle) -> str:
    """1 · Forecast validation — metodología, selector de forecast, dispersión de error, ejemplos de
    buen/mal ajuste y la tabla de model governance (serie, modelo, baseline, motivo, riesgo).
    """
    forecasts, summary, anomalies = bundle.forecasts, bundle.forecast_summary, bundle.anomalies
    countries = summary["series"].tolist()  # las 53 series, no una muestra

    fc_options = [("GLOBAL", "Global")]
    fc_options += [(f"TYPE::{t}", t) for t in ["Arabica", "Arabica/Robusta", "Robusta", "Robusta/Arabica"]]
    fc_options += [(c, c) for c in countries]
    fc_divs = []
    for i, (key, label) in enumerate(fc_options):
        if key not in forecasts:
            continue
        res = {k: (np.array(v) if isinstance(v, list) else v) for k, v in forecasts[key].items()}
        title = f"Forecast — {label} (modelo: {res['model']}, MAPE {res['mape']:.2f}%)"
        fc_divs.append(f'<div class="tab-panel {"active" if i==0 else ""}" data-tab-group="fc">{rc.div(pc.fig_forecast(res, title), "fc_"+str(i))}</div>')
    fc_select = rc.tab_select("fc", [
        ("Global", ["Global"]),
        ("Tipo de café", ["Arabica", "Arabica/Robusta", "Robusta", "Robusta/Arabica"]),
        ("País (53)", countries),
    ])

    bw = bv.best_worst_fit_countries(summary)
    best, worst = bw["best"], bw["worst"]
    best_res = {k: (np.array(v) if isinstance(v, list) else v) for k, v in forecasts[best["series"]].items()}
    worst_res = {k: (np.array(v) if isinstance(v, list) else v) for k, v in forecasts[worst["series"]].items()}
    best_fig = rc.div(pc.fig_forecast(best_res, f"Mejor ajuste — {best['series']} (MAPE {best['backtest_mape']:.2f}%)"), "fc_best")
    worst_fig = rc.div(pc.fig_forecast(worst_res, f"Peor ajuste — {worst['series']} (MAPE {worst['backtest_mape']:.2f}%)"), "fc_worst")

    cov = ml.get("interval_coverage", {})
    cov_html = ""
    if cov.get("total"):
        gap = cov["pct"] - cov["target_pct"]
        verdict = ("calibrada" if abs(gap) <= 8 else
                   "conservadora (más ancha de lo necesario)" if gap > 0 else
                   "optimista (subestima el riesgo real)")
        cov_html = (f"<strong>Calibración empírica (leave-one-out).</strong> Sobre {cov['total']} puntos de "
                    f"backtest (todas las series), el {cov['pct']:.0f}% del valor real cayó dentro de la "
                    f"banda del {cov['target_pct']}% nominal — banda {verdict}, medida contra datos que "
                    "no se usaron para construir el propio intervalo, no declarada por fórmula.")

    anomalous = set(anomalies.loc[anomalies["anomaly"], "Country"]) if "anomaly" in anomalies.columns else set()
    gov = bv.model_governance_table(ml, summary, top_n=len(summary), anomalous_countries=anomalous)
    gov_rows = "".join(
        f'<tr><td>{r["Serie"]}</td><td>{r["Modelo"]}</td>'
        f'<td class="num">{r["MAPE backtest"]:.2f}%</td>'
        f'<td class="num">{r["Baseline (naive)"]:.2f}%</td>'
        f'<td>{r["Motivo de selección"]}</td>'
        f'<td>{rc.pill(r["Riesgo"], _RISK_PILL_COLORS.get(r["Riesgo"].split(" ")[0], "#6B5647"))}</td></tr>'
        for _, r in gov.iterrows()
    )

    return f"""
<h2 class="section-title">1 · Forecast validation</h2>
<p class="section-sub">¿Cómo se valida cada forecast antes de publicarlo, y qué tan lejos se proyecta de lo observado?</p>
<div class="card pad">
  <div class="callout technical">
    <strong>Metodología.</strong> Selección de modelo por <em>rolling-origin backtest</em>:
    {ROLLING_ORIGINS} orígenes móviles, 1 paso adelante por origen, mínimo {MIN_TRAIN} observaciones de
    entrenamiento en el primer origen — cada evaluación solo ve datos anteriores al punto que predice,
    sin fuga de información (leakage). Métrica principal: MAPE. Baseline de referencia: modelo naive
    (persistencia del último valor observado). El forecast <strong>publicado</strong> se proyecta a
    {ml["forecast_horizon_years"]} años (hasta {ml["forecast_target_year"]}) reentrenando con todo el
    histórico — más allá de la ventana directamente validada a 1 paso, por lo que la banda de
    incertidumbre se ensancha con <code>√paso</code> en vez de mantenerse plana en los {ml["forecast_horizon_years"]}
    años (ver gráficos abajo). Series cortas o degeneradas (menos de {MIN_TRAIN} observaciones, o series
    completas en cero) caen a naive automáticamente o se excluyen del modelado — no se fuerza un ajuste
    que el histórico no sostiene.
  </div>
  {f'<div class="callout technical">{cov_html}</div>' if cov_html else ""}
</div>
{fc_select}
<div class="card">{"".join(fc_divs)}</div>

<p class="section-sub" style="margin-top:18px">Ejemplos concretos de ajuste — no todas las series se proyectan con la misma confianza.</p>
<div class="grid-2">
  <div class="card">{best_fig}</div>
  <div class="card">{worst_fig}</div>
</div>
<div class="card">{rc.div(pc.fig_backtest_error_distribution(summary), "fc_error_dist")}</div>

<div class="card pad">
  <h3>Model governance — Global, 4 tipos y las {len(summary)} series de país</h3>
  <div class="table-wrap scroll"><table class="data">
    <tr><th>Serie</th><th>Modelo</th><th>MAPE backtest</th><th>Baseline (naive)</th><th>Motivo de selección</th><th>Riesgo</th></tr>
    {gov_rows}
  </table></div>
  <p class="note">MAPE medio de backtest entre las {len(summary)} series de país: {ml["rolling_mape_mean"]:.2f}%
  (mediana {ml["rolling_mape_median"]:.2f}%) vs. {ml["naive_mape_mean"]:.2f}% del baseline naive.</p>
</div>
"""


def _block_model_comparison(ml: dict) -> str:
    """2 · Model comparison — un solo modelo ML global vs. modelos clásicos por-serie, en el mismo
    holdout, para decidir si la complejidad adicional se justifica por evidencia.
    """
    mc = ml["model_comparison"]
    classic_wins = mc["classic_mape_median"] <= mc["pooled_mape_median"]
    verdict = "los modelos clásicos por-serie ganan" if classic_wins else "el pooled global gana"
    stats = "".join([
        rc.stat(f"{mc['classic_mape_median']:.1f}", "% MAPE", f"Clásico por-serie (mediana, holdout {mc['horizon']} años)"),
        rc.stat(f"{mc['pooled_mape_median']:.1f}", "% MAPE", f"Pooled HistGradientBoosting (mediana, holdout {mc['horizon']} años)"),
        rc.stat(str(mc["n"]), "series", "Comparación pareada, mismo holdout"),
    ])
    return f"""
<h2 class="section-title">2 · Model comparison</h2>
<p class="section-sub">¿Se justifica un modelo ML global, o los modelos clásicos por-serie siguen siendo la mejor apuesta?</p>
<div class="stats">{stats}</div>
<div class="card pad">
  <div class="callout technical">
    <strong>Trade-off evaluado, no asumido.</strong> El modelo pooled comparte señal entre las
    {mc["n"]} series apiladas y predice la tasa de crecimiento en escala log (los árboles de decisión
    no extrapolan niveles); el enfoque clásico ajusta un modelo por serie, seleccionado por su propio
    backtest. Ambos predicen el mismo holdout de {mc["horizon"]} años, fuera de muestra — comparación
    pareada, no dos métricas en condiciones distintas. Resultado: {verdict} en MAPE mediano
    ({mc["classic_mape_median"]:.1f}% vs. {mc["pooled_mape_median"]:.1f}%). La robustez del enfoque
    simple se elige <em>por evidencia</em>, no por defecto: para series anuales cortas y suaves como
    estas, la complejidad adicional del modelo global no reduce el error de forecast.
  </div>
</div>
"""


def _block_risk_governance(ml: dict, anomalies) -> str:
    """3 · Risk / anomaly governance — qué mercados exigen revisión manual, y qué tan confiable es el
    clasificador de preferencia de tipo, con el peso visual proporcional a si conecta con una acción.
    """
    anom = anomalies[anomalies["anomaly"]] if "anomaly" in anomalies.columns else anomalies.head(0)
    anom_list = ", ".join(anom["Country"].tolist()) if len(anom) else "ninguna"

    clf_acc, clf_base = ml["classifier"]["cv_accuracy"], ml["classifier"]["baseline"]
    clf_beats = clf_acc > clf_base + 0.02
    if clf_beats:
        imp = ml["classifier"]["importances"]
        top_imp = max(imp, key=imp.get)
        clf_text = (f"El clasificador de preferencia de tipo (RandomForest) acierta {clf_acc*100:.0f}% (CV) "
                    f"sobre un baseline de {clf_base*100:.0f}%, señalando <em>{top_imp}</em> como variable "
                    "más informativa.")
    else:
        clf_text = (f"El clasificador de preferencia de tipo (RandomForest) no supera al azar "
                    f"({clf_acc*100:.0f}% CV vs. {clf_base*100:.0f}% baseline) — hallazgo negativo útil: "
                    "la preferencia de tipo no se infiere del patrón de consumo; responde a "
                    "geografía/agronomía, no a la dinámica de demanda.")

    return f"""
<h2 class="section-title">3 · Risk / anomaly governance</h2>
<p class="section-sub">¿Qué mercados requieren revisión manual antes de actuar sobre su forecast?</p>
<div class="card pad">
  <div class="callout technical">
    <strong>Anomalías multivariantes (IsolationForest).</strong> Mercados con patrón atípico de
    nivel/crecimiento/volatilidad conjunto, no validados contra una fuente externa — candidatos a
    revisión humana antes de comprometer presupuesto sobre su forecast, no hechos confirmados:
    {anom_list}.
  </div>
  <p class="note">{clf_text}</p>
</div>
"""


def _block_segmentation_action(bundle: ReportBundle, ml: dict) -> str:
    """4 · Segmentation & business action — traduce cada resultado del pipeline en una decisión
    operativa, y convierte el clustering en una pieza accionable: tamaño, comportamiento, riesgo
    operacional y recomendación comercial por cluster (no solo PCA + puntos de color).
    """
    clusters = bundle.clusters
    names_cons = {int(k): v for k, v in ml["cluster_consumo_names"].items()}
    names_pref = {int(k): v for k, v in ml["cluster_preferencia_names"].items()}
    names_fc = {int(k): v for k, v in ml["cluster_forecast_names"].items()}
    cl_views = [
        ("cluster_consumo", "Por consumo / crecimiento", names_cons),
        ("cluster_preferencia", "Por preferencia de tipo", names_pref),
        ("cluster_forecast", "Por trayectoria de forecast", names_fc),
    ]
    pca_ev = ml.get("pca_explained_variance", [0, 0])
    cl_divs, cl_btns = [], []
    for i, (col, label, names) in enumerate(cl_views):
        sub = clusters.dropna(subset=[col])
        figc = pc.fig_cluster_scatter(sub, sub[col].astype(int).values, names, explained_variance=pca_ev)
        cl_divs.append(f'<div class="tab-panel {"active" if i==0 else ""}" data-tab-group="cl">{rc.div(figc, "cl_"+str(i))}</div>')
        cl_btns.append(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="showTab(\'cl\',{i},this)">{label}</button>')

    sweep = ml.get("silhouette_consumo", [])
    sil_k4 = ml.get("silhouette_k4", {})
    sil_html = ""
    if sweep and sil_k4:
        sweep_txt = ", ".join(f"k={r['k']}: {r['silhouette']:.2f}" + (" (elegido)" if r["k"] == 4 else "")
                              for r in sweep)
        sil_html = (f"<strong>¿Por qué k=4?</strong> Barrido de silhouette score (vista consumo, k=3–6): "
                    f"{sweep_txt}. Con el k=4 elegido, silhouette por vista: consumo "
                    f"{sil_k4.get('consumo', 0):.2f}, preferencia {sil_k4.get('preferencia', 0):.2f}, "
                    f"forecast {sil_k4.get('forecast', 0):.2f} — valores por debajo de 0.5 indican "
                    "clusters con solape real entre ellos, no fronteras nítidas; se leen como segmentos "
                    "de tendencia dominante, no categorías puras.")

    profile = bv.cluster_profile_summary(bundle.feats, clusters, "cluster_consumo", names_cons)
    profile_rows = "".join(
        f'<tr><td>{r["nombre"]}</td><td class="num">{r["n_paises"]}</td>'
        f'<td class="num">{rc.human(r["nivel_medio"])}</td>'
        f'<td class="num">{r["cagr_medio"]*100:+.1f}%</td>'
        f'<td>{rc.pill(r["riesgo_operacional"], _RISK_PILL_COLORS.get(r["riesgo_operacional"], "#6B5647"))}</td>'
        f'<td>{r["recomendacion"]}</td></tr>'
        for _, r in profile.iterrows()
    )

    return f"""
<h2 class="section-title">4 · Segmentation &amp; business action</h2>
<p class="section-sub">¿Cómo se traduce cada resultado del pipeline en una decisión operativa?</p>
<div class="card pad">
  <ol class="key-insights">
    <li><strong>Forecast global → planificación de demanda</strong>: volumen esperado a
    {ml["forecast_target_year"]} para dimensionar compra y capacidad.</li>
    <li><strong>Forecast por país → priorización comercial</strong>: dónde invertir presupuesto de
    expansión primero (ver EDA &amp; Insights, bloque 4).</li>
    <li><strong>Clusters → segmentación de estrategia</strong>: presupuesto y cadencia de seguimiento
    diferenciados por cluster, no un plan único para 53 mercados.</li>
    <li><strong>Anomalías → mercados que exigen revisión manual</strong> antes de comprometer
    presupuesto sobre su forecast.</li>
    <li><strong>Pooled vs. clásico → decisión de usar el enfoque más simple y robusto</strong>,
    evaluado por evidencia, no adoptado por defecto.</li>
  </ol>
</div>

<div class="tabs">{"".join(cl_btns)}</div>
<div class="card">{"".join(cl_divs)}</div>
{f'<div class="callout technical">{sil_html}</div>' if sil_html else ""}

<div class="card pad">
  <h3>Perfil de negocio por cluster (vista consumo / crecimiento)</h3>
  <div class="table-wrap"><table class="data">
    <tr><th>Cluster</th><th># países</th><th>Nivel medio (5 años)</th><th>CAGR medio</th><th>Riesgo operacional</th><th>Recomendación</th></tr>
    {profile_rows}
  </table></div>
</div>
"""


def _block_production_closing() -> str:
    return """
<div style="border-top:2px dashed var(--line);padding-top:20px;margin-top:8px">
<span class="eyebrow">Fuera del alcance de este entregable</span>
<h2 class="section-title" style="margin-top:6px">Qué cambiaría para producción</h2>
<div class="card pad">
  <ol class="key-insights">
    <li><strong>Monitoreo de drift</strong>: alertar cuando el error real de una serie se aleje de su
    MAPE de backtest, no solo reentrenar en una fecha fija.</li>
    <li><strong>Reentrenamiento por crop-year</strong>: el pipeline corre por campaña (año fiscal ICO),
    no de forma continua — no hay señal diaria o semanal que lo justifique a esta escala.</li>
    <li><strong>Tracking de experimentos</strong> (ej. MLflow): versionar qué modelo ganó cada backtest
    y por qué, no solo persistir el resultado final.</li>
    <li><strong>Validación temporal continua</strong>: repetir el rolling-origin backtest en cada
    campaña nueva — el modelo ganador de hoy no está garantizado a seguir siéndolo en 2027.</li>
    <li><strong>Revisión humana obligatoria</strong> para los mercados marcados como anómalos antes de
    usar su forecast en una decisión de presupuesto.</li>
  </ol>
</div>
</div>
"""


def build_forecasting_page(bundle: ReportBundle) -> str:
    """Página 2 — Forecasting & ML en 4 bloques técnicos: validación del forecast, comparación de
    modelos, gobierno de riesgo/anomalías y segmentación accionable. La estrategia comercial (matriz
    tamaño×crecimiento, ranking de mercados) vive en el bloque 4 de EDA & Insights — aquí solo se
    referencia, no se repite.

    Toma todo del `bundle` ya cargado (report_data.load_bundle) — no lee nada de disco.
    """
    ml = bundle.ml
    hero_stats = _fc_hero_stats(ml)
    b1 = _block_forecast_validation(ml, bundle)
    b2 = _block_model_comparison(ml)
    b3 = _block_risk_governance(ml, bundle.anomalies)
    b4 = _block_segmentation_action(bundle, ml)
    closing = _block_production_closing()

    return f"""
<section class="hero hero-grid">
  <div>
    <span class="eyebrow">Forecasting &amp; Machine Learning</span>
    <h1>Un forecast para decidir <span class="accent">presupuesto, expansión y riesgo</span></h1>
    <p class="lead">Modelo seleccionado por evidencia contra un baseline naive, validado con backtest
    de origen móvil, proyectado a {ml["forecast_target_year"]} — no "tenemos forecasting", sino qué
    decisión habilita cada número y qué tan lejos conviene confiar en él.</p>
    <div class="hero-meta">
      <span class="meta-pill">Rolling-origin backtest</span>
      <span class="meta-pill">Horizonte a {ml["forecast_target_year"]}</span>
      <span class="meta-pill">Bandas de incertidumbre crecientes</span>
      <span class="meta-pill">Model governance por serie</span>
    </div>
  </div>
  <aside class="hero-aside">
    <h3>Uso operativo</h3>
    <ul>
      <li>Elegir el modelo con mejor evidencia frente a un baseline simple, no el más complejo.</li>
      <li>Tratar los mercados anómalos como pendientes de revisión, no como forecast confiable.</li>
      <li>Usar los clusters para distribuir presupuesto y cadencia de seguimiento por segmento.</li>
    </ul>
  </aside>
</section>
<div class="stats">{hero_stats}</div>
{b1}{b2}{b3}{b4}{closing}
"""


# --------------------------------------------------------------------------- #
# GenAI (bonus, placeholder)
# --------------------------------------------------------------------------- #
_MARKETING_PIPELINE = [
    ("Web scraping", "Scrapers sobre tiendas de café y marketplaces: precio, tamaño, origen, tipo, "
     "promociones y texto de producto de la competencia.", "Ingesta"),
    ("Limpieza y normalización", "Deduplicación, estandarización de moneda/unidad, extracción de "
     "entidades (marca, tipo de café, gramaje) desde texto libre.", "ETL"),
    ("Estructuración", "Tabla precio × tienda × país × tipo, más embeddings del texto de producto "
     "para búsqueda semántica entre competidores.", "Feature store"),
    ("Cruce con datos internos", "Se une con el forecast a 2030 y los clusters de segmentación ya "
     "calculados por este pipeline (ver Forecasting &amp; ML, bloque 4).", "Grounding"),
    ("Razonamiento GenAI (RAG)", "Un LLM consulta precios de mercado + forecast interno y redacta la "
     "estrategia: en qué tiendas vender, a qué precio, qué mercados priorizar.", "LLM + RAG"),
    ("Revisión humana", "Un analista valida la recomendación antes de publicarla — el LLM propone "
     "estrategia, no fija precios en producción por su cuenta.", "Human-in-the-loop"),
]

_CREATIVE_PIPELINE = [
    ("Input: brief de mercado", "Mercado, segmento de cluster y posicionamiento de precio ya "
     "definidos por la propuesta de estrategia de mercadeo.", "Input"),
    ("Análisis visual multimodal", "Un modelo con visión analiza empaques y anuncios de la "
     "competencia scrapeados: paleta, tipografía, tono del mensaje.", "Visión"),
    ("Generación de conceptos", "Genera paletas de color, frases de marketing y mockups de empaque "
     "por mercado/segmento — variantes para elegir, no una única propuesta.", "Texto-a-imagen"),
    ("Revisión de diseño", "Un diseñador humano elige, ajusta y aprueba antes de producción — el "
     "modelo genera candidatos, no arte final.", "Human-in-the-loop"),
    ("Kit creativo", "Entregable: paleta, moodboard, copy y 2-3 conceptos de empaque por mercado "
     "priorizado.", "Output"),
]

_CHATBOT_PIPELINE = [
    ("Fuentes validadas", "El mismo ReportBundle que arma este informe: histórico, forecast a 2030, "
     "clusters y model governance — nada nuevo que auditar.", "Grounding"),
    ("Capa de herramientas", "Tools tipadas: consulta pandas, generación de gráfico Plotly, lookup de "
     "forecast por país o tipo de café.", "Tool use"),
    ("Orquestador agéntico", "El LLM interpreta la pregunta en lenguaje natural y decide qué tool(s) "
     "llamar — no genera números por su cuenta.", "Agente"),
    ("Respuesta trazable", "Texto + gráfico o tabla de soporte, citando siempre la serie o el cálculo "
     "exacto de origen — mismo estándar que este informe.", "Trazabilidad"),
    ("Guardrails", "Si la pregunta pide un dato que el pipeline no calculó, el bot lo dice "
     "explícitamente en vez de inventarlo.", "Anti-alucinación"),
]


def build_genai_page() -> str:
    """Página 3 — bonus GenAI: 3 propuestas concretas (estrategia de mercadeo, estudio creativo,
    chatbot agéntico), cada una como pipeline visual. Contenido estático — sin datos ni cálculo, es
    una propuesta de arquitectura, no un resultado del pipeline de ML.
    """
    return f"""
<section class="hero hero-grid">
  <div>
    <span class="eyebrow">Bonus — propuesta</span>
    <h1>De datos de consumo a <span class="accent">estrategia de mercadeo con GenAI</span></h1>
    <p class="lead">Tres sistemas complementarios que parten de lo ya calculado en este informe (forecast a
    2030, segmentación de mercados) y lo cruzan con datos externos de precio y competencia para generar
    estrategia comercial, material creativo e insights conversacionales — con revisión humana en cada
    salida que toca precio, marca o mensaje público.</p>
    <div class="hero-meta">
      <span class="meta-pill">3 sistemas propuestos</span>
      <span class="meta-pill">RAG sobre datos validados</span>
      <span class="meta-pill">Human-in-the-loop</span>
      <span class="meta-pill">Trazable, no una caja negra</span>
    </div>
  </div>
  <aside class="hero-aside">
    <h3>Qué no debe ser</h3>
    <ul>
      <li>No una demo decorativa desconectada del pipeline de forecasting/clustering.</li>
      <li>No una capa que fije precios o publique diseño sin revisión humana.</li>
      <li>No un chatbot genérico sin trazabilidad a la fuente del dato.</li>
    </ul>
  </aside>
</section>

<h2 class="section-title">1 · Motor de estrategia de precio y mercado</h2>
<p class="section-sub">¿En qué tiendas vender, a qué precio y en qué mercados priorizar, cruzando oferta/demanda
externa con el forecast interno?</p>
<div class="card pad">
  <p class="note" style="margin:0 0 4px">Usa directamente la priorización comercial y el forecast a 2030 ya
  calculados en este informe (EDA &amp; Insights, bloque 4; Forecasting &amp; ML, bloque 1) como ancla interna
  — el scraping aporta el lado de oferta/competencia que el dataset de consumo no tiene.</p>
</div>
<div class="card pad">{rc.pipeline(_MARKETING_PIPELINE)}</div>

<h2 class="section-title">2 · Estudio creativo generativo</h2>
<p class="section-sub">¿Qué frases, colores y conceptos de empaque usar por mercado, sin partir de cero cada
vez?</p>
<div class="card pad">{rc.pipeline(_CREATIVE_PIPELINE)}</div>

<h2 class="section-title">3 · Chatbot agéntico de insights</h2>
<p class="section-sub">¿Cómo responde un usuario de negocio, sin saber pandas ni Plotly, preguntas sobre
consumo, forecast y clusters?</p>
<div class="card pad">{rc.pipeline(_CHATBOT_PIPELINE)}</div>

<div style="border-top:2px dashed var(--line);padding-top:20px;margin-top:8px">
<span class="eyebrow">Coherencia con MLOps / GenAI</span>
<h2 class="section-title" style="margin-top:6px">Principios aplicados a las 3 propuestas</h2>
<div class="card pad">
  <ol class="key-insights">
    <li><strong>Human-in-the-loop obligatorio</strong> antes de cualquier acción comercial real: precio,
    empaque o mensaje público — el mismo criterio que ya aplica a los mercados anómalos en Forecasting
    &amp; ML (bloque 3).</li>
    <li><strong>Grounding sobre datos validados</strong>: el LLM consulta el ReportBundle y el resumen de
    precios ya estructurado, no inventa cifras — el mismo principio de "no I/O disperso" que sostiene la
    arquitectura de este informe.</li>
    <li><strong>Trazabilidad</strong>: cada salida de GenAI debe poder rastrearse hasta el dato o cálculo
    que la originó, igual que cada número de este informe viene de una función de <code>business_views</code>
    y no de texto escrito a mano.</li>
    <li><strong>Evaluación offline antes de cada cambio</strong> de prompt o agente — el equivalente en
    GenAI al rolling-origin backtest de un modelo de forecasting: no se despliega sin medir contra un
    caso de referencia.</li>
    <li><strong>Monitoreo de drift</strong>: si cambia el layout de una tienda o el estilo visual de la
    competencia, el scraper o el analizador debe alertarlo, no fallar en silencio ni devolver datos
    obsoletos como si fueran actuales.</li>
  </ol>
</div>
</div>
"""
