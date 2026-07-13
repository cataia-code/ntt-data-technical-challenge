# ruff: noqa: E402

"""Genera el informe web interactivo (reports/web/) a partir del bundle de reports/data/.

Ejecutar DESPUÉS de scripts/run_pipeline.py. Exporta cada figura Plotly como div embebible y las
inyecta en 3 páginas con estilo café: EDA (narrativa accionable), Forecasting & ML, y GenAI
(placeholder).

Uso: py -3 scripts/build_report.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly

from data_prep import build_long_dataset
from features import build_country_features
import business_views as bv
import plotly_charts as pc

DATA_DIR = ROOT / "reports" / "data"
WEB_DIR = ROOT / "reports" / "web"
ASSETS_DIR = WEB_DIR / "assets"

PAGES = [("index.html", "EDA"), ("forecasting.html", "Forecasting & ML"), ("genai.html", "GenAI")]


def div(fig, name):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=name,
                       config={"displayModeBar": False, "responsive": True})


def nav(active):
    links = "".join(
        f'<a href="{href}" class="{"active" if label == active else ""}">{label}</a>'
        for href, label in PAGES
    )
    return f"""<nav class="nav">
  <div class="brand"><span class="bean">☕</span> High Garden Coffee</div>
  <div class="nav-links">{links}</div>
</nav>"""


def shell(title, active, body):
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · High Garden Coffee</title>
<link rel="stylesheet" href="assets/styles.css">
<script src="assets/plotly.min.js"></script>
</head>
<body>
{nav(active)}
<div class="wrap">
{body}
</div>
<footer>High Garden Coffee · Informe analítico generado desde el pipeline ML · Datos ICO 1990–2020
(consumo doméstico, en tazas)</footer>
<script>
function showTab(group, idx, btn) {{
  document.querySelectorAll('[data-tab-group="'+group+'"]').forEach((p,i)=>p.classList.toggle('active', i===idx));
  btn.parentNode.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active', i===idx));
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""


def stat(value, unit, label, delta=None, dclass=""):
    d = f'<div class="delta {dclass}">{delta}</div>' if delta else ""
    u = f'<span class="unit">{unit}</span>' if unit else ""
    return f'<div class="stat"><div class="value">{value} {u}</div><div class="label">{label}</div>{d}</div>'


def human(v):
    v = float(v)
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.0f} M"
    return f"{v:,.0f}"


def insight_box(insight, implicacion, accion):
    rows = [("insight", "Insight", insight), ("implicacion", "Implicación", implicacion),
            ("accion", "Acción sugerida", accion)]
    body = "".join(
        f'<div class="insight-row"><span class="tag {cls}">{label}</span><span class="txt">{txt}</span></div>'
        for cls, label, txt in rows
    )
    return f'<div class="insight-box">{body}</div>'


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'


# --------------------------------------------------------------------------- #
def build():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", ASSETS_DIR / "plotly.min.js")

    long_df = build_long_dataset()
    feats = build_country_features(long_df)
    ml = json.loads((DATA_DIR / "ml_summary.json").read_text(encoding="utf-8"))
    forecasts = json.loads((DATA_DIR / "forecasts.json").read_text(encoding="utf-8"))
    bubble = pd.read_parquet(DATA_DIR / "bubble.parquet")
    clusters = pd.read_parquet(DATA_DIR / "clusters.parquet")
    summary = pd.read_csv(ROOT / "reports" / "forecast_summary.csv")
    priority = pd.read_csv(ROOT / "reports" / "priority_markets.csv")
    anomalies = pd.read_csv(ROOT / "reports" / "anomalies.csv")

    valid = long_df[long_df["is_valid_series"]]
    global_hist = valid.groupby("fiscal_year_start")["consumption"].sum()
    last_year_val = global_hist.iloc[-1]
    prev_year_val = global_hist.iloc[-2]
    cagr_global = (global_hist.iloc[-1] / global_hist.iloc[0]) ** (1 / (global_hist.index[-1] - global_hist.index[0])) - 1
    # --- Vistas de negocio (EDA narrativo) --- #
    stats = bv.descriptive_stats(long_df)
    pareto = bv.pareto_table(long_df)
    quad = bv.classify_quadrant(bubble)
    wf = bv.waterfall_contribution(long_df)
    type_tbl = bv.type_summary_table(long_df)
    vol_rank = bv.volatility_ranking(feats, top_n=8)
    priority_tbl = bv.priority_ranking_table(quad, feats, top_n=15)

    drop_pct = (last_year_val - prev_year_val) / prev_year_val * 100
    n_priorizar = int((quad["quadrant"] == "Priorizar").sum())

    # ============================ PÁGINA 1 — EDA ============================ #
    eda_stats = "".join([
        stat(human(last_year_val), "tazas", "Consumo global 2019/20"),
        stat(ml["n_countries"], "", "Países con datos"),
        stat(f"+{cagr_global*100:.1f}", "%/año", "CAGR global 1990–2020", "Crecimiento compuesto", "up"),
        stat(f"{drop_pct:.1f}", "%", "Cambio último año", "vs. 2018/19", "down"),
    ])

    # --- Sección 1: qué datos tenemos --- #
    s1 = f"""
<h2 class="section-title">Alcance de los datos</h2>
<div class="scope-grid">
  <div class="scope-card can">
    <h4>✅ Podemos responder</h4>
    <ul>
      <li>Tendencias de consumo doméstico (en tazas) 1990–2020, por país, tipo de café y continente.</li>
      <li>55 países, 30 campañas cada uno; {ml['n_countries']} con serie utilizable para modelado.</li>
      <li>Proyecciones de consumo a 5 años con intervalos (ver página Forecasting &amp; ML).</li>
      <li>Segmentación de mercados por tamaño, crecimiento y tipo de café preferido.</li>
    </ul>
  </div>
  <div class="scope-card cannot">
    <h4>❌ NO podemos responder</h4>
    <ul>
      <li><strong>Precios</strong>: el dataset no contiene ninguna columna de precio, pese a que el
      enunciado original pide "rangos de precios futuros". Se necesitaría una fuente externa
      (ej. ICO Composite Price) para abordarlo.</li>
      <li>2 países (Equatorial Guinea, Nepal) tienen serie completa en cero — sin señal utilizable.</li>
      <li>Causalidad: el dataset no explica <em>por qué</em> cambia el consumo, solo <em>cuánto</em>.</li>
    </ul>
  </div>
</div>
"""

    # --- Sección 2: evolución del consumo --- #
    s2 = f"""
<h2 class="section-title">Evolución del consumo</h2>
<div class="card">{div(pc.fig_evolution_dual_selector(long_df), "eda_evo")}</div>
{insight_box(
    f"El consumo global casi se triplicó entre 1990/91 y 2018/19 (de {human(global_hist.iloc[0])} a "
    f"{human(global_hist.max())} tazas, CAGR {cagr_global*100:.1f}%/año), pero cayó "
    f"{abs(drop_pct):.1f}% en 2019/20 ({human(prev_year_val)} → {human(last_year_val)} tazas) — la "
    f"primera caída interanual relevante de toda la serie.",
    "Una sola campaña no define una tendencia; no es posible saber todavía si es un evento puntual o "
    "el inicio de una desaceleración estructural del consumo doméstico mundial.",
    "Monitorear la próxima campaña reportada por el ICO (2020/21) antes de ajustar proyecciones de "
    "largo plazo, presupuesto de exportación o niveles de inventario.",
)}
"""

    # --- Sección 3: concentración --- #
    s3 = f"""
<h2 class="section-title">Concentración de mercado</h2>
<div class="card">{div(pc.fig_pareto(pareto), "eda_pareto")}</div>
<div class="grid-2">
  <div class="card">{div(pc.fig_continent_distribution(long_df), "eda_cont")}</div>
  <div class="card">{div(pc.fig_type_distribution(long_df), "eda_typedist")}</div>
</div>
<div class="card">{div(pc.fig_distribution_hist(long_df, stats), "eda_hist")}</div>
<div class="card">{div(pc.fig_geo_bubble(bubble), "eda_geo")}</div>
{insight_box(
    f"Brasil concentra {pareto.attrs['brazil_share']:.0f}% del consumo doméstico acumulado; los 5 "
    f"países más grandes explican {pareto.attrs['top5_share']:.0f}% y los 10 más grandes, "
    f"{pareto.attrs['top10_share']:.0f}%. La media ({human(stats['mean'])}) es más de 12 veces la "
    "mediana (" + human(stats["median"]) + "), evidencia directa de la asimetría.",
    "El mercado doméstico de café está altamente concentrado: la exposición comercial de High Garden "
    "a este segmento depende desproporcionadamente de un puñado de países, con Brasil como caso "
    "extremo aislado del resto de la distribución.",
    "Diversificar la base de mercados prioritarios más allá de Brasil (ver matriz de la sección 4), "
    "sin dejar de atenderlo como mercado ancla de gran escala.",
)}
"""

    # --- Sección 4: matriz tamaño x crecimiento (chart principal) --- #
    s4 = f"""
<h2 class="section-title">Matriz tamaño × crecimiento</h2>
<div class="card">{div(pc.fig_quadrant_matrix(quad), "eda_quad")}</div>
{insight_box(
    f"{n_priorizar} países caen en 'Priorizar' (tamaño y crecimiento reciente por encima de la "
    "mediana) — desde mercados medianos en fuerte aceleración como Vietnam (8.2%/año) o Tailandia "
    "(7.2%/año), hasta Brasil, cuyo crecimiento es modesto (1.8%/año) pero sigue superando la mediana "
    "de una muestra donde la mitad de los países crece prácticamente cero. El tamaño de burbuja "
    "distingue ambos perfiles dentro del mismo cuadrante.",
    "No todo mercado grande es una oportunidad de crecimiento, y no todo mercado en crecimiento "
    "justifica inversión inmediata si es demasiado pequeño; el cuadrante por sí solo no basta — hay "
    "que leerlo junto con el tamaño real de la burbuja.",
    "Asignar presupuesto comercial diferenciado por cuadrante: expansión activa en los mercados "
    "medianos de alto crecimiento de 'Priorizar' (Vietnam, Tailandia, Filipinas, Colombia), "
    "retención/eficiencia en Brasil, y pilotos de bajo costo en 'Explorar'.",
)}
"""

    # --- Sección 5: qué explica el cambio reciente (waterfall) --- #
    movers_up = [c for c, v in wf["items"] if v > 0 and "Otros" not in c]
    movers_down = [c for c, v in wf["items"] if v < 0 and "Otros" not in c]
    s5 = f"""
<h2 class="section-title">Contribución al cambio reciente</h2>
<div class="card">{div(pc.fig_waterfall(wf), "eda_waterfall")}</div>
{insight_box(
    f"La caída de {human(abs(wf['total_delta']))} tazas no viene de un colapso generalizado: "
    f"{', '.join(movers_down)} tiraron hacia abajo, pero {', '.join(movers_up)} en realidad "
    "crecieron ese año y compensaron parte de la baja. Venezuela fue el mayor lastre individual, "
    "coincidiendo con el contexto de crisis económica conocido en el país en ese periodo — algo que "
    "este dataset por sí solo no puede confirmar.",
    "La señal es más ambigua que 'el mercado se está contrayendo': es una mezcla de países en baja y "
    "en alza con un neto ligeramente negativo, no una tendencia uniforme.",
    "Investigar puntualmente Venezuela y Brasil (mayores caídas individuales) antes de generalizar la "
    "conclusión de la sección 2 al resto del portafolio de países.",
)}
"""

    # --- Sección 6: tipos de café --- #
    type_rows = "".join(
        f'<tr><td>{pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td>'
        f'<td class="num">{r["share_pct"]:.1f}%</td>'
        f'<td class="num">{r["cagr_recent"]*100:+.1f}%</td>'
        f'<td class="num">{r["n_countries"]}</td>'
        f'<td>{r["dominant_country"]}</td></tr>'
        for _, r in type_tbl.iterrows()
    )
    declining_type = type_tbl.loc[type_tbl["cagr_recent"].idxmin(), "Coffee type"]
    s6 = f"""
<h2 class="section-title">Tipos de café</h2>
<div class="card pad">
  <div class="table-wrap"><table class="data">
    <tr><th>Tipo de café</th><th>% del consumo</th><th>CAGR reciente</th><th># países</th><th>País dominante</th></tr>
    {type_rows}
  </table></div>
</div>
<div class="card">{div(pc.fig_type_trend_by_scope(long_df), "eda_typescope")}</div>
{insight_box(
    f"'{type_tbl.iloc[0]['Coffee type']}' concentra {type_tbl.iloc[0]['share_pct']:.0f}% del consumo "
    f"(liderado por {type_tbl.iloc[0]['dominant_country']}), pero '{type_tbl.iloc[1]['Coffee type']}' "
    f"crece más rápido ({type_tbl.iloc[1]['cagr_recent']*100:.1f}%/año). "
    f"'{declining_type}' es el único tipo con CAGR reciente negativo — el segmento de menor volumen "
    "y en declive.",
    "La oferta de exportación óptima no es uniforme: el tipo dominante en volumen no es "
    "necesariamente el de mayor momentum comercial.",
    f"Priorizar mezclas '{type_tbl.iloc[1]['Coffee type']}' en los mercados emergentes de la sección "
    f"4, y evaluar si vale la pena seguir invirtiendo en '{declining_type}' dado su volumen marginal "
    "y tendencia negativa.",
)}
"""

    # --- Sección 7: riesgos --- #
    vol_rows = "".join(
        f'<tr><td>{r["Country"]}</td>'
        f'<td>{pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td>'
        f'<td class="num">{r["volatility"]:.2f}</td>'
        f'<td class="num">{r["cagr_recent"]*100:+.1f}%</td></tr>'
        for _, r in vol_rank.iterrows()
    )
    s7 = f"""
<h2 class="section-title">Volatilidad y riesgo</h2>
<div class="card pad">
  <div class="table-wrap"><table class="data">
    <tr><th>País</th><th>Tipo</th><th>Volatilidad (CV)</th><th>CAGR reciente</th></tr>
    {vol_rows}
  </table></div>
</div>
{insight_box(
    f"{vol_rank.iloc[0]['Country']} tiene la mayor volatilidad interanual del dataset (coeficiente de "
    f"variación {vol_rank.iloc[0]['volatility']:.2f}); varios países de esta lista también aparecen "
    "entre los de mayor CAGR reciente en la sección 4 (ej. Tanzania) — el crecimiento reportado ahí "
    "puede estar inflado por ruido de un año puntual, no por una tendencia sostenida.",
    "Confiar en el CAGR reciente de un país altamente volátil sin mirar su volatilidad es un error "
    "común: un solo año atípico puede simular 'alto crecimiento' donde en realidad hay inestabilidad.",
    "Antes de asignar presupuesto comercial a un mercado del cuadrante 'Explorar' o 'Priorizar', "
    "cruzar su CAGR contra esta tabla de volatilidad y, si aparece aquí, validar con una fuente "
    "adicional antes de comprometer inversión.",
)}
"""

    # --- Sección 8: ranking priorizado (tabla final) --- #
    prio_rows = "".join(
        f'<tr><td>{r["Country"]}</td>'
        f'<td>{pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td>'
        f'<td class="num">{human(r["level_last5_mean"])}</td>'
        f'<td class="num">{r["cagr_recent"]*100:+.1f}%</td>'
        f'<td class="num">{r["volatility"]:.2f}</td>'
        f'<td>{pill(r["quadrant"], pc.QUADRANT_COLORS.get(r["quadrant"], "#6B5647"))}</td></tr>'
        for _, r in priority_tbl.iterrows()
    )
    s8 = f"""
<h2 class="section-title">Ranking de mercados prioritarios</h2>
<div class="card pad">
  <div class="table-wrap"><table class="data">
    <tr><th>País</th><th>Tipo</th><th>Consumo reciente</th><th>CAGR reciente</th><th>Volatilidad</th><th>Recomendación</th></tr>
    {prio_rows}
  </table></div>
</div>
{insight_box(
    f"El top del ranking combina mercados medianos de alto crecimiento y baja volatilidad "
    f"({priority_tbl.iloc[0]['Country']}, {priority_tbl.iloc[1]['Country']}) con el propio Brasil, "
    "que aparece por su tamaño más que por su momentum.",
    "Esta tabla es el punto de partida operativo: cruza tamaño, crecimiento y confiabilidad "
    "(volatilidad) en una sola vista, algo que ningún gráfico individual anterior muestra por sí solo.",
    "Usar esta tabla como base de la conversación comercial trimestral con el área de exportación; "
    "revisar el detalle de forecast por país en la página Forecasting &amp; ML antes de comprometer "
    "presupuesto.",
)}
"""

    # --- Sección secundaria: detalle técnico --- #
    s_secondary = f"""
<div class="secondary-section">
  <span class="secondary-label">Detalle técnico</span>
  <h2 class="section-title">Trayectorias normalizadas</h2>
  <div class="card">{div(pc.fig_heatmap(long_df, 20), "eda_heat")}</div>
</div>
"""

    eda_body = f"""
<section class="hero">
  <span class="eyebrow">Análisis exploratorio</span>
  <h1>Tres décadas de <span class="accent">consumo de café</span> en el mundo</h1>
</section>
<div class="stats">{eda_stats}</div>
{s1}{s2}{s3}{s4}{s5}{s6}{s7}{s8}{s_secondary}
"""

    # ==================== PÁGINA 2 — FORECASTING & ML ==================== #
    fc_options = [("GLOBAL", "Global")]
    fc_options += [(f"TYPE::{t}", f"Tipo: {t}") for t in ["Arabica", "Arabica/Robusta", "Robusta", "Robusta/Arabica"]]
    fc_options += [(c, c) for c in ml["top_chart_countries"][:8]]

    fc_divs, fc_btns = [], []
    for i, (key, label) in enumerate(fc_options):
        if key not in forecasts:
            continue
        res = forecasts[key]
        res = {k: (np.array(v) if isinstance(v, list) else v) for k, v in res.items()}
        title = f"Forecast — {label} (modelo: {res['model']}, MAPE {res['mape']:.2f}%)"
        fc_divs.append(f'<div class="tab-panel {"active" if i==0 else ""}" data-tab-group="fc">{div(pc.fig_forecast(res, title), "fc_"+str(i))}</div>')
        fc_btns.append(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="showTab(\'fc\',{i},this)">{label}</button>')

    mc = ml["model_comparison"]
    classic_wins = mc["classic_mape_median"] <= mc["pooled_mape_median"]
    fc_stats = "".join([
        stat(human(ml["global_forecast_2024_25"]), "tazas", "Forecast global 2024/25"),
        stat(f'{mc["classic_mape_median"]:.1f}', "% MAPE", "Clásico por-serie (mediana, 5 años)"),
        stat(f'{mc["pooled_mape_median"]:.1f}', "% MAPE", "ML global pooled GBM (5 años)"),
        stat(f'{ml["classifier"]["cv_accuracy"]*100:.0f}', "%", "Clasificador de tipo (CV acc.)",
             f'baseline {ml["classifier"]["baseline"]*100:.0f}%'),
    ])

    fc_rows = ""
    for _, r in summary.head(12).iterrows():
        fc_rows += (f'<tr><td>{r["series"]}</td><td>{r["best_model"]}</td>'
                    f'<td class="num">{r["backtest_mape"]:.2f}%</td>'
                    f'<td class="num">{human(r["last_observed"])}</td>'
                    f'<td class="num">{human(r["forecast_final"])}</td></tr>')

    anom = anomalies[anomalies["anomaly"]] if "anomaly" in anomalies.columns else anomalies.head(0)
    anom_list = ", ".join(anom["Country"].tolist()) if len(anom) else "ninguna"

    names_cons = {int(k): v for k, v in ml["cluster_consumo_names"].items()}
    cl_views = [
        ("cluster_consumo", "Por consumo / crecimiento", names_cons),
        ("cluster_preferencia", "Por preferencia de tipo", None),
        ("cluster_forecast", "Por trayectoria de forecast", None),
    ]
    pca_ev = ml.get("pca_explained_variance", [0, 0])
    cl_divs, cl_btns = [], []
    for i, (col, label, names) in enumerate(cl_views):
        sub = clusters.dropna(subset=[col])
        figc = pc.fig_cluster_scatter(sub, sub[col].astype(int).values, names, explained_variance=pca_ev)
        cl_divs.append(f'<div class="tab-panel {"active" if i==0 else ""}" data-tab-group="cl">{div(figc, "cl_"+str(i))}</div>')
        cl_btns.append(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="showTab(\'cl\',{i},this)">{label}</button>')

    gap = ml["coherence_gap_hierarchical"]
    imp = ml["classifier"]["importances"]
    top_imp = max(imp, key=imp.get)
    clf_acc = ml["classifier"]["cv_accuracy"]
    clf_base = ml["classifier"]["baseline"]
    clf_beats = clf_acc > clf_base + 0.02
    clf_text = (f"acierta {clf_acc*100:.0f}% (CV) sobre un baseline de {clf_base*100:.0f}%, "
                f"señalando <em>{top_imp}</em> como variable más informativa."
                if clf_beats else
                f"alcanza {clf_acc*100:.0f}% (CV) frente a un baseline de {clf_base*100:.0f}%: "
                f"<strong>no supera al azar</strong>. Resultado negativo informativo — el patrón de "
                f"consumo no predice la preferencia de tipo, que responde más a geografía/agronomía "
                f"que a la dinámica de consumo.")

    fc_body = f"""
<section class="hero">
  <span class="eyebrow">Forecasting &amp; Machine Learning</span>
  <h1>Hacia dónde va el <span class="accent">consumo</span>, y dónde crecer</h1>
</section>
<div class="stats">{fc_stats}</div>

<h2 class="section-title">Proyección de consumo (rangos futuros)</h2>
<div class="tabs">{''.join(fc_btns)}</div>
<div class="card">{''.join(fc_divs)}</div>

<div class="grid-2">
  <div class="card pad">
    <h3>Resumen de forecast — top países</h3>
    <div class="table-wrap"><table class="data">
      <tr><th>País</th><th>Modelo</th><th>MAPE</th><th>2019/20</th><th>2024/25</th></tr>
      {fc_rows}
    </table></div>
  </div>
  <div class="card pad">
    <h3>Modelos ML aplicados</h3>
    <div class="callout"><strong>Modelo global pooled (HistGradientBoosting):</strong> entrena una sola
    máquina sobre las 53 series apiladas, compartiendo señal entre países y prediciendo la tasa de
    crecimiento (los árboles no extrapolan niveles). En el mismo holdout de 5 años obtiene
    {mc['pooled_mape_median']:.1f}% de MAPE mediano vs. {mc['classic_mape_median']:.1f}% de los modelos
    clásicos por-serie: para estas series maduras y suaves, {'los clásicos ganan' if classic_wins else 'el pooled gana'}.
    Se elige el modelo simple <em>por evidencia</em>, tras evaluar el enfoque ML moderno de forma justa.</div>
    <div class="callout"><strong>Clasificador de preferencia de tipo:</strong> un RandomForest intenta
    predecir si un mercado es Arabica-dominante desde su patrón de consumo; {clf_text}</div>
    <div class="callout"><strong>Detección de anomalías (IsolationForest):</strong> mercados con patrón
    atípico multivariante: {anom_list}. Revisar antes de confiar en su forecast.</div>
    <p class="note">Coherencia jerárquica (suma países vs. global directo): {gap:+.1%}.</p>
  </div>
</div>

<h2 class="section-title">Estrategia comercial: dónde invertir</h2>
<div class="card">{div(pc.fig_quadrant_matrix(quad), "fc_quad")}</div>

<div class="card pad">
  <h3>Mercados prioritarios de exportación</h3>
  <div class="table-wrap"><table class="data">
    <tr><th>País</th><th>Tipo</th><th>CAGR reciente</th><th>Nivel medio</th><th>Δ Consumo 2024/25</th></tr>
    {"".join(f'<tr><td>{r["Country"]}</td><td>{pill(r["Coffee type"], pc.COFFEE_TYPE_COLORS.get(r["Coffee type"], "#E08A1E"))}</td><td class="num">{r["cagr_recent"]*100:.1f}%</td><td class="num">{human(r["level_mean"])}</td><td class="num">{human(r["incremental_2024_25"])}</td></tr>' for _, r in priority.head(8).iterrows())}
  </table></div>
</div>

<h2 class="section-title">Segmentación de mercados</h2>
<div class="tabs">{''.join(cl_btns)}</div>
<div class="card">{''.join(cl_divs)}</div>
"""

    # ======================== PÁGINA 3 — GenAI ======================== #
    genai_body = """
<section class="hero">
  <span class="eyebrow">Bonus</span>
  <h1>Capa de <span class="accent">IA generativa</span></h1>
  <p class="lead">Espacio reservado para las capacidades de GenAI/LLM sobre esta solución. En evaluación.</p>
</section>
<div class="soon">
  <div class="big">☕🤖</div>
  <span class="badge-soon">PRÓXIMAMENTE</span>
  <p class="section-sub" style="margin:16px auto 0;max-width:620px">Esta página se deja intencionalmente
  en blanco mientras se definen las opciones de integración. Dos propuestas concretas e implementables:</p>
  <div class="soon-grid">
    <div class="card pad">
      <h3>① Narrativas automáticas de insights</h3>
      <p class="note">Un LLM (Claude) toma los resultados ya calculados por el pipeline (ranking de CAGR,
      perfiles de cluster, resumen de forecast) y redacta insights de negocio y recomendaciones en
      lenguaje natural. Bajo riesgo: el modelo resume resultados validados, no interpreta datos crudos.</p>
    </div>
    <div class="card pad">
      <h3>② Asistente de preguntas (text-to-pandas)</h3>
      <p class="note">Un asistente conversacional que responde preguntas de negocio ("¿qué mercados
      crecen más rápido?") generando y ejecutando código pandas sobre los datos y el forecast, con la
      tabla o el gráfico de soporte. Permite exploración sin escribir código.</p>
    </div>
  </div>
</div>
"""

    (WEB_DIR / "index.html").write_text(shell("EDA", "EDA", eda_body), encoding="utf-8")
    (WEB_DIR / "forecasting.html").write_text(shell("Forecasting & ML", "Forecasting & ML", fc_body), encoding="utf-8")
    (WEB_DIR / "genai.html").write_text(shell("GenAI", "GenAI", genai_body), encoding="utf-8")
    print("Informe web generado en reports/web/ (abre index.html en el navegador).")


if __name__ == "__main__":
    build()
