"""Figuras Plotly con estética café para el informe web interactivo.

El informe se compromete a un único look cálido ("café"), coherente con las maquetas de referencia.
Los 4 colores categóricos de tipo de café se eligen por separación perceptual real: Arabica y Robusta
antes compartían la misma familia naranja/caramelo (#C77E12 vs #B5651D) y eran casi indistinguibles a
tamaño de burbuja pequeño (reportado directamente sobre la matriz tamaño×crecimiento) — Robusta se
movió a un espresso oscuro para separarse por luminosidad, no solo por matiz, lo que además ayuda
bajo daltonismo rojo-verde.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Paleta café: 4 matices distintos + separación por luminosidad, no solo por matiz ------------ #
COFFEE_TYPE_COLORS = {
    "Arabica": "#C77E12",          # ámbar / tueste dorado — el más brillante y saturado
    "Arabica/Robusta": "#2A9D8F",  # teal — único hue frío, ancla de separación CVD
    "Robusta": "#5C3A21",          # espresso oscuro — antes "caramelo" #B5651D, muy cerca de Arabica
    "Robusta/Arabica": "#B23A48",  # vino
}
# Paleta categórica de continentes (validada aparte, distinta de la de tipos de café para no
# confundir ambas dimensiones cuando aparecen en la misma página; subconjunto blue/aqua/green/violet
# de la paleta de referencia de la skill dataviz, validado CVD-safe con validate_palette.js).
CONTINENT_COLORS = {
    "América": "#2a78d6", "África": "#1baf7a", "Asia": "#008300", "Oceanía": "#4a3aa7",
}

QUADRANT_COLORS = {
    "Priorizar": "#2F7D32", "Explorar": "#C77E12", "Defender": "#6B5647", "Baja prioridad": "#B23A48",
}

# --- Tokens de chrome (tema café claro, lienzo técnico) ------------------------------------------ #
SURFACE = "#FFFFFF"      # blanco puro (superficie de gráfico, se funde con la tarjeta .card)
INK = "#2E1D12"          # espresso (texto primario)
INK_SOFT = "#6B5647"     # marrón suave (texto secundario)
GRID = "rgba(46, 29, 18, 0.10)"  # retícula hairline: espresso a baja opacidad, no un tono plano
ACCENT = "#E08A1E"       # naranja acento

FORECAST_COLOR = "#E08A1E"
HISTORY_COLOR = "#4A2E1C"
BAND_COLOR = "rgba(224,138,30,0.18)"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Solo para ticks/valores numéricos (look técnico de hoja de datos); sin CDN externo, todo system font.
MONO_FONT = '"Cascadia Mono", "Segoe UI Mono", ui-monospace, Consolas, "Liberation Mono", Menlo, monospace'


def _base_layout(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=INK, family=FONT), x=0.01, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SOFT, size=13),
        margin=dict(l=60, r=24, t=48 if title else 20, b=48),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
        hoverlabel=dict(font=dict(family=MONO_FONT, size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, color=INK_SOFT, zeroline=False,
                     tickfont=dict(family=MONO_FONT), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, color=INK_SOFT, zeroline=False,
                     tickfont=dict(family=MONO_FONT), automargin=True)
    return fig


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def fig_evolution_dual_selector(long_df: pd.DataFrame) -> go.Figure:
    """Evolución con DOS selectores: País (🌍 Global + 53 países) y Tipo de Café (4 tipos)."""
    valid = long_df[long_df["is_valid_series"]]
    countries = ["🌍 Global"] + sorted(valid["Country"].unique())
    types = sorted(valid["Coffee type"].unique())

    fig = go.Figure()
    traces_by_country_type = {}
    trace_count = 0

    for country in countries:
        for ctype in types:
            if country == "🌍 Global":
                s = valid[valid["Coffee type"] == ctype].groupby("fiscal_year_start")["consumption"].sum()
            else:
                s = valid[(valid["Country"] == country) & (valid["Coffee type"] == ctype)].groupby("fiscal_year_start")["consumption"].sum()

            visible = (country == countries[0] and ctype == types[0])
            label = f"{country} · {ctype}"
            fig.add_trace(go.Scatter(
                x=s.index.tolist() if len(s) > 0 else [], y=s.values.tolist() if len(s) > 0 else [],
                mode="lines+markers", name=label, visible=visible,
                line=dict(color=HISTORY_COLOR, width=2.5), marker=dict(size=5, color=HISTORY_COLOR),
                hovertemplate=label + "<br>%{x}: %{y:,.0f} tazas<extra></extra>",
            ))
            traces_by_country_type[(country, ctype)] = trace_count
            trace_count += 1

    n_traces = trace_count

    # Selector de país
    country_buttons = []
    for country in countries:
        vis = [False] * n_traces
        for ctype in types:
            if (country, ctype) in traces_by_country_type and ctype == types[0]:
                vis[traces_by_country_type[(country, ctype)]] = True
        country_buttons.append(dict(label=country, method="update",
                                   args=[{"visible": vis}, {"title.text": f"Evolución — {country}"}]))

    # Selector de tipo
    type_buttons = []
    for ctype in types:
        vis = [False] * n_traces
        if (countries[0], ctype) in traces_by_country_type:
            vis[traces_by_country_type[(countries[0], ctype)]] = True
        type_buttons.append(dict(label=ctype, method="update",
                                args=[{"visible": vis}, {}]))

    # Los nombres de país son largos (p.ej. "Lao People's Democratic Republic"), así que el ancho del
    # dropdown cerrado se dimensiona por la etiqueta más larga del menú (~260px). Van EN LA MISMA FILA
    # (mismo y), uno junto al otro con suficiente hueco horizontal (x=0.0 y x=0.40) para no tocarse. El
    # título usa la referencia "container" (relativa a TODA la imagen) mientras que los
    # updatemenus/anotaciones usan "paper" (relativa solo al área del gráfico) — escalas distintas, por
    # eso se fijan explícitamente uno justo debajo del otro con un margen superior ajustado (no el
    # default, que deja un hueco grande).
    LABEL_Y, MENU_Y = 1.20, 1.12
    fig.update_layout(showlegend=False, updatemenus=[
        dict(buttons=country_buttons, direction="down", x=0.0, xanchor="left", y=MENU_Y, yanchor="top",
             bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=11), active=0),
        dict(buttons=type_buttons, direction="down", x=0.40, xanchor="left", y=MENU_Y, yanchor="top",
             bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=11), active=0),
    ])
    fig.add_annotation(text="País", xref="paper", yref="paper", x=0.0, xanchor="left", y=LABEL_Y,
                       yanchor="top", showarrow=False, font=dict(size=11, color=INK_SOFT, family=FONT))
    fig.add_annotation(text="Tipo de café", xref="paper", yref="paper", x=0.40, xanchor="left", y=LABEL_Y,
                       yanchor="top", showarrow=False, font=dict(size=11, color=INK_SOFT, family=FONT))
    fig = _base_layout(fig, "Evolución del consumo — 🌍 Global · Arabica", height=400)
    fig.update_layout(margin=dict(t=100, b=40), title=dict(y=0.96, yanchor="top"))
    fig.update_yaxes(title="Consumo (tazas)")
    return fig


def fig_distribution_by_continent(long_df: pd.DataFrame, stats: dict) -> go.Figure:
    """Distribución del consumo por país (último año), por continente, en escala log.

    Un boxplot por continente (con cada país como punto individual) muestra la misma asimetría que
    un histograma plano, pero además responde "¿quién compone cada nivel?": la mediana y el rango
    de cada continente, y qué país concreto es cada outlier (ej. Brasil en América) al pasar el mouse.
    """
    valid = long_df[long_df["is_valid_series"]]
    d = valid[(valid["fiscal_year_start"] == stats["year"]) & (valid["consumption"] > 0)]
    order = d.groupby("continent")["consumption"].median().sort_values(ascending=False).index

    fig = go.Figure()
    for cont in order:
        sub = d[d["continent"] == cont]
        fig.add_trace(go.Box(
            y=sub["consumption"], x=[cont] * len(sub), name=cont, text=sub["Country"],
            marker=dict(color=CONTINENT_COLORS.get(cont, ACCENT), size=6),
            line=dict(color=CONTINENT_COLORS.get(cont, ACCENT)),
            boxpoints="all", pointpos=0, jitter=0.5, showlegend=False,
            hovertemplate="<b>%{text}</b><br>%{y:,.0f} tazas<extra>" + cont + "</extra>",
        ))
    fig = _base_layout(fig, f"Distribución del consumo por continente ({stats['year']}/"
                            f"{str(stats['year']+1)[-2:]}) — escala logarítmica", height=380)
    fig.update_yaxes(title="Consumo (tazas)", type="log")
    fig.update_xaxes(title="")
    return fig


def fig_pareto(pareto_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Pareto de concentración: barras = % del total por país, línea = % acumulado.

    Ambas series están en la misma escala (0-100%) para respetar un único eje — evita el gráfico de
    doble eje (barras absolutas + línea acumulada) que dificulta la lectura.
    """
    df = pareto_df.head(top_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Country"], y=df["share_pct"], marker_color=ACCENT, name="% del consumo total",
        hovertemplate="%{x}<br>%{y:.1f}% del total<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Country"], y=df["cum_share_pct"], mode="lines+markers", name="% acumulado",
        line=dict(color=HISTORY_COLOR, width=2.5), marker=dict(size=6),
        hovertemplate="Acumulado hasta %{x}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line=dict(color=INK_SOFT, width=1, dash="dot"),
                  annotation_text="80%", annotation_position="right", annotation_font=dict(color=INK_SOFT, size=11))
    fig = _base_layout(fig, f"Concentración del consumo — Pareto (top {top_n} países)", height=380)
    fig.update_yaxes(title="% del consumo", range=[0, 105])
    fig.update_xaxes(tickangle=-40)
    return fig


def fig_quadrant_matrix(quad_df: pd.DataFrame) -> go.Figure:
    """Matriz tamaño × crecimiento con cuadrantes de negocio (Priorizar/Defender/Explorar/Baja
    prioridad), fondos sutiles por cuadrante y etiquetas de zona.

    El eje Y es logarítmico (los niveles de consumo abarcan varios órdenes de magnitud); el rango
    del eje y las posiciones de las etiquetas se calculan en espacio log10 explícitamente — Plotly
    exige log10 para `range` en ejes log, pero valores de dato normales para shapes/anotaciones.
    Se excluyen países con consumo reciente = 0 (no representable en escala log).
    """
    df = quad_df[quad_df["level_last5_mean"] > 0].copy()
    x_med, y_med = quad_df.attrs["x_median"] * 100, quad_df.attrs["y_median"]

    x_data = df["cagr_recent"] * 100
    x_pad = (x_data.max() - x_data.min()) * 0.08
    x_lo, x_hi = x_data.min() - x_pad, x_data.max() + x_pad

    y_log = np.log10(df["level_last5_mean"])
    y_pad = (y_log.max() - y_log.min()) * 0.08
    y_lo_log, y_hi_log = y_log.min() - y_pad, y_log.max() + y_pad
    y_lo, y_hi = 10 ** y_lo_log, 10 ** y_hi_log

    sizeref = 2.0 * df["total"].max() / (60.0 ** 2)
    fig = go.Figure()

    quads = [
        (x_med, x_hi, y_med, y_hi, "Priorizar"),
        (x_lo, x_med, y_med, y_hi, "Defender"),
        (x_med, x_hi, y_lo, y_med, "Explorar"),
        (x_lo, x_med, y_lo, y_med, "Baja prioridad"),
    ]
    for x0, x1, y0, y1, label in quads:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref="x", yref="y",
                     fillcolor=QUADRANT_COLORS[label], opacity=0.07, line=dict(width=0), layer="below")

    for t in sorted(df["Coffee type"].unique()):
        sub = df[df["Coffee type"] == t]
        fig.add_trace(go.Scatter(
            x=sub["cagr_recent"] * 100, y=sub["level_last5_mean"], mode="markers", name=t, text=sub["Country"],
            customdata=sub["quadrant"],
            marker=dict(size=sub["total"], sizemode="area", sizeref=sizeref, sizemin=4,
                        color=COFFEE_TYPE_COLORS.get(t, ACCENT), line=dict(width=0.5, color=SURFACE)),
            hovertemplate="<b>%{text}</b><br>CAGR reciente: %{x:.1f}%<br>Consumo reciente: %{y:,.0f} tazas"
                          "<br>Cuadrante: %{customdata}<extra>" + t + "</extra>",
        ))

    fig.add_vline(x=x_med, line=dict(color=INK_SOFT, width=1, dash="dash"))
    fig.add_hline(y=y_med, line=dict(color=INK_SOFT, width=1, dash="dash"))

    # Etiquetas de zona en coordenadas de PAPEL (no de datos): Plotly/kaleido tiene un bug conocido
    # que no renderiza anotaciones en espacio de datos sobre ejes logarítmicos. Las esquinas del
    # área de trazado son una posición estable independientemente del rango exacto de los datos.
    corner_pos = {
        "Priorizar": (0.98, 0.97, "right", "top"),
        "Defender": (0.02, 0.97, "left", "top"),
        "Explorar": (0.98, 0.03, "right", "bottom"),
        "Baja prioridad": (0.02, 0.03, "left", "bottom"),
    }
    for label, (px, py, xanchor, yanchor) in corner_pos.items():
        fig.add_annotation(x=px, y=py, xref="x domain", yref="y domain", text=f"<b>{label}</b>",
                           showarrow=False, xanchor=xanchor, yanchor=yanchor,
                           font=dict(size=12, color=QUADRANT_COLORS[label]), opacity=0.95)

    fig = _base_layout(fig, "Matriz tamaño × crecimiento — dónde priorizar inversión comercial", height=460)
    fig.update_xaxes(title="CAGR reciente (10 años, %)", showgrid=True, gridcolor=GRID, range=[x_lo, x_hi])
    fig.update_yaxes(title="Consumo reciente (promedio últimos 5 años, tazas)", type="log",
                     range=[y_lo_log, y_hi_log])
    return fig


# --------------------------------------------------------------------------- #
# Forecasting & ML
# --------------------------------------------------------------------------- #
def fig_forecast(res: dict, title: str) -> go.Figure:
    years, hist = res["years"], res["history"]
    fyears = np.arange(years[-1] + 1, years[-1] + 1 + len(res["point"]))
    fig = go.Figure()
    if res.get("lower") is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([fyears, fyears[::-1]]),
            y=np.concatenate([res["upper"], res["lower"][::-1]]),
            fill="toself", fillcolor=BAND_COLOR, mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="Banda P10–P90", hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=years, y=hist, mode="lines+markers", name="Histórico",
        line=dict(color=HISTORY_COLOR, width=2.5), marker=dict(size=5),
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([[years[-1]], fyears]),
        y=np.concatenate([[hist[-1]], res["point"]]), mode="lines+markers",
        name=f"Forecast ({res['model']})", line=dict(color=FORECAST_COLOR, width=2.5, dash="dot"),
        marker=dict(size=6, color=FORECAST_COLOR),
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
    ))
    fig = _base_layout(fig, title)
    fig.update_yaxes(title="Consumo (tazas)")
    return fig


def fig_backtest_error_distribution(forecast_summary: pd.DataFrame) -> go.Figure:
    """Dispersión del MAPE de backtest entre las series por país — un promedio esconde la cola de
    series difíciles de proyectar; esta vista la hace visible (cada punto es un país, hover = país +
    modelo ganador).
    """
    df = forecast_summary
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df["backtest_mape"], x=["Backtest por país"] * len(df), text=df["series"],
        customdata=df["best_model"],
        marker=dict(color=ACCENT, size=6), line=dict(color=HISTORY_COLOR),
        boxpoints="all", pointpos=0, jitter=0.6, showlegend=False,
        hovertemplate="<b>%{text}</b> (%{customdata})<br>MAPE: %{y:.2f}%<extra></extra>",
    ))
    fig = _base_layout(fig, "Dispersión del error de backtest por país (rolling-origin, 1 paso)", height=360)
    fig.update_yaxes(title="MAPE backtest (%)")
    fig.update_xaxes(title="")
    return fig


def fig_cluster_scatter(pca_df: pd.DataFrame, labels: np.ndarray, cluster_names: dict = None,
                        explained_variance=None) -> go.Figure:
    df = pca_df.copy()
    df["cluster"] = labels
    ev = explained_variance or pca_df.attrs.get("explained_variance", [0, 0])
    fig = go.Figure()
    # Reutiliza la paleta de tipo de café (ya separada por matiz+luminosidad) en vez de duplicar los
    # mismos 4 hex sueltos; 2 neutros extra por si algún día hay más de 4 clusters.
    palette = list(COFFEE_TYPE_COLORS.values()) + ["#6B5647", "#4A2E1C"]
    for i, c in enumerate(sorted(df["cluster"].unique())):
        sub = df[df["cluster"] == c]
        name = cluster_names.get(c, f"Cluster {c}") if cluster_names else f"Cluster {c}"
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers", name=name, text=sub["Country"],
            marker=dict(size=11, color=palette[i % len(palette)], line=dict(width=0.5, color=SURFACE)),
            hovertemplate="<b>%{text}</b><extra>" + name + "</extra>",
        ))
    fig = _base_layout(fig, "Segmentación de países (proyección PCA)", height=400)
    fig.update_xaxes(title=f"PC1 ({ev[0]:.0%} var.)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=f"PC2 ({ev[1]:.0%} var.)")
    return fig
