"""Figuras Plotly con estética café para el informe web interactivo.

Los colores categóricos de los 4 tipos de café fueron validados con el validador de la skill dataviz
(CVD-safe en light y dark; solo un WARN de contraste que se cubre con leyendas + etiquetas directas).
El informe se compromete a un único look cálido ("café"), coherente con las maquetas de referencia.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Paleta café validada (no elegida a ojo; ver validate_palette.js) --------------------------- #
COFFEE_TYPE_COLORS = {
    "Arabica": "#C77E12",          # ámbar / tueste dorado
    "Arabica/Robusta": "#2A9D8F",  # teal (ancla de separación CVD)
    "Robusta": "#B5651D",          # caramelo
    "Robusta/Arabica": "#B23A48",  # vino
}
# Ramp secuencial marrón-café (light->dark) para heatmap / choropleth.
COFFEE_SEQUENTIAL = [
    [0.0, "#F5E9DA"], [0.2, "#E4C9A3"], [0.4, "#D0A56E"],
    [0.6, "#B5773C"], [0.8, "#8A5626"], [1.0, "#5E3A18"],
]

# Paleta categórica de continentes (validada aparte, distinta de la de tipos de café para no
# confundir ambas dimensiones cuando aparecen en la misma página; subconjunto blue/aqua/green/violet
# de la paleta de referencia de la skill dataviz, validado CVD-safe con validate_palette.js).
CONTINENT_COLORS = {
    "América": "#2a78d6", "África": "#1baf7a", "Asia": "#008300", "Oceanía": "#4a3aa7",
}

QUADRANT_COLORS = {
    "Priorizar": "#2F7D32", "Explorar": "#C77E12", "Defender": "#6B5647", "Baja prioridad": "#B23A48",
}

# --- Tokens de chrome (tema café claro) --------------------------------------------------------- #
SURFACE = "#FBF6EF"      # crema (superficie de gráfico)
INK = "#2E1D12"          # espresso (texto primario)
INK_SOFT = "#6B5647"     # marrón suave (texto secundario)
GRID = "#E7DAC8"         # retícula hairline
ACCENT = "#E08A1E"       # naranja acento

FORECAST_COLOR = "#E08A1E"
HISTORY_COLOR = "#4A2E1C"
BAND_COLOR = "rgba(224,138,30,0.18)"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=INK, family=FONT), x=0.01, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_SOFT, size=13),
        margin=dict(l=60, r=24, t=48 if title else 20, b=48),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, color=INK_SOFT, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, color=INK_SOFT, zeroline=False)
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


def fig_top_countries(long_df: pd.DataFrame, n: int = 15) -> go.Figure:
    valid = long_df[long_df["is_valid_series"]]
    totals = valid.groupby(["Country", "Coffee type"])["consumption"].sum().reset_index()
    totals = totals.sort_values("consumption", ascending=False).head(n).iloc[::-1]
    colors = [COFFEE_TYPE_COLORS.get(t, ACCENT) for t in totals["Coffee type"]]
    fig = go.Figure(go.Bar(
        x=totals["consumption"], y=totals["Country"], orientation="h", marker_color=colors,
        customdata=totals["Coffee type"],
        hovertemplate="%{y}<br>%{x:,.0f} tazas<br>Tipo: %{customdata}<extra></extra>",
    ))
    fig = _base_layout(fig, f"Top {n} países por consumo doméstico total (1990–2020)", height=440)
    fig.update_xaxes(title="Consumo total (tazas)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def _human_short(v: float) -> str:
    if v >= 1e9:
        return f"{v/1e9:.0f}B"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def fig_distribution_hist(long_df: pd.DataFrame, stats: dict) -> go.Figure:
    """Distribución del consumo por país (último año) en escala log, con media y mediana marcadas.

    La brecha entre media y mediana visualiza directamente la asimetría (unos pocos países muy
    grandes empujan la media muy por encima de la mediana). Se usa log10 porque el consumo abarca
    varios órdenes de magnitud — en escala lineal, Brasil comprime a todo el resto en una sola barra.
    """
    valid = long_df[long_df["is_valid_series"]]
    s = valid[valid["fiscal_year_start"] == stats["year"]]["consumption"]
    s = s[s > 0]
    log_s = np.log10(s)

    fig = go.Figure(go.Histogram(
        x=log_s, nbinsx=14, marker_color=ACCENT, opacity=0.85,
        hovertemplate="%{y} países<extra></extra>",
    ))
    fig.add_vline(x=np.log10(stats["mean"]), line=dict(color=HISTORY_COLOR, width=2, dash="dash"))
    fig.add_vline(x=np.log10(stats["median"]), line=dict(color="#2A9D8F", width=2, dash="dot"))
    fig.add_annotation(x=np.log10(stats["mean"]), y=1.0, yref="y domain", yanchor="bottom", showarrow=False,
                       text=f"Media: {_human_short(stats['mean'])}", font=dict(size=11, color=HISTORY_COLOR))
    fig.add_annotation(x=np.log10(stats["median"]), y=0.88, yref="y domain", yanchor="bottom", showarrow=False,
                       text=f"Mediana: {_human_short(stats['median'])}", font=dict(size=11, color="#2A9D8F"))

    tick_log = np.arange(np.floor(log_s.min()), np.ceil(log_s.max()) + 1)
    fig = _base_layout(fig, f"Distribución del consumo por país ({stats['year']}/{str(stats['year']+1)[-2:]}) "
                            "— escala logarítmica", height=320)
    fig.update_xaxes(title="Consumo (tazas)", tickvals=tick_log, ticktext=[_human_short(10**v) for v in tick_log])
    fig.update_yaxes(title="# de países")
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
    fig.update_yaxes(title="% del consumo mundial (individual y acumulado)", range=[0, 105])
    fig.update_xaxes(tickangle=-40)
    return fig


def fig_waterfall(wf: dict) -> go.Figure:
    """Puente del cambio de consumo global entre las dos últimas campañas: total inicial, la
    contribución de cada país (top movers + resto agregado), y el total final."""
    y0_label = f"{wf['y0']}/{str(wf['y0'] + 1)[-2:]}"
    y1_label = f"{wf['y1']}/{str(wf['y1'] + 1)[-2:]}"
    labels = [f"Total {y0_label}"] + [c for c, _ in wf["items"]] + [f"Total {y1_label}"]
    values = [wf["total_y0"]] + [v for _, v in wf["items"]] + [wf["total_y1"]]
    measures = ["absolute"] + ["relative"] * len(wf["items"]) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing=dict(marker_color="#2F7D32"), decreasing=dict(marker_color="#B23A48"),
        totals=dict(marker_color=INK_SOFT),
        connector=dict(line=dict(color=GRID, width=1)),
        hovertemplate="%{x}<br>%{y:,.0f} tazas<extra></extra>",
    ))
    fig = _base_layout(fig, f"Qué explica el cambio {y0_label} → {y1_label}", height=380)
    # El eje se acota (no parte de 0) para que el detalle del puente sea legible: el cambio total es
    # ~0.5% de la base, invisible en una escala completa. Se declara explícitamente para no engañar.
    lo = min(wf["total_y0"], wf["total_y1"]) - abs(wf["total_delta"]) * 2.2
    hi = max(wf["total_y0"], wf["total_y1"]) + abs(wf["total_delta"]) * 2.2
    # Nota: el eje se acota (no parte de 0) para que el puente sea legible — el cambio total es
    # ~0.5% de la base y sería invisible en escala completa. Se declara en el texto de la página
    # (no dentro de la figura, para evitar colisión de coordenadas con el título en distintos anchos).
    fig.update_yaxes(title="Consumo (tazas)", range=[lo, hi])
    fig.update_xaxes(tickangle=-30)
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
                        color=COFFEE_TYPE_COLORS.get(t, ACCENT), line=dict(width=0.5, color="#FBF6EF")),
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


def fig_type_distribution(long_df: pd.DataFrame) -> go.Figure:
    valid = long_df[long_df["is_valid_series"]]
    by_type = valid.groupby("Coffee type")["consumption"].sum().sort_values(ascending=False)
    colors = [COFFEE_TYPE_COLORS.get(t, ACCENT) for t in by_type.index]
    fig = go.Figure(go.Pie(
        labels=by_type.index, values=by_type.values, hole=0.55,
        marker=dict(colors=colors), sort=False,
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Consumo acumulado por tipo de café", font=dict(size=17, color=INK, family=FONT), x=0.01),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=20, r=20, t=44, b=20), height=340, legend=dict(font=dict(color=INK)),
    )
    return fig


def fig_type_trend_by_scope(long_df: pd.DataFrame) -> go.Figure:
    """Tendencia por tipo de café (4 líneas) con selector de alcance: Global o por continente."""
    valid = long_df[long_df["is_valid_series"]]
    scopes = ["Global"] + sorted(valid["continent"].dropna().unique().tolist())
    types = sorted(valid["Coffee type"].unique())

    fig = go.Figure()
    trace_idx = {}
    idx = 0
    for scope in scopes:
        trace_idx[scope] = []
        scoped = valid if scope == "Global" else valid[valid["continent"] == scope]
        for t in types:
            s = scoped[scoped["Coffee type"] == t].groupby("fiscal_year_start")["consumption"].sum()
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=t, legendgroup=t,
                visible=(scope == "Global"), showlegend=(scope == "Global"),
                line=dict(color=COFFEE_TYPE_COLORS.get(t, ACCENT), width=2.5),
                hovertemplate=f"{t} — {scope}<br>" + "%{x}: %{y:,.0f} tazas<extra></extra>",
            ))
            trace_idx[scope].append(idx)
            idx += 1

    n = idx
    buttons = []
    for scope in scopes:
        vis = [False] * n
        leg = [False] * n
        for i in trace_idx[scope]:
            vis[i] = True
            leg[i] = True
        buttons.append(dict(label=scope, method="update",
                            args=[{"visible": vis, "showlegend": leg},
                                  {"title.text": f"Tendencia por tipo de café — {scope}"}]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.24, yanchor="top",
        bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=12),
    )])
    fig = _base_layout(fig, "Tendencia por tipo de café — Global", height=360)
    fig.update_layout(margin=dict(t=80))
    fig.update_yaxes(title="Consumo (tazas)")
    return fig


def fig_continent_distribution(long_df: pd.DataFrame) -> go.Figure:
    valid = long_df[long_df["is_valid_series"]]
    by_cont = valid.groupby("continent")["consumption"].sum().sort_values(ascending=False)
    colors = [CONTINENT_COLORS.get(c, ACCENT) for c in by_cont.index]
    fig = go.Figure(go.Pie(
        labels=by_cont.index, values=by_cont.values, hole=0.55,
        marker=dict(colors=colors), sort=False,
        hovertemplate="%{label}<br>%{value:,.0f} tazas (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Consumo acumulado por continente", font=dict(size=17, color=INK, family=FONT), x=0.01),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=20, r=20, t=44, b=20), height=340, legend=dict(font=dict(color=INK)),
    )
    return fig


def fig_geo_bubble(bubble_df: pd.DataFrame, size_col: str = "total", title: str = "") -> go.Figure:
    """Mapa geográfico de burbujas: tamaño=consumo HISTÓRICO total, color=tipo de café.

    Muestra únicamente datos observados (1990–2020) — el forecast se presenta en la página
    Forecasting & ML, no aquí, para no mezclar hecho con proyección en el EDA.
    """
    df = bubble_df.copy()
    sizeref = 2.0 * df[size_col].max() / (55.0 ** 2)
    fig = go.Figure()
    for t in sorted(df["Coffee type"].unique()):
        sub = df[df["Coffee type"] == t]
        fig.add_trace(go.Scattergeo(
            locations=sub["iso3"], locationmode="ISO-3", text=sub["Country"], name=t,
            marker=dict(size=sub[size_col], sizemode="area", sizeref=sizeref, sizemin=3,
                        color=COFFEE_TYPE_COLORS.get(t, ACCENT), line=dict(width=0.5, color="#FBF6EF")),
            customdata=sub[["total"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>Consumo histórico total: %{customdata[0]:,.0f} tazas"
                          "<extra>" + t + "</extra>",
        ))
    fig.update_geos(
        showcountries=True, countrycolor="#E7DAC8", showland=True, landcolor="#F0E6D6",
        showocean=True, oceancolor="#FBF6EF", showframe=False, coastlinecolor="#E7DAC8",
        projection_type="natural earth", bgcolor=SURFACE,
    )
    fig.update_layout(
        title=dict(text=title or "Consumo doméstico histórico por país y tipo de café", font=dict(size=17, color=INK, family=FONT), x=0.01),
        paper_bgcolor=SURFACE, font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=10, r=10, t=44, b=10), height=440, legend=dict(font=dict(color=INK)),
    )
    return fig


def fig_heatmap(long_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    valid = long_df[long_df["is_valid_series"]]
    top = valid.groupby("Country")["consumption"].sum().sort_values(ascending=False).head(top_n).index
    pivot = valid[valid["Country"].isin(top)].pivot(index="Country", columns="fiscal_year_start", values="consumption")
    pivot = pivot.div(pivot.max(axis=1), axis=0).loc[top[::-1]]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=COFFEE_SEQUENTIAL,
        colorbar=dict(title="% de su máx."),
        hovertemplate="%{y}<br>%{x}: %{z:.0%} del máximo histórico<extra></extra>",
    ))
    fig = _base_layout(fig, f"Trayectoria de consumo normalizada — Top {top_n} países", height=420)
    fig.update_yaxes(showgrid=False)
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


def fig_cluster_scatter(pca_df: pd.DataFrame, labels: np.ndarray, cluster_names: dict = None,
                        explained_variance=None) -> go.Figure:
    df = pca_df.copy()
    df["cluster"] = labels
    ev = explained_variance or pca_df.attrs.get("explained_variance", [0, 0])
    fig = go.Figure()
    palette = ["#C77E12", "#2A9D8F", "#B5651D", "#B23A48", "#6B5647", "#4A2E1C"]
    for i, c in enumerate(sorted(df["cluster"].unique())):
        sub = df[df["cluster"] == c]
        name = cluster_names.get(c, f"Cluster {c}") if cluster_names else f"Cluster {c}"
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers", name=name, text=sub["Country"],
            marker=dict(size=11, color=palette[i % len(palette)], line=dict(width=0.5, color="#FBF6EF")),
            hovertemplate="<b>%{text}</b><extra>" + name + "</extra>",
        ))
    fig = _base_layout(fig, "Segmentación de países (proyección PCA)", height=400)
    fig.update_xaxes(title=f"PC1 ({ev[0]:.0%} var.)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=f"PC2 ({ev[1]:.0%} var.)")
    return fig


def fig_model_comparison(bt_df: pd.DataFrame, title: str = "Comparación de modelos (MAPE rolling)") -> go.Figure:
    df = bt_df.sort_values("mape")
    fig = go.Figure(go.Bar(
        x=df["mape"], y=df["model"], orientation="h", marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.2f}% MAPE<extra></extra>",
    ))
    fig = _base_layout(fig, title, height=300)
    fig.update_xaxes(title="MAPE (%)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig
