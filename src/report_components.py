"""Componentes HTML reutilizables del informe web ('UI kit'): page shell, navegación, tarjetas de
estadística, cajas de insight y formateo de números. Sin lógica de negocio ni acceso a datos —
solo toman valores ya calculados y devuelven HTML. Ver `src/report_sections.py` para el contenido
de cada página.
"""

PAGES = [("index.html", "EDA"), ("forecasting.html", "Forecasting & ML"), ("genai.html", "GenAI")]


def div(fig, name):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=name,
                       config={"displayModeBar": False, "responsive": True})


def tab_select(group: str, sections: list) -> str:
    """<select> agrupado que alterna paneles data-tab-group=`group` por índice — alternativa a una
    fila de botones cuando hay demasiadas opciones (ej. selector de país entre 53 series).
    `sections`: lista de (nombre_de_grupo, [etiquetas]) en el mismo orden que los paneles renderizados.
    """
    idx, parts = 0, []
    for section_name, labels in sections:
        opts = "".join(f'<option value="{idx + i}">{lbl}</option>' for i, lbl in enumerate(labels))
        idx += len(labels)
        parts.append(f'<optgroup label="{section_name}">{opts}</optgroup>')
    return f'<select class="tab-select" onchange="showTabSelect(\'{group}\', this)">{"".join(parts)}</select>'


def nav(active):
    links = "".join(
        f'<a href="{href}" class="{"active" if label == active else ""}">{label}</a>'
        for href, label in PAGES
    )
    return f"""<nav class="nav">
  <div class="brand">
    <span class="brand-mark">HG</span>
    <div>
      <span class="brand-text">High Garden Coffee</span>
      <span class="brand-sub">Market Intelligence</span>
    </div>
  </div>
  <div class="nav-links">{links}</div>
</nav>"""


def shell(title, active, body, css_version=""):
    """`css_version` (ej. mtime de styles.css) evita que el navegador sirva una hoja de estilos vieja
    cacheada mientras se itera sobre reports/web/assets/styles.css entre builds sucesivos."""
    css_href = f"assets/styles.css?v={css_version}" if css_version else "assets/styles.css"
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · High Garden Coffee</title>
<link rel="stylesheet" href="{css_href}">
<script src="assets/plotly.min.js"></script>
</head>
<body>
{nav(active)}
<div class="wrap">
{body}
</div>
<footer>High Garden Coffee · Informe analítico reproducible desde el pipeline ML · Datos ICO 1990–2020
(consumo doméstico, en tazas)</footer>
<script>
function showTab(group, idx, btn) {{
  document.querySelectorAll('[data-tab-group="'+group+'"]').forEach((p,i)=>p.classList.toggle('active', i===idx));
  btn.parentNode.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active', i===idx));
  window.dispatchEvent(new Event('resize'));
}}
function showTabSelect(group, select) {{
  const idx = parseInt(select.value, 10);
  document.querySelectorAll('[data-tab-group="'+group+'"]').forEach((p,i)=>p.classList.toggle('active', i===idx));
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


def pipeline(steps: list) -> str:
    """Diagrama de pipeline horizontal (scroll en mobile): pasos numerados conectados por flechas.
    `steps`: lista de (título, descripción, tag técnico corto — ej. "RAG", "Human-in-the-loop").
    """
    parts = []
    for i, (title, desc, tag) in enumerate(steps):
        if i > 0:
            parts.append('<div class="pipeline-arrow">&#8594;</div>')
        parts.append(f'<div class="pipeline-step"><span class="step-num">{i + 1}</span>'
                     f'<h4>{title}</h4><p>{desc}</p><span class="step-tag">{tag}</span></div>')
    return f'<div class="pipeline">{"".join(parts)}</div>'


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{text}</span>'
