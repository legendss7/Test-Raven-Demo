# ================================================================
# Raven Matrices PRO — 60 Ítems (estructura Big Five PRO)
# - Auto-avance, fondo blanco, UI PRO, tarjetas + KPIs
# - Carga perezosa: muestra SOLO la imagen del ítem actual
# - Imágenes desde GitHub: raven_pagina_001.png ... raven_pagina_060.png
# - Gabarito opcional: answer_key.json en el mismo repo
# ================================================================
import io
import json
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Dependencias opcionales
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except Exception:
    HAS_MPL = False

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


# ---------------------------------------------------------------
# CONFIG & ESTILO
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Raven PRO | Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Ocultar sidebar */
[data-testid="stSidebar"] { display:none !important; }

/* Base */
html, body, [data-testid="stAppViewContainer"]{
  background:#ffffff !important; color:#111 !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{ max-width:1200px; padding-top:0.8rem; padding-bottom:2rem; }

/* Tarjetas */
.card{
  border:1px solid #eee; border-radius:14px; background:#fff;
  box-shadow: 0 2px 0 rgba(0,0,0,0.03); padding:18px;
}

/* KPIs */
.kpi-grid{
  display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr));
  gap:12px; margin:10px 0 6px 0;
}
.kpi{
  border:1px solid #eee; border-radius:14px; background:#fff; padding:16px;
  position:relative; overflow:hidden;
}
.kpi::after{
  content:""; position:absolute; inset:0;
  background: linear-gradient(120deg, rgba(255,255,255,0) 0%,
    rgba(240,240,240,0.7) 45%, rgba(255,255,255,0) 90%);
  transform: translateX(-100%); animation: shimmer 2s ease-in-out 1;
}
@keyframes shimmer { to{ transform: translateX(100%);} }
.kpi .label{ font-size:.95rem; opacity:.85; }
.kpi .value{ font-size:2.1rem; font-weight:900; line-height:1; }

/* Pregunta */
.dim-title{
  font-size:clamp(2rem, 5vw, 2.8rem);
  font-weight:900; letter-spacing:.2px; line-height:1.12;
  margin:.2rem 0 .6rem 0;
  animation: slideIn .3s ease-out both;
}
@keyframes slideIn{
  from{ transform: translateY(6px); opacity:0; }
  to{ transform: translateY(0); opacity:1; }
}
.badge{
  display:inline-flex; align-items:center; gap:6px; padding:.25rem .55rem; font-size:.82rem;
  border-radius:999px; border:1px solid #eaeaea; background:#fafafa;
}
.q-image-wrap{
  display:flex; justify-content:center; align-items:center;
  background:#fafafa; border:1px solid #eee; border-radius:12px; padding:10px;
}
.options-grid{
  display:grid; grid-template-columns: repeat(auto-fit, minmax(60px,1fr));
  gap:8px; margin-top:12px;
}
.opt{
  border:1px solid #eee; border-radius:12px; background:#fff; padding:10px;
  text-align:center; cursor:pointer; transition: transform .06s ease;
}
.opt:hover{ transform: translateY(-1px); }
.opt.selected{ outline:2px solid #111; }

/* Tabla */
[data-testid="stDataFrame"] div[role="grid"]{ font-size:0.95rem; }

.small{ font-size:0.95rem; opacity:.9; }
hr{ border:none; border-top:1px solid #eee; margin:16px 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------------
SERIES = ["A","B","C","D","E","F"]
ITEMS_PER_SERIES = 10
TOTAL_ITEMS = len(SERIES)*ITEMS_PER_SERIES  # 60

# Mapeo de opciones por serie (común en muchas baterías Raven)
SERIES_OPTIONS = {
    "A": 6, "B": 6,  # primeras series suelen tener 6 opciones
    "C": 8, "D": 8, "E": 8, "F": 8
}

# Base URL de tu repo con las 60 imágenes 001..060
RAW_BASE = "https://raw.githubusercontent.com/legendss7/Test-Raven-Demo/main"

def image_url_for_index(idx_0_based: int) -> str:
    # idx: 0..59 -> nombre raven_pagina_001.png ... raven_pagina_060.png
    page_num = idx_0_based + 1
    return f"{RAW_BASE}/raven_pagina_{page_num:03d}.png"

# Gabarito: se intenta cargar answer_key.json de tu repo (si existe)
# Estructura esperada:
# {
#   "A": {"1": 3, "2": 2, ..., "10": 5},
#   "B": {...}, ... "F": {...}
# }
DEFAULT_KEY = {}  # si no hay clave, se dejan vacías (la app sigue operativa)

@st.cache_data(show_spinner=False)
def fetch_answer_key(url: str) -> Dict[str, Dict[int, int]]:
    if not HAS_REQUESTS:
        return DEFAULT_KEY
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        # normalizar a int
        out = {}
        for s, mp in data.items():
            out[s] = {int(k): int(v) for k, v in mp.items()}
        return out
    except Exception:
        return DEFAULT_KEY

ANSWER_KEY = fetch_answer_key(f"{RAW_BASE}/answer_key.json")


# ---------------------------------------------------------------
# MANIFIESTO DE ÍTEMS (60)
# ---------------------------------------------------------------
def build_manifest() -> List[Dict]:
    items = []
    for si, s in enumerate(SERIES):
        for k in range(1, ITEMS_PER_SERIES+1):
            iid = si*ITEMS_PER_SERIES + k  # 1..60
            n_opts = SERIES_OPTIONS[s]
            items.append({
                "id": iid,
                "series": s,
                "index": k,
                "n_options": n_opts,
                "correct": ANSWER_KEY.get(s, {}).get(k, None)
            })
    return items

if "items" not in st.session_state:
    st.session_state.items = build_manifest()


# ---------------------------------------------------------------
# ESTADO
# ---------------------------------------------------------------
if "stage" not in st.session_state: st.session_state.stage = "inicio"  # inicio | test | resultados
if "q_idx" not in st.session_state: st.session_state.q_idx = 0
if "answers" not in st.session_state: st.session_state.answers = {}  # item_id -> int (1..n)
if "fecha" not in st.session_state: st.session_state.fecha = None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun = False


# ---------------------------------------------------------------
# CÁLCULO DE RESULTADOS
# ---------------------------------------------------------------
def compute_results(answers: Dict[int,int], items: List[Dict]) -> Dict:
    rows = []
    for it in items:
        iid = it["id"]; s = it["series"]; k = it["index"]
        n_opts = it["n_options"]; correct = it["correct"]
        rta = answers.get(iid, None)
        is_correct = (rta == correct) if (correct is not None and rta is not None) else None
        rows.append({
            "item_id": iid, "serie": s, "index": k, "n_options": n_opts,
            "respuesta": rta, "correcta": correct, "acierto": is_correct
        })
    df = pd.DataFrame(rows).sort_values(["serie","index"]).reset_index(drop=True)
    n_answered = df["respuesta"].notna().sum()
    n_with_key = df["correcta"].notna().sum()
    n_correct = df["acierto"].sum() if df["acierto"].notna().any() else np.nan
    pct = (n_correct / n_with_key * 100) if (n_with_key and pd.notna(n_correct)) else np.nan

    by_series = (
        df.groupby("serie")
          .agg(total=("item_id","count"),
               contestados=("respuesta", lambda x: x.notna().sum()),
               con_clave=("correcta", lambda x: x.notna().sum()),
               aciertos=("acierto", lambda x: x.sum(skipna=True)))
          .reset_index()
    )
    by_series["% acierto"] = np.where(
        by_series["con_clave"]>0, (by_series["aciertos"]/by_series["con_clave"])*100, np.nan
    )

    # proxy dificultad (posición global A1..F10)
    df["pos_global"] = (df["serie"].map({s:i for i,s in enumerate(SERIES)}))*ITEMS_PER_SERIES + df["index"]
    df["dificultad"] = df["pos_global"]

    return {
        "df": df,
        "totales": {
            "contestados": int(n_answered),
            "con_clave": int(n_with_key),
            "aciertos": int(n_correct) if pd.notna(n_correct) else None,
            "porcentaje": float(pct) if pd.notna(pct) else None
        },
        "series": by_series.sort_values("serie").reset_index(drop=True)
    }


# ---------------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------------
def plot_series_bars(by_series: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_series["serie"], y=by_series["% acierto"],
        text=[f"{x:.1f}%" if pd.notna(x) else "-" for x in by_series["% acierto"]],
        textposition="outside",
        marker=dict(color=["#81B29A","#F2CC8F","#E07A5F","#9C6644","#6D597A","#84A59D"])
    ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis=dict(title="% acierto", range=[0,100]),
        xaxis=dict(title="Serie"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_difficulty_curve(df: pd.DataFrame):
    tmp = df.copy()
    tmp["ok"] = tmp["acierto"].fillna(False).astype(int)
    tmp = tmp.sort_values("pos_global")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tmp["pos_global"],
        y=tmp["ok"].rolling(5, min_periods=1).mean()*100,
        mode="lines+markers",
        name="Media móvil aciertos (x5)",
        line=dict(width=2)
    ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis=dict(title="% acierto (rolling x5)", range=[0,100]),
        xaxis=dict(title="Progresión (A1 → F10)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# ---------------------------------------------------------------
# PDF/HTML Reporte
# ---------------------------------------------------------------
def _pdf_card(ax, x,y,w,h,title,val):
    r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       edgecolor="#dddddd", facecolor="#ffffff")
    ax.add_patch(r)
    ax.text(x+w*0.06, y+h*0.60, title, fontsize=10, color="#333")
    ax.text(x+w*0.06, y+h*0.25, f"{val}", fontsize=20, fontweight='bold')

def build_pdf_report(result: Dict, fecha: str) -> bytes:
    df = result["df"]; tot = result["totales"]; bys = result["series"]
    pct = tot["porcentaje"] if tot["porcentaje"] is not None else 0.0

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # KPIs
        fig = plt.figure(figsize=(8.27,11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(.5,.94,"Informe — Matrices Progresivas de Raven", ha='center', fontsize=20, fontweight='bold')
        ax.text(.5,.91,f"Fecha: {fecha}", ha='center', fontsize=11)

        Y0 = .80; H = .10; W = .40; GAP = .02
        _pdf_card(ax, .06, Y0, W, H, "Ítems contestados", str(tot["contestados"]))
        _pdf_card(ax, .54, Y0, W, H, "Ítems con clave", str(tot["con_clave"]))
        _pdf_card(ax, .06, Y0-(H+GAP), W, H, "Aciertos", str(tot["aciertos"]) if tot["aciertos"] is not None else "—")
        _pdf_card(ax, .54, Y0-(H+GAP), W, H, "% Acierto", f"{pct:.1f}%")

        ax.text(.5,.58,"Resumen por Series", ha='center', fontsize=14, fontweight='bold')
        ylist = .54
        for _, r in bys.iterrows():
            s = r["serie"]; p = r["% acierto"]
            ax.text(.12, ylist, f"Serie {s}: {p:.1f}% acierto" if pd.notna(p) else f"Serie {s}: —", fontsize=11)
            ylist -= 0.03
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Detalle por ítem
        fig2 = plt.figure(figsize=(8.27,11.69)); ax2 = fig2.add_axes([0,0,1,1]); ax2.axis('off')
        ax2.text(.5,.96,"Detalle por ítem", ha='center', fontsize=16, fontweight='bold')

        def draw_block(df_block, y0):
            yy = y0
            for _, rr in df_block.iterrows():
                s = rr["serie"]; k = rr["index"]; a = rr["respuesta"]; c = rr["correcta"]; ok = rr["acierto"]
                t = f"{s}{k:02d}  —  resp: {a if pd.notna(a) else '—'} · ok: {c if pd.notna(c) else '—'} · {'✔' if ok else '✘' if ok is not None else ' '}"
                ax2.text(.08, yy, t, fontsize=10)
                yy -= 0.025
            return yy

        start = 0; ystart = .90
        while start < len(df):
            end = min(start+22, len(df))
            ystart = draw_block(df.iloc[start:end], ystart)
            start = end
            if start < len(df):
                pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
                fig2 = plt.figure(figsize=(8.27,11.69)); ax2 = fig2.add_axes([0,0,1,1]); ax2.axis('off'); ystart = .90
                ax2.text(.5,.96,"Detalle por ítem (cont.)", ha='center', fontsize=16, fontweight='bold')

        pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

    buf.seek(0)
    return buf.read()

def build_html_report(result: Dict, fecha: str) -> bytes:
    df = result["df"]; tot = result["totales"]; bys = result["series"]
    pct = tot["porcentaje"]
    rows = ""
    for _, r in df.iterrows():
        rows += f"<tr><td>{r['serie']}{r['index']:02d}</td><td>{r['respuesta'] if pd.notna(r['respuesta']) else '—'}</td><td>{r['correcta'] if pd.notna(r['correcta']) else '—'}</td><td>{'✔' if r['acierto'] else ('✘' if r['acierto'] is not None else ' ')}</td></tr>"
    srows = ""
    for _, r in bys.iterrows():
        val = (f"{r['% acierto']:.1f}%" if pd.notna(r['% acierto']) else "—")
        srows += f"<tr><td>{r['serie']}</td><td>{r['total']}</td><td>{r['contestados']}</td><td>{r['con_clave']}</td><td>{val}</td></tr>"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>Informe Raven</title>
<style>
body{{font-family:Inter,Arial; margin:24px; color:#111;}}
h1{{font-size:24px; margin:0 0 8px 0;}}
h3{{font-size:18px; margin:.2rem 0;}}
table{{border-collapse:collapse; width:100%; margin-top:8px}}
th,td{{border:1px solid #eee; padding:8px; text-align:left;}}
.kpi-grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}}
.kpi{{border:1px solid #eee; border-radius:12px; padding:12px; background:#fff;}}
.kpi .label{{font-size:13px; opacity:.85}}
.kpi .value{{font-size:22px; font-weight:800}}
@media print{{ .no-print{{display:none}} }}
</style>
</head><body>
<h1>Informe — Matrices Progresivas de Raven</h1>
<p>Fecha: <b>{fecha}</b></p>
<div class="kpi-grid">
  <div class="kpi"><div class="label">Ítems contestados</div><div class="value">{tot["contestados"]}</div></div>
  <div class="kpi"><div class="label">Ítems con clave</div><div class="value">{tot["con_clave"]}</div></div>
  <div class="kpi"><div class="label">Aciertos</div><div class="value">{tot["aciertos"] if tot["aciertos"] is not None else '—'}</div></div>
  <div class="kpi"><div class="label">% Acierto</div><div class="value">{f'{pct:.1f}%' if pct else '—'}</div></div>
</div>

<h3>Resumen por series</h3>
<table>
<thead><tr><th>Serie</th><th>Ítems</th><th>Contestados</th><th>Con clave</th><th>% acierto</th></tr></thead>
<tbody>{srows}</tbody>
</table>

<h3>Detalle por ítem</h3>
<table>
<thead><tr><th>Ítem</th><th>Respuesta</th><th>Correcta</th><th>OK</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<div class="no-print" style="margin-top:16px;">
  <button onclick="window.print()" style="padding:10px 14px; border:1px solid #ddd; background:#f9f9f9; border-radius:8px; cursor:pointer;">
    Imprimir / Guardar como PDF
  </button>
</div>
</body></html>"""
    return html.encode("utf-8")


# ---------------------------------------------------------------
# CALLBACK (auto-avance sin doble click)
# ---------------------------------------------------------------
def on_answer(item_id: int, value: int):
    st.session_state.answers[item_id] = value
    if st.session_state.q_idx < TOTAL_ITEMS - 1:
        st.session_state.q_idx += 1
    else:
        st.session_state.stage = "resultados"
        st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.session_state._needs_rerun = True


# ---------------------------------------------------------------
# VISTAS
# ---------------------------------------------------------------
def view_inicio():
    st.markdown(
        """
        <div class="card">
          <h1 style="margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;">
            🧩 Test de Matrices Progresivas de Raven — PRO
          </h1>
          <p class="small" style="margin:0;">Fondo blanco · Texto negro · Diseño profesional y responsivo</p>
        </div>
        """, unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">¿Qué evalúa?</h3>
              <p>Razonamiento analógico y detección de patrones. 60 ítems (Series A–F, 10 cada una).</p>
              <ul style="line-height:1.6">
                <li>Una figura por pantalla; eliges la alternativa; <b>auto-avance</b>.</li>
                <li>Resultados con KPIs, % por serie, curva de dificultad y reporte PDF/HTML.</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">Imágenes</h3>
              <p>Se cargarán directamente desde tu repo:</p>
              <code>legendss7/Test-Raven-Demo/main/raven_pagina_001.png ... 060.png</code>
              <p>No necesitas subir nada extra mientras mantengas nombres y ruta.</p>
            </div>
            """, unsafe_allow_html=True
        )

        if st.button("🚀 Iniciar evaluación", type="primary", use_container_width=True):
            st.session_state.stage = "test"
            st.session_state.q_idx = 0
            st.session_state.answers = {}
            st.session_state.fecha = None
            st.rerun()


def view_test():
    i = st.session_state.q_idx
    items = st.session_state.items
    it = items[i]
    s = it["series"]; k = it["index"]; iid = it["id"]
    n_opts = it["n_options"]

    st.progress((i+1)/TOTAL_ITEMS, text=f"Progreso: {i+1}/{TOTAL_ITEMS}")
    st.markdown(f"<div class='dim-title'>Serie {s} — Ítem {k:02d}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge'>Opciones: {n_opts}</div>", unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Imagen (carga perezosa desde URL)
        url = image_url_for_index(i)
        try:
            # st.image puede leer URL directamente (no descargamos toda la batería)
            st.markdown("<div class='q-image-wrap'>", unsafe_allow_html=True)
            st.image(url, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            st.error("No pude cargar la imagen desde GitHub. Verifica el nombre/ruta.")

        # Opciones 1..n_opts (auto-avance)
        st.markdown("<div class='options-grid'>", unsafe_allow_html=True)
        cols = st.columns(n_opts)
        prev = st.session_state.answers.get(iid, None)
        for opt in range(1, n_opts+1):
            with cols[opt-1]:
                pressed = st.button(f"{opt}", use_container_width=True, key=f"opt_{iid}_{opt}")
                if prev == opt:
                    st.markdown("<div class='opt selected'>Seleccionado</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='opt'> </div>", unsafe_allow_html=True)
                if pressed:
                    on_answer(iid, opt)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def view_resultados():
    res = compute_results(st.session_state.answers, st.session_state.items)
    df = res["df"]; tot = res["totales"]; bys = res["series"]
    pct = res["totales"]["porcentaje"]
    fecha = st.session_state.fecha

    st.markdown(
        f"""
        <div class="card">
          <h1 style="margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;">
            📊 Informe Raven — Resultados
          </h1>
          <p class="small" style="margin:0;">Fecha: <b>{fecha}</b></p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Ítems contestados</div><div class='value'>{tot['contestados']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Ítems con clave</div><div class='value'>{tot['con_clave']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Aciertos</div><div class='value'>{tot['aciertos'] if tot['aciertos'] is not None else '—'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>% Acierto</div><div class='value'>{f'{pct:.1f}%' if pct else '—'}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 % acierto por serie")
        st.plotly_chart(plot_series_bars(bys), use_container_width=True)
    with c2:
        st.subheader("📈 Curva de dificultad (proxy)")
        st.plotly_chart(plot_difficulty_curve(df), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Tabla de resultados por ítem")
    show = df.copy()
    show["ítem"] = show["serie"] + show["index"].apply(lambda x: f"{x:02d}")
    show["OK"] = show["acierto"].map(lambda x: "✔" if x is True else ("✘" if x is False else " "))
    show = show[["ítem","n_options","respuesta","correcta","OK"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔎 Errores (para revisión)")
    wrong = df[df["acierto"] == False].copy()
    if wrong.empty:
        st.success("¡Sin errores (o no hay clave de corrección)! 🎉")
    else:
        wrong["ítem"] = wrong["serie"] + wrong["index"].apply(lambda x: f"{x:02d}")
        wrong = wrong[["ítem","respuesta","correcta"]]
        st.dataframe(wrong, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    if HAS_MPL:
        pdf_bytes = build_pdf_report(res, fecha)
        st.download_button(
            "⬇️ Descargar PDF (servidor)",
            data=pdf_bytes,
            file_name="Informe_Raven_PRO.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        html_bytes = build_html_report(res, fecha)
        st.download_button(
            "⬇️ Descargar Reporte (HTML) — Imprime como PDF",
            data=html_bytes,
            file_name="Informe_Raven_PRO.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Para PDF directo, instala Matplotlib en el entorno.")

    st.markdown("---")
    if st.button("🔄 Nueva evaluación", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {}
        st.session_state.fecha = None
        st.rerun()


# ---------------------------------------------------------------
# FLUJO
# ---------------------------------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    view_resultados()

# Rerun único si lo marcó el callback
if st.session_state._needs_rerun:
    st.session_state._needs_rerun = False
    st.rerun()
