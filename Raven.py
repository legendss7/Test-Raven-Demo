# ================================================================
# Raven Matrices PRO — 60 Ítems (estructura Big Five PRO)
# - Auto-avance, fondo blanco, tipografía negra, tarjetas
# - Imágenes desde PDF remoto (GitHub) o local, render on-demand
# - KPIs + Gráficos + Tabla de aciertos/errores + PDF/HTML reporte
# ================================================================
import os
import io
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import plotly.graph_objects as go

# ---------------------------------------------------------------
# Dependencias opcionales
# ---------------------------------------------------------------
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend no-GUI
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except Exception:
    HAS_MPL = False

HAS_FITZ = False
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

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

.small{ font-size:0.95rem; opacity:.9; }
hr{ border:none; border-top:1px solid #eee; margin:16px 0; }

/* Botones mejorados */
div[data-testid="stButton"] button {
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# PARÁMETROS DEL TEST
# ---------------------------------------------------------------
SERIES = ["A","B","C","D","E","F"]
ITEMS_PER_SERIES = 10
TOTAL_ITEMS = len(SERIES)*ITEMS_PER_SERIES  # 60

# Clave de corrección (EJEMPLO). Reemplaza por tu gabarito oficial si difiere.
ANSWER_KEY: Dict[str, Dict[int, int]] = {
    "A": {1:3, 2:2, 3:6, 4:5, 5:1, 6:4, 7:2, 8:6, 9:1, 10:3},
    "B": {1:2, 2:4, 3:6, 4:1, 5:5, 6:3, 7:6, 8:2, 9:4, 10:1},
    "C": {1:5, 2:6, 3:2, 4:8, 5:7, 6:3, 7:1, 8:4, 9:6, 10:5},
    "D": {1:8, 2:2, 3:7, 4:6, 5:3, 6:5, 7:4, 8:1, 9:2, 10:7},
    "E": {1:4, 2:7, 3:5, 4:2, 5:8, 6:1, 7:6, 8:3, 9:7, 10:2},
    "F": {1:6, 2:1, 3:8, 4:5, 5:4, 6:2, 7:3, 8:8, 9:1, 10:6},
}


# ---------------------------------------------------------------
# ESTADO
# ---------------------------------------------------------------
def init_session_state():
    """Inicializa todas las variables de session_state"""
    defaults = {
        "stage": "inicio",
        "q_idx": 0,
        "answers": {},
        "fecha": None,
        "pdf_url": "",
        "pdf_path": None,
        "pdf_pages": 0,
        "page_map": list(range(60)),
        "doc_ready": False,
        "items": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ---------------------------------------------------------------
# MANIFEST (60 ítems)
# ---------------------------------------------------------------
def build_manifest() -> List[Dict]:
    items = []
    for si, s in enumerate(SERIES):
        for k in range(1, ITEMS_PER_SERIES+1):
            item_id = si*ITEMS_PER_SERIES + k  # 1..60
            n_opts = 6 if s in ["A","B"] else 8  # común en muchas versiones
            items.append({
                "id": item_id,
                "series": s,
                "index": k,
                "n_options": n_opts,
                "correct": ANSWER_KEY.get(s, {}).get(k, None)
            })
    return items

if st.session_state.items is None:
    st.session_state.items = build_manifest()


# ---------------------------------------------------------------
# PDF: descarga (si URL), apertura y render on-demand
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_pdf(url: str) -> Optional[str]:
    if not (HAS_REQUESTS and url):
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest = "raven_source.pdf"
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
    except Exception as e:
        st.error(f"Error descargando PDF: {str(e)}")
        return None

def open_pdf(path: str) -> Optional[int]:
    """Abre el PDF con fitz y retorna cantidad de páginas."""
    if not (HAS_FITZ and path and os.path.exists(path)):
        return None
    try:
        doc = fitz.open(path)
        n = doc.page_count
        doc.close()
        return n
    except Exception as e:
        st.error(f"Error abriendo PDF: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def render_page_as_image(path: str, page_index: int, dpi: int = 150) -> Optional[bytes]:
    """
    Rasteriza 1 página a PNG bytes. Se usa en cada ítem (carga perezosa).
    """
    if not (HAS_FITZ and path and os.path.exists(path)):
        return None
    try:
        doc = fitz.open(path)
        if page_index < 0 or page_index >= doc.page_count:
            doc.close()
            return None
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        st.error(f"Error renderizando página: {str(e)}")
        return None


# ---------------------------------------------------------------
# CÁLCULOS RESULTADOS
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
    n_correct = df["acierto"].sum() if df["acierto"].notna().any() else 0
    pct = (n_correct / n_with_key * 100) if n_with_key > 0 else 0.0

    by_series = (
        df.groupby("serie")
          .agg(total=("item_id","count"),
               contestados=("respuesta", lambda x: x.notna().sum()),
               con_clave=("correcta", lambda x: x.notna().sum()),
               aciertos=("acierto", lambda x: x.sum(skipna=True)))
          .reset_index()
    )
    by_series["% acierto"] = np.where(
        by_series["con_clave"]>0, (by_series["aciertos"]/by_series["con_clave"])*100, 0.0
    )

    # proxy dificultad (posición global A1..F10)
    df["pos_global"] = (df["serie"].map({s:i for i,s in enumerate(SERIES)}))*ITEMS_PER_SERIES + df["index"]

    return {
        "df": df,
        "totales": {
            "contestados": int(n_answered),
            "con_clave": int(n_with_key),
            "aciertos": int(n_correct),
            "porcentaje": float(pct)
        },
        "series": by_series.sort_values("serie").reset_index(drop=True)
    }


# ---------------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------------
def plot_series_bars(by_series: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_series["serie"], 
        y=by_series["% acierto"],
        text=[f"{x:.1f}%" if pd.notna(x) else "-" for x in by_series["% acierto"]],
        textposition="outside",
        marker=dict(color=["#81B29A","#F2CC8F","#E07A5F","#9C6644","#6D597A","#84A59D"])
    ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis=dict(title="% acierto", range=[0,105]),
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
        line=dict(width=2, color="#E07A5F")
    ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        yaxis=dict(title="% acierto (rolling x5)", range=[0,105]),
        xaxis=dict(title="Progresión (A1 → F10)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# ---------------------------------------------------------------
# PDF/HTML Reporte
# ---------------------------------------------------------------
def _pdf_card(ax, x,y,w,h,title,val):
    r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       edgecolor="#dddddd", facecolor="#ffffff", linewidth=1)
    ax.add_patch(r)
    ax.text(x+w*0.06, y+h*0.60, title, fontsize=10, color="#333")
    ax.text(x+w*0.06, y+h*0.25, f"{val}", fontsize=20, fontweight='bold')

def build_pdf_report(result: Dict, fecha: str) -> bytes:
    df = result["df"]; tot = result["totales"]; bys = result["series"]
    pct = tot["porcentaje"]

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Página 1: KPIs
        fig = plt.figure(figsize=(8.27,11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        
        ax.text(.5,.94,"Informe — Matrices Progresivas de Raven", ha='center', fontsize=20, fontweight='bold')
        ax.text(.5,.91,f"Fecha: {fecha}", ha='center', fontsize=11)

        Y0 = .80; H = .10; W = .40; GAP = .02
        _pdf_card(ax, .06, Y0, W, H, "Ítems contestados", str(tot["contestados"]))
        _pdf_card(ax, .54, Y0, W, H, "Ítems con clave", str(tot["con_clave"]))
        _pdf_card(ax, .06, Y0-(H+GAP), W, H, "Aciertos", str(tot["aciertos"]))
        _pdf_card(ax, .54, Y0-(H+GAP), W, H, "% Acierto", f"{pct:.1f}%")

        ax.text(.5,.58,"Resumen por Series", ha='center', fontsize=14, fontweight='bold')
        ylist = .54
        for _, r in bys.iterrows():
            s = r["serie"]; p = r["% acierto"]
            ax.text(.12, ylist, f"Serie {s}: {p:.1f}% acierto", fontsize=11)
            ylist -= 0.03
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Página 2+: Detalle
        items_per_page = 22
        for start_idx in range(0, len(df), items_per_page):
            fig2 = plt.figure(figsize=(8.27,11.69))
            ax2 = fig2.add_axes([0,0,1,1]); ax2.axis('off')
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            
            title = "Detalle por ítem" if start_idx == 0 else "Detalle por ítem (cont.)"
            ax2.text(.5,.96, title, ha='center', fontsize=16, fontweight='bold')

            yy = .90
            end_idx = min(start_idx + items_per_page, len(df))
            for _, rr in df.iloc[start_idx:end_idx].iterrows():
                s = rr["serie"]; k = rr["index"]; a = rr["respuesta"]; c = rr["correcta"]; ok = rr["acierto"]
                status = '✔' if ok else ('✘' if ok is not None else ' ')
                t = f"{s}{k:02d}  —  resp: {a if pd.notna(a) else '—'} · ok: {c if pd.notna(c) else '—'} · {status}"
                ax2.text(.08, yy, t, fontsize=10)
                yy -= 0.025

            pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

    buf.seek(0)
    return buf.read()

def build_html_report(result: Dict, fecha: str) -> bytes:
    df = result["df"]; tot = result["totales"]; bys = result["series"]
    pct = tot["porcentaje"]
    
    rows = ""
    for _, r in df.iterrows():
        status = '✔' if r['acierto'] else ('✘' if r['acierto'] is not None else ' ')
        rows += f"<tr><td>{r['serie']}{r['index']:02d}</td><td>{r['respuesta'] if pd.notna(r['respuesta']) else '—'}</td><td>{r['correcta'] if pd.notna(r['correcta']) else '—'}</td><td>{status}</td></tr>"
    
    srows = ""
    for _, r in bys.iterrows():
        srows += f"<tr><td>{r['serie']}</td><td>{r['total']}</td><td>{r['contestados']}</td><td>{r['con_clave']}</td><td>{r['% acierto']:.1f}%</td></tr>"
    
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
  <div class="kpi"><div class="label">Aciertos</div><div class="value">{tot["aciertos"]}</div></div>
  <div class="kpi"><div class="label">% Acierto</div><div class="value">{pct:.1f}%</div></div>
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
                <li>Resultados con KPIs, % por serie, gráficos y PDF profesional.</li>
              </ul>
            </div>
            """, unsafe_allow_html=True
        )
    
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">Origen de las imágenes</h3>
              <p>Pega aquí la URL <b>RAW</b> de tu PDF en GitHub, o deja vacío para usar un PDF local llamado <code>raven_source.pdf</code>.</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        url_input = st.text_input(
            "URL RAW del PDF en GitHub (opcional)",
            value=st.session_state.pdf_url,
            placeholder="https://raw.githubusercontent.com/usuario/repo/main/archivo.pdf",
            key="url_input"
        )
        st.session_state.pdf_url = url_input

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔽 Cargar PDF", type="secondary", use_container_width=True):
                with st.spinner("Cargando PDF..."):
                    local_path = None
                    if st.session_state.pdf_url:
                        local_path = download_pdf(st.session_state.pdf_url)
                        if not local_path:
                            st.error("No pude descargar el PDF. Verifica el enlace RAW.")
                    else:
                        if os.path.exists("raven_source.pdf"):
                            local_path = "raven_source.pdf"
                        else:
                            st.warning("No se encontró 'raven_source.pdf' local.")
                    
                    if local_path:
                        pages = open_pdf(local_path)
                        if not pages:
                            st.error("No pude abrir el PDF.")
                        else:
                            st.success(f"✅ PDF listo. Páginas: {pages}")
                            st.session_state.pdf_path = local_path
                            st.session_state.pdf_pages = pages
                            st.session_state.page_map = [min(i, pages-1) for i in range(60)]
                            st.session_state.doc_ready = True
        
        with col2:
            if st.button("🚀 Iniciar Test", type="primary", use_container_width=True):
                if not st.session_state.doc_ready:
                    if not st.session_state.pdf_path and os.path.exists("raven_source.pdf"):
                        pages = open_pdf("raven_source.pdf")
                        if pages:
                            st.session_state.pdf_path = "raven_source.pdf"
                            st.session_state.pdf_pages = pages
                            st.session_state.page_map = [min(i, pages-1) for i in range(60)]
                            st.session_state.doc_ready = True
                    
                    if not st.session_state.doc_ready:
                        st.error("⚠️ Debes cargar un PDF primero.")
                        st.stop()
                
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
    st.markdown(f"<div class='badge'>📋 Opciones: {n_opts}</div>", unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Render on-demand
        page_index = st.session_state.page_map[i]
        if not HAS_FITZ:
            st.error("⚠️ PyMuPDF (fitz) no está instalado. Agrégalo a requirements.txt")
        elif not st.session_state.pdf_path:
            st.error("⚠️ No hay PDF cargado.")
        else:
            img_bytes = render_page_as_image(st.session_state.pdf_path, page_index, dpi=150)
            if img_bytes:
                st.markdown("<div class='q-image-wrap'>", unsafe_allow_html=True)
                st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No pude rasterizar esta página.")

        # Opciones
        st.markdown("### Selecciona tu respuesta:")
        cols = st.columns(n_opts)
        prev = st.session_state.answers.get(iid, None)
        
        for opt in range(1, n_opts+1):
            with cols[opt-1]:
                button_type = "primary" if prev == opt else "secondary"
                if st.button(f"**{opt}**", use_container_width=True, key=f"opt_{iid}_{opt}", type=button_type):
                    st.session_state.answers[iid] = opt
                    if st.session_state.q_idx < TOTAL_ITEMS - 1:
                        st.session_state.q_idx += 1
                    else:
                        st.session_state.stage = "resultados"
                        st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def view_resultados():
    res = compute_results(st.session_state.answers, st.session_state.items)
    df = res["df"]; tot = res["totales"]; bys = res["series"]
    pct = tot["porcentaje"]; fecha = st.session_state.fecha

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
    st.markdown(f"<div class='kpi'><div class='label'>Aciertos</div><div class='value'>{tot['aciertos']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>% Acierto</div><div class='value'>{pct:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 % acierto por serie")
        st.plotly_chart(plot_series_bars(bys), use_container_width=True)
    with c2:
        st.subheader("📈 Curva de dificultad")
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
        st.success("¡Sin errores! 🎉")
    else:
        wrong["ítem"] = wrong["serie"] + wrong["index"].apply(lambda x: f"{x:02d}")
        wrong = wrong[["ítem","respuesta","correcta"]]
        st.dataframe(wrong, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if HAS_MPL:
            pdf_bytes = build_pdf_report(res, fecha)
            st.download_button(
                "⬇️ Descargar PDF",
                data=pdf_bytes,
                file_name=f"Informe_Raven_{fecha.replace('/', '-').replace(':', '-')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("📦 Instala Matplotlib para exportar en PDF")
    
    with col2:
        html_bytes = build_html_report(res, fecha)
        st.download_button(
            "⬇️ Descargar HTML",
            data=html_bytes,
            file_name=f"Informe_Raven_{fecha.replace('/', '-').replace(':', '-')}.html",
            mime="text/html",
            use_container_width=True
        )

    st.markdown("---")
    if st.button("🔄 Nueva evaluación", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {}
        st.session_state.fecha = None
        st.rerun()


# ---------------------------------------------------------------
# FLUJO PRINCIPAL
# ---------------------------------------------------------------
def main():
    if st.session_state.stage == "inicio":
        view_inicio()
    elif st.session_state.stage == "test":
        view_test()
    else:
        view_resultados()

if __name__ == "__main__":
    main()
