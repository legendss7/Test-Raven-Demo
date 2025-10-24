# raven_bigfive_ui_blanco.py
# =====================================================================================
# Test de Razonamiento Matricial (estilo Raven) — 60 ítems
# Estructura y DISEÑO basados en el código Big Five (UI blanco, tarjetas, KPIs, gauges)
# Flujo: stage = inicio → test → resultados | Autoavance al elegir alternativa
# Genera y CACHEA imágenes de los ítems/alternativas en ./assets/raven_items/
# Exporta Informe (PDF con matplotlib si está disponible, de lo contrario HTML imprimible)
# =====================================================================================

import os
import io
import json
import math
import time
import random
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import plotly.graph_objects as go

# Intento usar matplotlib para PDF (seguimos patrón del Big Five)
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch, Wedge, Circle
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -------------------------------------------------------------------------------------------------
# Config general de página (igual estética y estructura base que el Big Five de referencia)
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Raven • Evaluación Matricial (60 ítems)",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------------------------------------
# Estilos (copian el espíritu del Big Five: fondo blanco, tipografías, tarjetas, KPIs, pastel classes)
# -------------------------------------------------------------------------------------------------
st.markdown(
    """
<style>
[data-testid="stSidebar"] { display:none !important; }
html, body, [data-testid="stAppViewContainer"]{
  background:#ffffff !important; color:#111 !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{ max-width:1200px; padding-top:0.8rem; padding-bottom:2rem; }
.dim-title{ font-size:clamp(2.2rem, 5vw, 3.2rem); font-weight:900; letter-spacing:.2px; line-height:1.12; margin:.2rem 0 .6rem 0; }
.dim-desc{ margin:.1rem 0 1rem 0; opacity:.9; }
.card{ border:1px solid #eee; border-radius:14px; background:#fff; box-shadow: 0 2px 0 rgba(0,0,0,0.03); padding:18px; }
.kpi-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:12px; margin:10px 0 6px 0; }
.kpi{ border:1px solid #eee; border-radius:14px; background:#fff; padding:16px; position:relative; overflow:hidden; }
.kpi .label{ font-size:.95rem; opacity:.85; }
.kpi .value{ font-size:2.2rem; font-weight:900; line-height:1; }
.tag{ display:inline-block; padding:.2rem .6rem; border:1px solid #eee; border-radius:999px; font-size:.82rem; }
hr{ border:none; border-top:1px solid #eee; margin:16px 0; }

/* Tarjeta de análisis */
.dim-card{ border:1px solid #eee; border-radius:14px; overflow:hidden; background:#fff; }
.dim-card-header{ padding:14px 16px; display:flex; align-items:center; gap:10px; border-bottom:1px solid #eee; }
.dim-chip { font-weight:800; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(0,0,0,.06); background:#fff; }
.dim-title-row{ display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
.dim-title-name{ font-size:1.2rem; font-weight:800; margin:0; }
.dim-score{ font-size:1.1rem; font-weight:800; }
.dim-body{ padding:16px; }
.dim-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap:12px; }
.dim-section{ border:1px solid #eee; border-radius:12px; padding:12px; background:#fff; }
.dim-section h4{ margin:.2rem 0 .4rem 0; font-size:1rem; }
.small{ font-size:0.95rem; opacity:.9; }

/* Paleta pastel para bloques informativos */
.pastel-R { background:#F0F7FF; border-color:#E0EEFF; }
.pastel-M { background:#FFF6F2; border-color:#FFE7DE; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------------------------------
# Paths y fuentes
# -------------------------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.getcwd())
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "raven_items")
os.makedirs(ASSETS_DIR, exist_ok=True)

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def get_font(size=24):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()

# -------------------------------------------------------------------------------------------------
# Generación procedural de ítems (original; NO usa ítems oficiales del RPM de Raven)
# -------------------------------------------------------------------------------------------------
CANVAS_SIZE = 512
GRID = 3
CELL = CANVAS_SIZE // GRID
BORDER = 12
OPTION_SIZE = 140
SHAPES = ["circle", "square", "triangle", "diamond", "star"]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, size: int, fill, rotate_deg: int = 0):
    if shape == "circle":
        draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=fill)
    elif shape == "square":
        draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=fill)
    elif shape == "triangle":
        pts = [(cx, cy-size), (cx-size, cy+size), (cx+size, cy+size)]
        draw.polygon(pts, fill=fill)
    elif shape == "diamond":
        pts = [(cx, cy-size), (cx-size, cy), (cx, cy+size), (cx+size, cy)]
        draw.polygon(pts, fill=fill)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = i * math.pi/5 + math.radians(rotate_deg)
            r = size if i % 2 == 0 else size*0.5
            pts.append((cx + r*math.cos(ang), cy + r*math.sin(ang)))
        draw.polygon(pts, fill=fill)


def generate_panel(shape, count, size, rotation, shade):
    img = Image.new("RGB", (CELL - BORDER*2, CELL - BORDER*2), (245,246,248))
    d = ImageDraw.Draw(img)
    rng = random.Random(count*1000 + size + rotation + shade)
    for _ in range(count):
        cx = rng.randint(30, img.width-30)
        cy = rng.randint(30, img.height-30)
        sz = max(10, int(size*rng.uniform(0.85, 1.15)))
        col = (40+shade, 40+shade, 40+shade)
        rot = int(rotation + rng.randint(-10,10))
        draw_shape(d, shape, cx, cy, sz, fill=col, rotate_deg=rot)
    return img


def rule_progressions(seed):
    rng = random.Random(seed)
    shape_seq = rng.sample(SHAPES, 3)
    count_start = rng.randint(1, 3)
    size_start = rng.randint(14, 22)
    rotation_start = rng.choice([0, 15, 30, 45])
    shade_start = rng.randint(10, 120)
    count_step_row = rng.choice([1, 1, 2])
    size_step_col = rng.choice([2, 3, 4])
    rotation_step_col = rng.choice([15, 30])
    shade_step_row = rng.choice([10, 15, 20])
    grid_params = []
    for r in range(3):
        row = []
        for c in range(3):
            shape = shape_seq[r]
            count = count_start + r*count_step_row
            size = size_start + c*size_step_col
            rotation = (rotation_start + c*rotation_step_col) % 360
            shade = min(200, shade_start + r*shade_step_row)
            row.append((shape, count, size, rotation, shade))
        grid_params.append(row)
    return grid_params


def compose_matrix_image(params_grid, missing_pos=(2,2)):
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (230,232,236))
    for r in range(GRID):
        for c in range(GRID):
            x0, y0 = c*CELL, r*CELL
            block = Image.new("RGB", (CELL, CELL), (250,251,253))
            bd = ImageDraw.Draw(block)
            bd.rectangle([0,0,CELL-1,CELL-1], outline=(210,214,220), width=2)
            if (r,c) != missing_pos:
                shape, count, size, rotation, shade = params_grid[r][c]
                panel = generate_panel(shape, count, size, rotation, shade)
                block.paste(panel, (BORDER, BORDER))
            else:
                bd.text((CELL//2-10, CELL//2-20), "?", fill=(120,124,130), font=get_font(64))
            img.paste(block, (x0,y0))
    return img


def generate_distractors(correct_params, rng):
    variants = []
    shape, count, size, rotation, shade = correct_params
    for _ in range(7):
        v_shape = rng.choice([shape] + [s for s in SHAPES if s != shape])
        v_count = max(1, int(round(count + rng.choice([-1, 1, 0, 2, -2]))))
        v_size = max(8, int(round(size + rng.choice([-4, -2, 2, 4, 6]))))
        v_rotation = (rotation + rng.choice([-30,-15,0,15,30,45])) % 360
        v_shade = min(220, max(10, shade + rng.choice([-20,-10,0,10,20,30])))
        variants.append((v_shape, v_count, v_size, v_rotation, v_shade))
    return variants


def render_option_image(params):
    img = Image.new("RGB", (OPTION_SIZE, OPTION_SIZE), (245,246,248))
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,OPTION_SIZE-1,OPTION_SIZE-1], outline=(210,214,220), width=2)
    shape, count, size, rotation, shade = params
    inner = Image.new("RGB", (OPTION_SIZE-16, OPTION_SIZE-16), (245,246,248))
    di = ImageDraw.Draw(inner)
    rng = random.Random(sum([hash(x) for x in params]))
    for _ in range(count):
        cx = rng.randint(20, inner.width-20)
        cy = rng.randint(20, inner.height-20)
        sz = max(8, int(size*rng.uniform(0.85,1.15)))
        col = (40+shade, 40+shade, 40+shade)
        rot = int(rotation + rng.randint(-10,10))
        draw_shape(di, shape, cx, cy, sz, fill=col, rotate_deg=rot)
    img.paste(inner, (8,8))
    return img


def build_item_bank(total_items=60, master_seed=20251024):
    bank = []
    for idx in range(1, total_items+1):
        folder = os.path.join(ASSETS_DIR, f"item_{idx:02d}")
        os.makedirs(folder, exist_ok=True)
        meta_path = os.path.join(folder, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                bank.append(json.load(f))
            continue
        seed = master_seed + idx*777
        grid_params = rule_progressions(seed)
        missing = (2,2)
        stem = compose_matrix_image(grid_params, missing)
        correct_params = grid_params[missing[0]][missing[1]]
        rng_item = random.Random(seed*33)
        distractors = generate_distractors(correct_params, rng_item)
        options_params = distractors[:7]
        correct_index = rng_item.randint(0,7)
        options_params.insert(correct_index, correct_params)
        stem_path = os.path.join(folder, "stem.png")
        stem.save(stem_path)
        option_paths = []
        for j, p in enumerate(options_params):
            oimg = render_option_image(p)
            pth = os.path.join(folder, f"opt_{j}.png")
            oimg.save(pth)
            option_paths.append(pth)
        meta = {
            "id": idx,
            "folder": folder,
            "stem": stem_path,
            "options": option_paths,
            "correct": correct_index,
            "difficulty": round(0.2 + 0.8*(idx/total_items), 3)
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        bank.append(meta)
    return bank

# -------------------------------------------------------------------------------------------------
# Estado (mismo patrón de etapas del Big Five)
# -------------------------------------------------------------------------------------------------
if "stage" not in st.session_state: st.session_state.stage = "inicio"   # inicio | test | resultados
if "q_idx" not in st.session_state: st.session_state.q_idx = 0
if "answers" not in st.session_state: st.session_state.answers = {}
if "fecha" not in st.session_state: st.session_state.fecha = None
if "item_bank" not in st.session_state: st.session_state.item_bank = build_item_bank(60, master_seed=20251024)
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun = False
if "timer" not in st.session_state: st.session_state.timer = time.time()

# -------------------------------------------------------------------------------------------------
# Utilidades de resultados
# -------------------------------------------------------------------------------------------------

def compute_raven_scores(answers: dict):
    total = len(st.session_state.item_bank)
    correct = sum(1 for v in answers.values() if v and v.get("correct"))
    answered = len(answers)
    avg_rt = float(np.mean([v.get("rt",0.0) for v in answers.values()])) if answered else 0.0
    raw_pct = (correct/total) if total>0 else 0.0
    percentile_est = int(100 * (1/(1+math.exp(-10*(raw_pct-0.5)))))
    # buckets por dificultad (5 grupos) para gráficas
    diffs = [st.session_state.item_bank[i-1]["difficulty"] for i in range(1,total+1)]
    buckets = {k: {"n":0, "ok":0} for k in ["Muy baja","Baja","Media","Alta","Muy alta"]}
    for i in range(total):
        d = diffs[i]
        key = "Muy baja" if d<0.36 else ("Baja" if d<0.52 else ("Media" if d<0.68 else ("Alta" if d<0.84 else "Muy alta")))
        buckets[key]["n"] += 1
        if answers.get(i) and answers[i].get("correct"): buckets[key]["ok"] += 1
    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "raw_pct": raw_pct,
        "avg_rt": avg_rt,
        "percentile_est": percentile_est,
        "by_diff": buckets,
    }


def raven_narrative(scores: dict):
    p = scores["raw_pct"]
    if p>=0.85:
        lvl = "Muy alto"
        txt = (
            "Rendimiento excepcional en razonamiento abstracto y detección de patrones; alta flexibilidad cognitiva "
            "y control atencional. Suele aprender reglas nuevas con rapidez y transferirlas a contextos distintos."
        )
        rec = ["Desafíos de alta complejidad analítica", "Proyectos de modelamiento/algoritmos", "Mentoría técnica"]
    elif p>=0.70:
        lvl = "Alto"
        txt = (
            "Superior al promedio: discriminación sólida de reglas visuales, buena consistencia y adaptación. "
            "Se beneficia de tareas con carga analítica creciente."
        )
        rec = ["Ejercicios bajo tiempo moderado", "Tareas con reglas combinadas", "Análisis comparativo"]
    elif p>=0.50:
        lvl = "Medio"
        txt = (
            "Dentro del promedio: reconoce patrones frecuentes; puede mejorar en rapidez y verificación en ítems complejos."
        )
        rec = ["Rompecabezas visuales y series", "Chequeo sistemático fila/columna", "Práctica con feedback inmediato"]
    elif p>=0.30:
        lvl = "Medio-bajo"
        txt = (
            "Le cuesta estabilizar la regla cuando hay distractores sutiles o combinaciones múltiples; sugiere consolidar fundamentos."
        )
        rec = ["Revisión guiada de principios (cantidad, tamaño, giro, sombreado)", "Entrenamiento progresivo por bloques", "Pausas breves para manejo de estrés"]
    else:
        lvl = "Bajo"
        txt = (
            "Dificultades consistentes en la detección de patrones; posible interferencia por ansiedad/tiempo. "
            "Se recomienda práctica estructurada y ejemplos resueltos."
        )
        rec = ["Guías paso a paso", "Respiración 4-7-8 en transición entre ítems", "Repetición espaciada"]
    return lvl, txt, rec

# -------------------------------------------------------------------------------------------------
# Gráficos (gauge semicircular, barras de dificultad)
# -------------------------------------------------------------------------------------------------

def gauge_plotly(value: float, title: str = ""):
    v = max(0, min(100, float(value)))
    bounds = [0, 25, 40, 60, 75, 100]
    colors = ["#fde2e1", "#fff0c2", "#e9f2fb", "#e7f6e8", "#d9f2db"]
    vals = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    fig = go.Figure()
    fig.add_trace(go.Pie(values=vals, hole=0.6, rotation=180, direction="clockwise",
                         textinfo="none", marker=dict(colors=colors, line=dict(color="#ffffff", width=1)),
                         hoverinfo="skip", showlegend=False, sort=False))
    theta = (180 * (v/100.0))
    r = 0.95; x0, y0 = 0.5, 0.5
    xe = x0 + r*math.cos(math.radians(180 - theta))
    ye = y0 + r*math.sin(math.radians(180 - theta))
    fig.add_shape(type="line", x0=x0, y0=y0, x1=xe, y1=ye, line=dict(color="#6D597A", width=4))
    fig.add_shape(type="circle", x0=x0-0.02, y0=y0-0.02, x1=x0+0.02, y1=y0+0.02,
                  line=dict(color="#6D597A"), fillcolor="#6D597A")
    fig.update_layout(annotations=[
        dict(text=f"<b>{v:.1f}</b>", x=0.5, y=0.32, showarrow=False, font=dict(size=24, color="#111")),
        dict(text=title, x=0.5, y=0.16, showarrow=False, font=dict(size=13, color="#333"))],
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False, height=220, template="plotly_white")
    return fig


def plot_by_difficulty(scores: dict):
    data = scores["by_diff"]
    cats = ["Muy baja","Baja","Media","Alta","Muy alta"]
    vals = []
    for c in cats:
        n = data[c]["n"]; ok = data[c]["ok"]
        pct = 0 if n==0 else 100.0*ok/n
        vals.append(pct)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=vals, marker=dict(color=["#d9f2db","#e7f6e8","#e9f2fb","#fff0c2","#fde2e1"])) )
    fig.update_layout(height=380, template="plotly_white",
                      yaxis=dict(range=[0,100], title="% Acierto"), xaxis=dict(title="Dificultad"))
    return fig

# -------------------------------------------------------------------------------------------------
# Exportar reporte
# -------------------------------------------------------------------------------------------------

def pdf_semicircle(ax, value, cx=0.5, cy=0.5, r=0.45):
    v = max(0, min(100, float(value)))
    bands = [(0,25,"#fde2e1"), (25,40,"#fff0c2"), (40,60,"#e9f2fb"), (60,75,"#e7f6e8"), (75,100,"#d9f2db")]
    for a,b,c in bands:
        ang1 = 180*(a/100.0); ang2 = 180*(b/100.0)
        w = Wedge((cx,cy), r, 180-ang2, 180-ang1, facecolor=c, edgecolor="#fff", lw=1)
        ax.add_patch(w)
    theta = math.radians(180*(v/100.0))
    x2 = cx + r*0.95*math.cos(np.pi - theta)
    y2 = cy + r*0.95*math.sin(np.pi - theta)
    ax.plot([cx, x2], [cy, y2], color="#6D597A", lw=3)
    ax.add_patch(Circle((cx,cy), 0.02, color="#6D597A"))
    ax.text(cx, cy-0.12, f"{v:.1f}", ha="center", va="center", fontsize=16, color="#111")


def build_pdf(scores: dict, fecha: str, answers: dict) -> bytes:
    if not HAS_MPL:
        return None
    total = scores["total"]
    avg_rt = scores["avg_rt"]
    prec = scores["raw_pct"]*100
    perc = scores["percentile_est"]

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada con KPIs y medidores
        fig = plt.figure(figsize=(8.27,11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(.5,.95,"Informe – Test Matricial (estilo Raven)", ha='center', fontsize=20, fontweight='bold')
        ax.text(.5,.92,f"Fecha: {fecha}", ha='center', fontsize=11)

        def card(ax, x,y,w,h,title,val):
            r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               edgecolor="#dddddd", facecolor="#ffffff")
            ax.add_patch(r)
            ax.text(x+w*0.06, y+h*0.60, title, fontsize=10, color="#333")
            ax.text(x+w*0.06, y+h*0.25, f"{val}", fontsize=20, fontweight='bold')

        Y0 = .82; H = .10; W = .40; GAP = .02
        card(ax, .06, Y0, W, H, "Total ítems", f"{total}")
        card(ax, .54, Y0, W, H, "Precisión", f"{prec:.1f}%")
        card(ax, .06, Y0-(H+GAP), W, H, "Tiempo medio por ítem", f"{avg_rt:.1f} s")
        card(ax, .54, Y0-(H+GAP), W, H, "Percentil estimado", f"{perc}")

        axg1 = fig.add_axes([.18, .54, .22, .16]); axg1.axis('off'); pdf_semicircle(axg1, prec, 0.5, 0.0, 0.9); axg1.text(.5,-.35,"Precisión",ha="center",fontsize=10)
        axg2 = fig.add_axes([.39, .54, .22, .16]); axg2.axis('off'); pdf_semicircle(axg2, perc, 0.5, 0.0, 0.9); axg2.text(.5,-.35,"Percentil (ref.)",ha="center",fontsize=10)

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Tabla de respuestas
        fig2 = plt.figure(figsize=(8.27,11.69))
        a2 = fig2.add_subplot(111); a2.axis('off')
        a2.text(.5,.95, "Detalle de respuestas", ha='center', fontsize=16, fontweight='bold')
        y = .90
        a2.text(.08, y, "Ítem", fontsize=11); a2.text(.20, y, "Resp.", fontsize=11); a2.text(.32, y, "Correcto", fontsize=11); a2.text(.48, y, "RT (s)", fontsize=11)
        y -= .02
        for i in range(total):
            v = answers.get(i)
            ch = '-' if (v is None or v.get('choice') is None) else chr(ord('A') + v.get('choice'))
            corr = 'Sí' if (v and v.get('correct')) else 'No'
            rt = '-' if (v is None) else f"{v.get('rt',0):.1f}"
            a2.text(.08, y, f"{i+1:02d}", fontsize=10)
            a2.text(.20, y, f"{ch}", fontsize=10)
            a2.text(.32, y, f"{corr}", fontsize=10)
            a2.text(.48, y, f"{rt}", fontsize=10)
            y -= .02
            if y < .06:
                pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
                fig2 = plt.figure(figsize=(8.27,11.69)); a2 = fig2.add_subplot(111); a2.axis('off'); y=.95
        if plt.fignum_exists(fig2.number):
            pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

    buf.seek(0)
    return buf.read()


def build_html(scores: dict, fecha: str, answers: dict) -> bytes:
    total = scores["total"]; prec = scores["raw_pct"]*100; avg_rt = scores["avg_rt"]; perc = scores["percentile_est"]
    rows = ""
    for i in range(total):
        v = answers.get(i)
        ch = '-' if (v is None or v.get('choice') is None) else chr(ord('A') + v.get('choice'))
        corr = 'Sí' if (v and v.get('correct')) else 'No'
        rt = '-' if (v is None) else f"{v.get('rt',0):.1f}"
        rows += f"<tr><td>{i+1:02d}</td><td>{ch}</td><td>{corr}</td><td>{rt}</td></tr>"
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\" /><title>Informe Raven – Estilo</title>
<style>
body{{font-family:Inter,Arial; margin:24px; color:#111;}}
h1{{font-size:24px; margin:0 0 8px 0;}}
h3{{font-size:18px; margin:.2rem 0;}}
h4{{font-size:15px; margin:.2rem 0;}}
table{{border-collapse:collapse; width:100%; margin-top:8px}}
th,td{{border:1px solid #eee; padding:8px; text-align:left;}}
.tag{{display:inline-block; padding:.2rem .6rem; border:1px solid #eee; border-radius:999px; font-size:.82rem;}}
.kpi-grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}}
.kpi{{border:1px solid #eee; border-radius:12px; padding:12px; background:#fff;}}
.kpi .label{{font-size:13px; opacity:.85}}
.kpi .value{{font-size:22px; font-weight:800}}
@media print{{ .no-print{{display:none}} }}
</style></head>
<body>
<h1>Informe – Test Matricial (estilo Raven)</h1>
<p>Fecha: <b>{fecha}</b></p>
<div class=\"kpi-grid\">
  <div class=\"kpi\"><div class=\"label\">Total ítems</div><div class=\"value\">{total}</div></div>
  <div class=\"kpi\"><div class=\"label\">Precisión</div><div class=\"value\">{prec:.1f}%</div></div>
  <div class=\"kpi\"><div class=\"label\">Tiempo medio por ítem</div><div class=\"value\">{avg_rt:.1f} s</div></div>
  <div class=\"kpi\"><div class=\"label\">Percentil estimado (ref.)</div><div class=\"value\">{perc}</div></div>
</div>
<h3>Detalle de respuestas</h3>
<table><thead><tr><th>Ítem</th><th>Resp.</th><th>Correcto</th><th>RT (s)</th></tr></thead><tbody>
{rows}
</tbody></table>
<div class=\"no-print\" style=\"margin-top:16px;\"><button onclick=\"window.print()\" style=\"padding:10px 14px; border:1px solid #ddd; background:#f9f9f9; border-radius:8px; cursor:pointer;\">Imprimir / Guardar como PDF</button></div>
</body></html>"""
    return html.encode("utf-8")

# -------------------------------------------------------------------------------------------------
# Vistas (estructura igual a la referencia: view_inicio, view_test, view_resultados)
# -------------------------------------------------------------------------------------------------

def view_inicio():
    st.markdown(
        """
        <div class="card">
          <h1 style="margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;">🧩 Test Matricial (estilo Raven) — 60 ítems</h1>
          <p class="tag" style="margin:0;">UI blanco · Tarjetas · KPIs · Autoavance · Informe PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">¿Qué mide?</h3>
              <ul style="line-height:1.6">
                <li>Razonamiento abstracto y <b>detección de patrones</b> en matrices 3×3.</li>
                <li><b>60 ítems</b> con dificultad creciente (generados proceduralmente).</li>
                <li>Cada ítem tiene 8 alternativas (A–H).</li>
              </ul>
              <p class="small">Simulación inspirada en Raven; no reemplaza una evaluación psicométrica estandarizada.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">Cómo funciona</h3>
              <ol style="line-height:1.6">
                <li>Ves 1 pregunta por pantalla con la celda faltante.</li>
                <li>Elige la alternativa (A–H). <b>Al elegir, avanzas automáticamente</b>.</li>
                <li>Al final, resultados con KPIs, gauge, % de acierto por dificultad y descarga de informe.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Iniciar evaluación", type="primary", use_container_width=True):
            st.session_state.stage = "test"
            st.session_state.q_idx = 0
            st.session_state.answers = {}
            st.session_state.fecha = None
            st.session_state.timer = time.time()
            st.rerun()


def view_test():
    i = st.session_state.q_idx
    bank = st.session_state.item_bank
    total = len(bank)
    if i >= total:
        st.session_state.stage = "resultados"
        st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.rerun()
        return

    item = bank[i]
    p = (i+1)/total
    st.progress(p, text=f"Progreso: {i+1}/{total}")

    st.markdown(f"<div class='dim-title'>Matriz {i+1} de {total}</div>", unsafe_allow_html=True)
    st.markdown("<div class='dim-desc'>Selecciona la alternativa que completa la matriz (celda inferior derecha).</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.25,1])
    with col1:
        st.image(item["stem"], use_column_width=True, caption="Matriz 3×3")
    with col2:
        st.markdown("**Alternativas (A–H)**")
        labels = list("ABCDEFGH")
        cols = st.columns(2)
        clicks = []
        for j, pth in enumerate(item["options"]):
            with cols[j % 2]:
                st.image(pth, use_column_width=True)
                clicks.append(st.button(f"Elegir {labels[j]}", key=f"btn_{i}_{j}", use_container_width=True))
    st.markdown("</div>", unsafe_allow_html=True)

    # Registrar y auto-avanzar
    for j, clk in enumerate(clicks):
        if clk:
            rt = time.time() - st.session_state.timer
            is_ok = (j == item["correct"])\

            st.session_state.answers[i] = {
                "choice": j,
                "correct": bool(is_ok),
                "rt": round(rt,3),
                "difficulty": item.get("difficulty", 0.5),
            }
            st.session_state.q_idx = i + 1
            st.session_state.timer = time.time()
            st.rerun()
            return

    # Controles inferiores
    cA, cB = st.columns(2)
    with cA:
        if st.button("⏭️ Omitir", use_container_width=True):
            rt = time.time() - st.session_state.timer
            st.session_state.answers[i] = {"choice": None, "correct": False, "rt": round(rt,3), "difficulty": item.get("difficulty",0.5)}
            st.session_state.q_idx = i + 1
            st.session_state.timer = time.time()
            st.rerun()
    with cB:
        if st.button("⏹️ Finalizar ahora", use_container_width=True):
            st.session_state.stage = "resultados"
            st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
            st.rerun()


def view_resultados():
    scores = compute_raven_scores(st.session_state.answers)
    lvl, txt, recs = raven_narrative(scores)

    # Encabezado
    st.markdown(
        f"""
        <div class="card">
          <h1 style="margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;">📊 Informe – Test Matricial (estilo Raven)</h1>
          <p class="small" style="margin:0;">Fecha: <b>{st.session_state.fecha}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    prec = scores['raw_pct']*100
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Aciertos</div><div class='value'>{scores['correct']}/{scores['total']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Precisión</div><div class='value'>{prec:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Tiempo medio por ítem</div><div class='value'>{scores['avg_rt']:.1f} s</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Percentil estimado (ref.)</div><div class='value'>{scores['percentile_est']}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gauges
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Precisión")
        st.plotly_chart(gauge_plotly(prec, title="Precisión"), use_container_width=True)
    with c2:
        st.subheader("🏁 Percentil (referencial)")
        st.plotly_chart(gauge_plotly(scores['percentile_est'], title="Percentil (ref.)"), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 % de acierto por dificultad")
    st.plotly_chart(plot_by_difficulty(scores), use_container_width=True)

    st.markdown("---")
    st.subheader(f"📝 Interpretación: {lvl}")
    with st.container():
        st.markdown("<div class='dim-card'>", unsafe_allow_html=True)
        st.markdown("<div class='dim-card-header pastel-R'><div class='dim-chip'>🧠 Razonamiento</div><div class='dim-title-row' style='flex:1;'><h3 class='dim-title-name' style='margin:0;'>Perfil global</h3><div class='tag'>"+lvl+"</div></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='dim-body'><div class='dim-grid'><div class='dim-section'><h4>Resumen</h4><p class='small'>"+txt+"</p></div><div class='dim-section'><h4>Recomendaciones</h4>"+"".join([f"<p class='small'>• {r}</p>" for r in recs])+"</div></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    if HAS_MPL:
        pdf_bytes = build_pdf(scores, st.session_state.fecha, st.session_state.answers)
        st.download_button(
            "⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="Informe_Raven.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    else:
        html_bytes = build_html(scores, st.session_state.fecha, st.session_state.answers)
        st.download_button(
            "⬇️ Descargar Reporte (HTML) — Imprime como PDF",
            data=html_bytes,
            file_name="Informe_Raven.html",
            mime="text/html",
            use_container_width=True,
            type="primary",
        )
        st.caption("Instala matplotlib para obtener el PDF directo con gauges: `pip install matplotlib`.")

    st.markdown("---")
    if st.button("🔄 Nueva evaluación", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {}
        st.session_state.fecha = None
        st.session_state.timer = time.time()
        st.rerun()

# -------------------------------------------------------------------------------------------------
# FLUJO PRINCIPAL (igual patrón que el Big Five de referencia)
# -------------------------------------------------------------------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    if st.session_state.fecha is None:
        st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    view_resultados()

# Rerun final (si lo llegáramos a usar con radio; aquí se usa con botones, pero mantenemos patrón)
if st.session_state._needs_rerun:
    st.session_state._needs_rerun = False
    st.rerun()
