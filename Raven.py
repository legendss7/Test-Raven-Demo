# ======================================================================
#  Raven PRO — 60 ítems, render diferido y sin bloqueos
#  - Lazy generation: cada ítem se construye al mostrarse
#  - Render de imágenes al vuelo con PIL (sin cache pesado)
#  - Auto-avance al seleccionar, sin doble click
#  - KPIs + tabla de resultados, export HTML imprimible
# ======================================================================

import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from io import BytesIO
from datetime import datetime
import random

from PIL import Image, ImageDraw   # Ligero y estable en Cloud

# ----------------------------------------
# Config de página
# ----------------------------------------
st.set_page_config(
    page_title="Raven PRO | Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------
# Parámetros clave
# ----------------------------------------
N_ITEMS = 60         # ⇦ 60 ítems
SEED    = 2025       # semilla determinística para reproducibilidad
IMG_MTX = (500, 500) # tamaño de la matriz (suficiente nítido / rápido)
IMG_OPT = (150, 150) # tamaño de opción

# ----------------------------------------
# Estilos UI
# ----------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] { display:none !important; }
html, body, [data-testid="stAppViewContainer"]{
  background:#fff !important; color:#111 !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{ max-width:1200px; padding-top:0.8rem; padding-bottom:2rem; }
.card{
  border:1px solid #eee; border-radius:14px; background:#fff;
  box-shadow:0 2px 0 rgba(0,0,0,.03); padding:18px;
}
.big-title{
  font-size:clamp(2.1rem,4.5vw,3rem); font-weight:900; margin:.2rem 0 .6rem 0;
  animation: slideIn .3s ease-out both;
}
@keyframes slideIn{ from{ transform:translateY(6px); opacity:0;} to{ transform:translateY(0); opacity:1;} }

.kpi-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}
.kpi{ border:1px solid #eee; border-radius:14px; background:#fff; padding:16px; position:relative; overflow:hidden;}
.kpi .label{ font-size:.95rem; opacity:.85;}
.kpi .value{ font-size:2rem; font-weight:900; line-height:1;}

.choice{
  border:1px solid #eee; border-radius:12px; padding:10px; background:#fff; text-align:center;
}
.choice .num{ font-size:.85rem; opacity:.8; }
.choice img{ border-radius:8px; border:1px solid #eee; }

.small{ font-size:.95rem; opacity:.9; }
hr{ border:none; border-top:1px solid #eee; margin:14px 0; }
.badge{ display:inline-flex; align-items:center; gap:6px; padding:.25rem .55rem; font-size:.82rem; border-radius:999px; border:1px solid #eaeaea; background:#fafafa;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Modelo de ítem
# ----------------------------------------
@dataclass
class RavenItem:
    idx: int
    rule: str
    difficulty: str
    correct_tuple: Tuple[str, float, float]     # (shape, size_rel, rot_rad)
    options: List[Tuple[str, float, float]]     # 8 alternativas
    cells: List[Dict]                           # 9 celdas (la [8] es “faltante”)

# ----------------------------------------
# Utilidades de dibujo
# ----------------------------------------
SHAPES = ["square","circle","triangle","pentagon"]

def polygon_points(cx, cy, radius, sides, rotation_rad):
    pts=[]
    for k in range(sides):
        ang = rotation_rad + 2*np.pi*k/sides
        x = cx + radius*np.cos(ang)
        y = cy + radius*np.sin(ang)
        pts.append((x,y))
    return pts

def draw_shape(d: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, size_px: int, rot_rad: float, color=(17,17,17)):
    if shape == "circle":
        r = size_px//2
        d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=2)
    elif shape == "square":
        pts = polygon_points(cx, cy, size_px/2, 4, rot_rad)
        d.polygon(pts, outline=color, width=2)
    elif shape == "triangle":
        pts = polygon_points(cx, cy, size_px/2, 3, rot_rad)
        d.polygon(pts, outline=color, width=2)
    else:  # pentagon
        pts = polygon_points(cx, cy, size_px/2, 5, rot_rad)
        d.polygon(pts, outline=color, width=2)

def grid_centers(w, h):
    x0, y0, W, H = int(w*0.05), int(h*0.05), int(w*0.90), int(h*0.90)
    xs = [x0 + int(W/6), x0 + int(W/2), x0 + int(5*W/6)]
    ys = [y0 + int(H/6), y0 + int(H/2), y0 + int(5*H/6)]
    coords=[]
    for r in range(3):
        for c in range(3):
            coords.append((xs[c], ys[r]))
    return (x0,y0,W,H), coords

def clamp(v, a, b): return max(a, min(b, v))

# ----------------------------------------
# Reglas (ligeras)
# ----------------------------------------
def rule_rotation(seed:int):
    rng = random.Random(seed)
    rule, diff = "Rotación progresiva", "Media"
    shape    = rng.choice(SHAPES)
    base_rot = rng.choice([0, np.pi/6, np.pi/4])
    step_r   = rng.choice([np.pi/10, np.pi/8])
    step_c   = rng.choice([np.pi/12, np.pi/10])

    cells=[]
    for r in range(3):
        for c in range(3):
            rot = base_rot + r*step_r + c*step_c
            cells.append({"shape":shape,"size":0.32,"rot":rot})
    correct = cells[8]
    opts=[(correct["shape"], correct["size"], correct["rot"])]
    used={ (correct["shape"], round(correct["size"],3), round(float(correct["rot"])%6.283,4)) }
    for _ in range(7):
        delta = rng.choice([-1,1])*rng.choice([np.pi/12,np.pi/10,np.pi/8])
        rot2 = correct["rot"] + delta
        t=(shape, 0.32, round(float(rot2)%6.283,4))
        if t in used:
            rot2 += rng.choice([np.pi/16,-np.pi/16])
            t=(shape, 0.32, round(float(rot2)%6.283,4))
        used.add(t)
        opts.append((shape, 0.32, rot2))
    rng.shuffle(opts)
    return rule, diff, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def rule_size(seed:int):
    rng = random.Random(seed)
    rule, diff = "Tamaño progresivo", "Media"
    shape = rng.choice(SHAPES)
    base  = rng.choice([0.24,0.26,0.28])
    drow  = rng.choice([0.02,0.025])
    dcol  = rng.choice([0.02,0.025])

    cells=[]
    for r in range(3):
        for c in range(3):
            size = clamp(base + r*drow + c*dcol, 0.18, 0.42)
            cells.append({"shape":shape,"size":size,"rot":0.0})
    correct = cells[8]
    opts=[(correct["shape"], correct["size"], 0.0)]
    used={ (correct["shape"], round(correct["size"],3), 0.0) }
    for _ in range(7):
        delta = rng.choice([-1,1])*rng.choice([0.01,0.015,0.02])
        s2 = clamp(correct["size"]+delta, 0.16, 0.44)
        t=(shape, round(s2,3), 0.0)
        if t in used:
            s2 = clamp(s2 + rng.choice([0.005,-0.005]), 0.16, 0.44)
            t=(shape, round(s2,3), 0.0)
        used.add(t)
        opts.append((shape, s2, 0.0))
    rng.shuffle(opts)
    return rule, diff, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def rule_shape(seed:int):
    rng = random.Random(seed)
    rule, diff = "Cambio de forma", "Baja"
    shapes_pick = rng.sample(SHAPES, k=len(SHAPES))
    row_step = rng.choice([1,2]); col_step = rng.choice([1,3]); base_idx = rng.randint(0,3)
    cells=[]
    for r in range(3):
        for c in range(3):
            idx = (base_idx + r*row_step + c*col_step) % len(SHAPES)
            shape = shapes_pick[idx]
            cells.append({"shape":shape,"size":0.32,"rot":0.0})
    correct = cells[8]
    opts=[(correct["shape"], 0.32, 0.0)]
    used={ (correct["shape"], 0.32, 0.0) }
    for _ in range(7):
        s2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
        while (s2, 0.32, 0.0) in used:
            s2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
        used.add((s2,0.32,0.0)); opts.append((s2, 0.32, 0.0))
    rng.shuffle(opts)
    return rule, diff, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def rule_mix(seed:int):
    rng = random.Random(seed)
    rule, diff = "Mezcla forma+rotación", "Alta"
    base_idx = rng.randint(0,3); row_step = rng.choice([1,2]); col_rot_step = rng.choice([np.pi/12, np.pi/8])
    cells=[]
    for r in range(3):
        for c in range(3):
            shape = SHAPES[(base_idx + r*row_step) % len(SHAPES)]
            rot   = c*col_rot_step
            cells.append({"shape":shape,"size":0.32,"rot":rot})
    correct = cells[8]
    opts=[(correct["shape"], 0.32, correct["rot"])]
    used={ (correct["shape"], 0.32, round(float(correct["rot"])%6.283,4)) }
    for _ in range(7):
        if rng.random()<0.5:
            s2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
            t=(s2, 0.32, round(float(correct["rot"])%6.283,4))
            if t in used:
                s2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
                t=(s2, 0.32, round(float(correct["rot"])%6.283,4))
            used.add(t); opts.append((s2, 0.32, correct["rot"]))
        else:
            delta = rng.choice([np.pi/12,-np.pi/12,np.pi/8,-np.pi/8])
            rot2 = correct["rot"] + delta
            t=(correct["shape"],0.32,round(float(rot2)%6.283,4))
            if t in used:
                rot2 += rng.choice([np.pi/16,-np.pi/16])
                t=(correct["shape"],0.32,round(float(rot2)%6.283,4))
            used.add(t); opts.append((correct["shape"], 0.32, rot2))
    rng.shuffle(opts)
    return rule, diff, cells, (correct["shape"], correct["size"], correct["rot"]), opts

RULES = [rule_rotation, rule_size, rule_shape, rule_mix]

# ----------------------------------------
# Construcción perezosa de un ítem (determinístico por índice)
# ----------------------------------------
def get_item(idx:int, seed:int=SEED)->RavenItem:
    rng = random.Random(seed + idx*97)
    rule_fn = rng.choice(RULES)
    rule, diff, cells, correct, opts = rule_fn(seed + idx*137)
    return RavenItem(idx=idx, rule=rule, difficulty=diff, correct_tuple=correct, options=opts, cells=cells)

# ----------------------------------------
# Render de imágenes (sin cache pesado)
# ----------------------------------------
def render_matrix_png(cells:List[Dict], size=(500,500)) -> bytes:
    w,h = size
    img = Image.new("RGB", (w,h), (255,255,255))
    d = ImageDraw.Draw(img)
    x0,y0,W,H = int(w*0.05), int(h*0.05), int(w*0.90), int(h*0.90)
    d.rectangle([x0,y0,x0+W,y0+H], outline=(17,17,17), width=3)
    # líneas de la grilla
    for frac in [1/3, 2/3]:
        y = y0 + int(H*frac); x = x0 + int(W*frac)
        d.line([x0, y, x0+W, y], fill=(190,190,190), width=1)
        d.line([x, y0, x, y0+H], fill=(190,190,190), width=1)
    # centros
    xs = [x0 + int(W/6), x0 + int(W/2), x0 + int(5*W/6)]
    ys = [y0 + int(H/6), y0 + int(H/2), y0 + int(5*H/6)]
    centers=[(xs[c], ys[r]) for r in range(3) for c in range(3)]
    # dibujar celdas (la última es faltante, dibujamos caja punteada)
    for k, cell in enumerate(cells):
        cx, cy = centers[k]
        if k == 8:
            dash = 8; r = int(min(W,H)*0.12)
            for xx in range(cx-r, cx+r, dash*2):
                d.line([xx, cy-r, min(xx+dash, cx+r), cy-r], fill=(160,160,160), width=2)
                d.line([xx, cy+r, min(xx+dash, cx+r), cy+r], fill=(160,160,160), width=2)
            for yy in range(cy-r, cy+r, dash*2):
                d.line([cx-r, yy, cx-r, min(yy+dash, cy+r)], fill=(160,160,160), width=2)
                d.line([cx+r, yy, cx+r, min(yy+dash, cy+r)], fill=(160,160,160), width=2)
            continue
        shape, size_rel, rot = cell["shape"], cell["size"], cell["rot"]
        size_px = int(min(W,H) * size_rel)
        draw_shape(d, shape, cx, cy, size_px, rot)
    buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return buf.read()

def render_option_png(shape:str, size_rel:float, rot:float, size=(150,150)) -> bytes:
    w,h = size
    img = Image.new("RGB", (w,h), (255,255,255))
    d = ImageDraw.Draw(img)
    x0,y0,W,H = int(w*0.05), int(h*0.05), int(w*0.90), int(h*0.90)
    d.rectangle([x0,y0,x0+W,y0+H], outline=(210,210,210), width=2)
    cx, cy = w//2, h//2
    size_px = int(min(W,H) * size_rel)
    draw_shape(d, shape, cx, cy, size_px, rot)
    buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return buf.read()

# ----------------------------------------
# Estado
# ----------------------------------------
if "stage" not in st.session_state: st.session_state.stage="inicio"  # inicio | test | resultados
if "seed"  not in st.session_state: st.session_state.seed=SEED
if "q_idx" not in st.session_state: st.session_state.q_idx=0
if "answers" not in st.session_state: st.session_state.answers={}     # idx -> 0..7
if "start_time" not in st.session_state: st.session_state.start_time=None
if "end_time"   not in st.session_state: st.session_state.end_time=None
if "fecha"      not in st.session_state: st.session_state.fecha=None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun=False

# ----------------------------------------
# Scoring
# ----------------------------------------
def compute_result(n_items:int, answers:Dict[int,int])->Dict:
    raw=0
    rule_stats: Dict[str,Dict[str,int]] = {}
    diff_stats = {"Baja":{"ok":0,"tot":0}, "Media":{"ok":0,"tot":0}, "Alta":{"ok":0,"tot":0}}
    for i in range(n_items):
        it = get_item(i, st.session_state.seed)
        ans = answers.get(i, None)
        ok  = (ans is not None) and (it.options[ans]==it.correct_tuple)
        if ok: raw += 1
        rule_stats.setdefault(it.rule, {"ok":0,"tot":0})
        rule_stats[it.rule]["tot"] += 1
        if ok: rule_stats[it.rule]["ok"] += 1
        diff_stats[it.difficulty]["tot"] += 1
        if ok: diff_stats[it.difficulty]["ok"] += 1

    pct = raw/n_items*100 if n_items>0 else 0.0
    # percentil simple aproximado
    if raw <= 15: perc = 10
    elif raw <= 22: perc = 20
    elif raw <= 30: perc = 35
    elif raw <= 38: perc = 50
    elif raw <= 46: perc = 70
    elif raw <= 54: perc = 85
    else: perc = 95

    if st.session_state.start_time and st.session_state.end_time:
        secs = int((st.session_state.end_time - st.session_state.start_time).total_seconds())
    else:
        secs = 0

    return {"raw":raw, "pct":round(pct,1), "percentil":perc, "rule_stats":rule_stats, "diff_stats":diff_stats, "secs":secs}

# ----------------------------------------
# Callbacks
# ----------------------------------------
def on_pick(i:int):
    choice = st.session_state.get(f"resp_{i}")
    if choice is None: return
    st.session_state.answers[i] = int(choice)
    if i < N_ITEMS-1:
        st.session_state.q_idx = i+1
        st.session_state.stage = "test"
    else:
        st.session_state.stage = "resultados"
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")
    st.session_state._needs_rerun = True

# ----------------------------------------
# Export HTML (imprimible como PDF)
# ----------------------------------------
def export_html(n_items:int, answers:Dict[int,int], result:Dict)->bytes:
    rows=""
    for i in range(n_items):
        it = get_item(i, st.session_state.seed)
        ans = answers.get(i, None)
        ok  = (ans is not None) and (it.options[ans]==it.correct_tuple)
        rows += f"<tr><td>{i+1:02d}</td><td>{it.rule}</td><td>{it.difficulty}</td><td>{'✓' if ok else '✗'}</td></tr>"
    secs=result["secs"]; m,s=divmod(secs,60)
    html=f"""<!doctype html><html><head><meta charset="utf-8" />
<title>Informe Raven (HTML)</title>
<style>
body{{font-family:Inter,Arial; margin:24px; color:#111;}}
h1{{font-size:24px; margin:0 0 8px 0;}}
h3{{font-size:18px; margin:.8rem 0 .2rem 0;}}
table{{border-collapse:collapse; width:100%; margin-top:8px}}
th,td{{border:1px solid #eee; padding:8px; text-align:left;}}
.kpi-grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}}
.kpi{{border:1px solid #eee; border-radius:12px; padding:12px; background:#fff;}}
.kpi .label{{font-size:13px; opacity:.85}} .kpi .value{{font-size:22px; font-weight:800}}
@media print{{ .no-print{{display:none}} }}
</style></head><body>
<h1>Informe Raven — Matrices Progresivas</h1>
<p>Fecha: <b>{st.session_state.fecha}</b></p>
<div class="kpi-grid">
  <div class="kpi"><div class="label">Ítems correctos</div><div class="value">{result['raw']} / {n_items}</div></div>
  <div class="kpi"><div class="label">% Acierto</div><div class="value">{result['pct']:.1f}%</div></div>
  <div class="kpi"><div class="label">Percentil (aprox.)</div><div class="value">{result['percentil']}</div></div>
  <div class="kpi"><div class="label">Tiempo total</div><div class="value">{m:02d}:{s:02d} min</div></div>
</div>
<h3>Resumen por ítem</h3>
<table><thead><tr><th>#</th><th>Regla</th><th>Dificultad</th><th>Resultado</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="no-print" style="margin-top:16px;">
  <button onclick="window.print()" style="padding:10px 14px; border:1px solid #ddd; background:#f9f9f9; border-radius:8px; cursor:pointer;">
    Imprimir / Guardar como PDF
  </button>
</div>
</body></html>"""
    return html.encode("utf-8")

# ----------------------------------------
# Vistas
# ----------------------------------------
def view_inicio():
    st.markdown("""
    <div class="card">
      <div class="big-title">🧩 Test Raven — Matrices Progresivas (PRO)</div>
      <p class="small" style="margin:0;">60 ítems · imágenes generadas al vuelo (lazy) · diseño liviano y responsivo</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown(f"""
        <div class="card">
          <h3 style="margin-top:0">¿Qué mide?</h3>
          <p>Razonamiento fluido: detectar reglas visuales en una matriz 3×3 con una casilla faltante.</p>
          <ul>
            <li><b>Número de ítems:</b> {N_ITEMS}</li>
            <li><b>Auto-avance:</b> al seleccionar una opción pasas al siguiente ítem.</li>
            <li><b>Resultados:</b> KPIs generales, exactitud por regla/dificultad, tabla por ítem, exportación HTML.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <h3 style="margin-top:0">Recomendaciones</h3>
          <ul>
            <li>Navegador actualizado (Chrome/Edge/Firefox).</li>
            <li>Conexión estable (las imágenes se dibujan en el momento).</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Iniciar prueba", type="primary", use_container_width=True):
            st.session_state.stage="test"
            st.session_state.q_idx=0
            st.session_state.answers={}
            st.session_state.start_time=datetime.now()
            st.session_state.end_time=None
            st.session_state.fecha=None
            st.rerun()

def view_test():
    i = st.session_state.q_idx
    it = get_item(i, st.session_state.seed)

    st.progress((i+1)/N_ITEMS, text=f"Progreso: {i+1}/{N_ITEMS}")
    st.markdown(f"""
    <div class="card">
      <h3 style="margin:.2rem 0">Ítem {i+1} de {N_ITEMS}</h3>
      <div class="small">Regla: <b>{it.rule}</b> · Dificultad: <b>{it.difficulty}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Matriz del ítem actual
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        q_png = render_matrix_png(it.cells, size=IMG_MTX)
        st.image(q_png, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Alternativas
    st.markdown("### Alternativas")
    cols = st.columns(4)
    for k, tup in enumerate(it.options):
        col = cols[k%4]
        with col:
            st.markdown("<div class='choice'>", unsafe_allow_html=True)
            op_png = render_option_png(*tup, size=IMG_OPT)
            st.image(op_png, use_container_width=True)
            st.markdown(f"<div class='num'>Opción {k+1}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Selección y auto-avance (sin índice por defecto para evitar selección automática)
    prev = st.session_state.answers.get(i, None)
    st.radio(
        "Selecciona tu respuesta",
        options=list(range(8)),
        index=prev if prev is not None else None,
        format_func=lambda x: f"Opción {x+1}",
        key=f"resp_{i}",
        horizontal=True,
        label_visibility="collapsed",
        on_change=on_pick,
        args=(i,)
    )

def view_resultados():
    if st.session_state.end_time is None:
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")

    result = compute_result(N_ITEMS, st.session_state.answers)

    st.markdown(f"""
    <div class="card">
      <div class="big-title">📊 Informe Raven — Resultados</div>
      <p class="small" style="margin:0;">Fecha: <b>{st.session_state.fecha}</b></p>
    </div>
    """, unsafe_allow_html=True)

    m,s = divmod(result["secs"],60)
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Ítems correctos</div><div class='value'>{result['raw']} / {N_ITEMS}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>% Acierto</div><div class='value'>{result['pct']:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Percentil (aprox.)</div><div class='value'>{result['percentil']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Tiempo total</div><div class='value'>{m:02d}:{s:02d} min</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Resumen por ítem")
    rows=[]
    for i in range(N_ITEMS):
        it = get_item(i, st.session_state.seed)
        ans = st.session_state.answers.get(i, None)
        ok  = (ans is not None) and (it.options[ans]==it.correct_tuple)
        rows.append({
            "Ítem": i+1,
            "Regla": it.rule,
            "Dificultad": it.difficulty,
            "Respuesta": f"Opción {ans+1}" if ans is not None else "—",
            "Resultado": "✓" if ok else "✗"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔎 Exactitud por tipo de regla y dificultad")
    rule_rows=[]
    for r,stt in result["rule_stats"].items():
        ok, tot = stt["ok"], stt["tot"]; acc = ok/tot*100 if tot>0 else 0
        rule_rows.append({"Regla": r, "Correctos": ok, "Total": tot, "Exactitud %": round(acc,1)})
    st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

    diff_rows=[]
    for d,stt in result["diff_stats"].items():
        ok, tot = stt["ok"], stt["tot"]; acc = ok/tot*100 if tot>0 else 0
        diff_rows.append({"Dificultad": d, "Correctos": ok, "Total": tot, "Exactitud %": round(acc,1)})
    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe (HTML imprimible)")
    html_bytes = export_html(N_ITEMS, st.session_state.answers, result)
    st.download_button(
        "⬇️ Descargar Reporte (HTML) — Imprime como PDF",
        data=html_bytes,
        file_name="Informe_Raven_PRO.html",
        mime="text/html",
        use_container_width=True
    )

    st.markdown("---")
    if st.button("🔄 Nueva prueba", type="primary", use_container_width=True):
        st.session_state.stage="inicio"
        st.session_state.q_idx=0
        st.session_state.answers={}
        st.session_state.start_time=None
        st.session_state.end_time=None
        st.session_state.fecha=None
        st.rerun()

# ----------------------------------------
# Flujo principal
# ----------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    view_resultados()

# Rerun (una sola vez) si un callback lo solicitó
if st.session_state._needs_rerun:
    st.session_state._needs_rerun=False
    st.rerun()
