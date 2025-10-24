# ================================================================
#  Raven PRO — Matrices Progresivas 
#  Diseño profesional 
# 
#
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from io import BytesIO
from datetime import datetime
import random

# Intento usar matplotlib para renderizar ítems y construir PDF
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, RegularPolygon
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Raven PRO | Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------
# Estilos (blanco y negro, tarjetas y KPIs)
# ---------------------------------------------------------------
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

/* Título dimensión / sección */
.big-title{
  font-size:clamp(2.2rem, 4.5vw, 3rem);
  font-weight:900; letter-spacing:.2px; line-height:1.12;
  margin:.2rem 0 .6rem 0;
  animation: slideIn .3s ease-out both;
}
@keyframes slideIn{
  from{ transform: translateY(6px); opacity:0; }
  to{ transform: translateY(0); opacity:1; }
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
  transform: translateX(-100%);
  animation: shimmer 2s ease-in-out 1;
}
@keyframes shimmer { to{ transform: translateX(100%);} }
.kpi .label{ font-size:.95rem; opacity:.85; }
.kpi .value{ font-size:2rem; font-weight:900; line-height:1; }

/* Botones */
button[kind="primary"], button[kind="secondary"]{ width:100%; }

/* Opciones */
.choice{
  border:1px solid #eee; border-radius:12px; padding:10px; background:#fff; text-align:center;
}
.choice .num{ font-size:.85rem; opacity:.8; }
.choice img{ border-radius:8px; border:1px solid #eee; }

/* Tabla */
[data-testid="stDataFrame"] div[role="grid"]{ font-size:0.95rem; }

/* Secciones de análisis */
.section{
  border:1px solid #eee; border-radius:14px; background:#fff; padding:18px;
}
.section h3{ margin-top:0; }

/* Badges */
.badge{
  display:inline-flex; align-items:center; gap:6px; padding:.25rem .55rem; font-size:.82rem;
  border-radius:999px; border:1px solid #eaeaea; background:#fafafa;
}

/* Pequeño tip */
.small{ font-size:0.95rem; opacity:.9; }

/* Línea */
hr{ border:none; border-top:1px solid #eee; margin:14px 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Definición de ítems (estructura en memoria)
# ---------------------------------------------------------------
@dataclass
class RavenItem:
    idx: int
    rule: str
    difficulty: str
    question_png: bytes                   # imagen PNG de la matriz (con casilla en blanco)
    options_png: List[bytes]              # lista de 8 opciones (PNG)
    correct_idx: int                      # 0..7
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------
# Utilidades de dibujo con Matplotlib
# ---------------------------------------------------------------
def _fig_to_png_bytes(fig) -> bytes:
    """Convierte una figura Matplotlib a bytes PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _draw_shape(ax, shape: str, xy: Tuple[float, float], size: float, rot: float, color: str = "#222"):
    """Dibuja una forma geométrica básica en (x,y)."""
    x, y = xy
    if shape == "square":
        s = size
        r = Rectangle((x - s/2, y - s/2), s, s, angle=np.degrees(rot),
                      linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(r)
    elif shape == "circle":
        c = Circle((x, y), radius=size/2, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(c)
    elif shape == "triangle":
        t = RegularPolygon((x, y), numVertices=3, radius=size/2, orientation=rot,
                           linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(t)
    elif shape == "pentagon":
        p = RegularPolygon((x, y), numVertices=5, radius=size/2, orientation=rot,
                           linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(p)
    else:
        # default square
        s = size
        r = Rectangle((x - s/2, y - s/2), s, s, angle=np.degrees(rot),
                      linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(r)


def _new_figure(w=600, h=600) -> Tuple[any, any]:
    """Crea figura cuadrada sin ejes."""
    fig = plt.figure(figsize=(w/96, h/96), dpi=96)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------
# Generación procedimental de matrices tipo Raven
# (Reglas sencillas pero variadas, 3x3 con casilla faltante [2,2])
# ---------------------------------------------------------------
SHAPES = ["square", "circle", "triangle", "pentagon"]
COLORS = ["#222"]  # monocromo (Raven real es B/N; mantenemos líneas negras)

def _grid_positions():
    """Devuelve coordenadas XY normalizadas para una grilla 3x3."""
    xs = [0.2, 0.5, 0.8]
    ys = [0.8, 0.5, 0.2]
    coords = []
    for r in range(3):
        for c in range(3):
            coords.append((xs[c], ys[r]))
    return coords  # index 0..8

def _render_matrix(cells: List[dict], hole_idx: int = 8, show_hole=True) -> bytes:
    """
    cells: lista de 9 dicts con {shape, size, rot}
    hole_idx: índice de la casilla vacía (0..8)
    """
    fig, ax = _new_figure(520, 520)
    # Dibujar marco de la grilla
    ax.add_patch(Rectangle((0.05, 0.05), 0.90, 0.90, fill=False, ec="#111", lw=1.5))
    # Líneas internas
    for i in [1/3, 2/3]:
        ax.plot([0.05, 0.95], [0.05+i*0.90, 0.05+i*0.90], color="#bbb", lw=.8)
        ax.plot([0.05+i*0.90, 0.05+i*0.90], [0.05, 0.95], color="#bbb", lw=.8)
    # Dibuja celdas
    coords = _grid_positions()
    for idx, cell in enumerate(cells):
        if idx == hole_idx and show_hole:
            # casilla vacía
            ax.add_patch(Rectangle((coords[idx][0]-0.12, coords[idx][1]-0.12), 0.24, 0.24, fill=False, ec="#999", lw=2, ls="--"))
            continue
        sh = cell["shape"]; size = cell["size"]; rot = cell["rot"]
        _draw_shape(ax, sh, coords[idx], size, rot, color="#111")
    return _fig_to_png_bytes(fig)


def _render_option(shape: str, size: float, rot: float) -> bytes:
    """Render de una opción en recuadro pequeño."""
    fig, ax = _new_figure(170, 170)
    ax.add_patch(Rectangle((0.05, 0.05), 0.90, 0.90, fill=False, ec="#ddd", lw=1.2))
    _draw_shape(ax, shape, (0.5, 0.5), size, rot, color="#111")
    return _fig_to_png_bytes(fig)


# -------------------------------
# Reglas (4 familias simples)
# -------------------------------
def rule_rotation(idx_seed: int) -> Tuple[List[dict], int, List[Tuple[str,float,float]]]:
    """
    Rotación progresiva por fila/col:
    shape constante; rotación aumenta por pasos.
    """
    rng = random.Random(idx_seed)
    shape = rng.choice(SHAPES)
    base_rot = rng.choice([0, np.pi/4, np.pi/6])
    step_row = rng.choice([np.pi/8, np.pi/6])
    step_col = rng.choice([np.pi/12, np.pi/8])

    cells = []
    for r in range(3):
        for c in range(3):
            rot = base_rot + r*step_row + c*step_col
            size = 0.20
            cells.append({"shape": shape, "size": size, "rot": rot})
    hole_idx = 8
    correct = cells[hole_idx]
    # generar opciones
    opts = []
    used = set()
    for k in range(8):
        if k == 0:
            opts.append((correct["shape"], correct["size"], correct["rot"]))
            used.add((correct["shape"], round(correct["size"], 3), round(float(correct["rot"])%6.283, 3)))
        else:
            # perturbar rot
            delta = rng.choice([-1,-1,1,1])*rng.choice([np.pi/12, np.pi/10, np.pi/8])
            rot2 = correct["rot"] + delta
            tup = (shape, 0.20, round(float(rot2)%6.283, 3))
            if tup in used: 
                rot2 += rng.choice([np.pi/16, -np.pi/16])
                tup = (shape, 0.20, round(float(rot2)%6.283, 3))
            used.add(tup)
            opts.append((shape, 0.20, rot2))
    rng.shuffle(opts)
    correct_idx = opts.index((correct["shape"], correct["size"], correct["rot"]))
    return cells, correct_idx, opts


def rule_size(idx_seed: int) -> Tuple[List[dict], int, List[Tuple[str,float,float]]]:
    """
    Tamaño progresivo: shape constante; tamaño crece por fila/col.
    """
    rng = random.Random(idx_seed)
    shape = rng.choice(SHAPES)
    base = rng.choice([0.12, 0.14])
    drow = rng.choice([0.02, 0.025])
    dcol = rng.choice([0.02, 0.025])

    cells = []
    for r in range(3):
        for c in range(3):
            size = base + r*drow + c*dcol
            rot = 0.0
            cells.append({"shape": shape, "size": size, "rot": rot})
    hole_idx = 8
    correct = cells[hole_idx]
    opts = []
    used = set()
    for k in range(8):
        if k == 0:
            opts.append((correct["shape"], correct["size"], correct["rot"]))
            used.add((correct["shape"], round(correct["size"],3), 0.0))
        else:
            # perturbar tamaño
            delta = rng.choice([-1,1])*rng.choice([0.01, 0.015, 0.02])
            size2 = max(0.08, min(0.28, correct["size"] + delta))
            tup = (shape, round(size2,3), 0.0)
            if tup in used:
                size2 = max(0.08, min(0.28, size2 + rng.choice([0.005, -0.005])))
                tup = (shape, round(size2,3), 0.0)
            used.add(tup)
            opts.append((shape, size2, 0.0))
    rng.shuffle(opts)
    correct_idx = opts.index((correct["shape"], correct["size"], correct["rot"]))
    return cells, correct_idx, opts


def rule_shape(idx_seed: int) -> Tuple[List[dict], int, List[Tuple[str,float,float]]]:
    """
    Cambio de forma por fila/col: patrón tipo suma modular en formas.
    """
    rng = random.Random(idx_seed)
    shapes_pick = rng.sample(SHAPES, k=len(SHAPES))
    # asignar forma por fila y col
    row_step = rng.choice([1, 2])  # mod 4
    col_step = rng.choice([1, 3])  # mod 4
    base_idx = rng.randint(0,3)
    cells = []
    for r in range(3):
        for c in range(3):
            idx = (base_idx + r*row_step + c*col_step) % len(SHAPES)
            shape = shapes_pick[idx]
            size = 0.20
            rot = 0.0
            cells.append({"shape": shape, "size": size, "rot": rot})
    hole_idx = 8
    correct = cells[hole_idx]
    opts = []
    used = set()
    for k in range(8):
        if k==0:
            opts.append((correct["shape"], correct["size"], correct["rot"]))
            used.add((correct["shape"], 0.20, 0.0))
        else:
            # otra forma distinta
            shape2 = rng.choice(SHAPES)
            while shape2 == correct["shape"] or (shape2, 0.20, 0.0) in used:
                shape2 = rng.choice(SHAPES)
            used.add((shape2, 0.20, 0.0))
            opts.append((shape2, 0.20, 0.0))
    rng.shuffle(opts)
    correct_idx = opts.index((correct["shape"], correct["size"], correct["rot"]))
    return cells, correct_idx, opts


def rule_mix(idx_seed: int) -> Tuple[List[dict], int, List[Tuple[str,float,float]]]:
    """
    Mezcla: forma cambia por fila; rotación por columna. Tamaño fijo.
    """
    rng = random.Random(idx_seed)
    base_idx = rng.randint(0,3)
    row_step = rng.choice([1, 2])
    col_rot_step = rng.choice([np.pi/12, np.pi/8])
    cells = []
    for r in range(3):
        for c in range(3):
            shape = SHAPES[(base_idx + r*row_step) % len(SHAPES)]
            rot = c*col_rot_step
            cells.append({"shape": shape, "size": 0.20, "rot": rot})
    hole_idx = 8
    correct = cells[hole_idx]
    opts = []
    used = set()
    for k in range(8):
        if k==0:
            opts.append((correct["shape"], 0.20, correct["rot"]))
            used.add((correct["shape"], 0.20, round(float(correct["rot"])%6.283,3)))
        else:
            # o cambio forma (misma rot) o ligera rotación
            if rng.random()<0.5:
                shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
                tup = (shape2, 0.20, round(float(correct["rot"])%6.283,3))
                if tup in used:
                    shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
                    tup = (shape2, 0.20, round(float(correct["rot"])%6.283,3))
                used.add(tup)
                opts.append((shape2, 0.20, correct["rot"]))
            else:
                delta = rng.choice([np.pi/12, -np.pi/12, np.pi/8, -np.pi/8])
                rot2 = correct["rot"] + delta
                tup = (correct["shape"], 0.20, round(float(rot2)%6.283,3))
                if tup in used:
                    rot2 += rng.choice([np.pi/16, -np.pi/16])
                    tup = (correct["shape"], 0.20, round(float(rot2)%6.283,3))
                used.add(tup)
                opts.append((correct["shape"], 0.20, rot2))
    rng.shuffle(opts)
    correct_idx = opts.index((correct["shape"], 0.20, correct["rot"]))
    return cells, correct_idx, opts


# Mapa de reglas a funciones y dificultad estimada
RULES = [
    ("Rotación progresiva", rule_rotation, "Media"),
    ("Tamaño progresivo", rule_size, "Media"),
    ("Cambio de forma", rule_shape, "Baja"),
    ("Mezcla forma+rotación", rule_mix, "Alta"),
]


# ---------------------------------------------------------------
# Generación del banco de ítems
# ---------------------------------------------------------------
def generate_item(idx: int, seed: int) -> RavenItem:
    rng = random.Random(seed + idx * 97)
    rule_name, fn, diff = rng.choice(RULES)
    cells, correct_idx, opts = fn(seed + idx * 137)
    q_png = _render_matrix(cells, hole_idx=8, show_hole=True) if HAS_MPL else b""
    options_png = []
    for (shape, size, rot) in opts:
        op_png = _render_option(shape, size, rot) if HAS_MPL else b""
        options_png.append(op_png)
    meta = {"seed": seed, "rule_name": rule_name}
    return RavenItem(
        idx=idx,
        rule=rule_name,
        difficulty=diff,
        question_png=q_png,
        options_png=options_png,
        correct_idx=correct_idx,
        meta=meta
    )


def generate_bank(seed: int, n_items: int = 36) -> List[RavenItem]:
    bank = []
    for i in range(n_items):
        bank.append(generate_item(i, seed))
    return bank


# ---------------------------------------------------------------
# Estado global y saneador (arregla TypeError/IndexError)
# ---------------------------------------------------------------
def ensure_bank(n_items: int = 36):
    # items
    if not isinstance(st.session_state.get("items"), list) or len(st.session_state.items) == 0:
        st.session_state.items = generate_bank(st.session_state.get("seed", 2025), n_items)

    # q_idx entero y dentro de rango
    q_idx = st.session_state.get("q_idx", 0)
    try:
        q_idx = int(q_idx)
    except Exception:
        q_idx = 0
    q_idx = max(0, min(q_idx, len(st.session_state.items) - 1))
    st.session_state.q_idx = q_idx


# ---------------------------------------------------------------
# Inicialización de estado
# ---------------------------------------------------------------
if "stage" not in st.session_state: st.session_state.stage = "inicio"   # inicio | test | resultados
if "seed" not in st.session_state: st.session_state.seed = 2025
if "items" not in st.session_state: st.session_state.items = []
if "q_idx" not in st.session_state: st.session_state.q_idx = 0
if "answers" not in st.session_state: st.session_state.answers = {}     # idx -> 0..7
if "start_time" not in st.session_state: st.session_state.start_time = None
if "end_time" not in st.session_state: st.session_state.end_time = None
if "fecha" not in st.session_state: st.session_state.fecha = None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun = False


# ---------------------------------------------------------------
# Lógica de corrección y métricas
# ---------------------------------------------------------------
def compute_result(items: List[RavenItem], answers: Dict[int, int]) -> Dict:
    total = len(items)
    correct_list = []
    rule_stats = {}  # por regla
    diff_stats = {"Baja": {"ok":0,"tot":0},
                  "Media":{"ok":0,"tot":0},
                  "Alta": {"ok":0,"tot":0}}

    for it in items:
        a = answers.get(it.idx, None)
        ok = (a == it.correct_idx)
        correct_list.append(1 if ok else 0)

        rule_stats.setdefault(it.rule, {"ok":0, "tot":0})
        rule_stats[it.rule]["tot"] += 1
        if ok: rule_stats[it.rule]["ok"] += 1

        diff_stats[it.difficulty]["tot"] += 1
        if ok: diff_stats[it.difficulty]["ok"] += 1

    raw = int(sum(correct_list))
    pct = (raw/total)*100 if total>0 else 0.0

    # Percentil (aprox.) – tabla simple de referencia ocupacional (no clínica)
    # Esto es meramente orientativo.
    if raw <= 10: perc = 10
    elif raw <= 14: perc = 20
    elif raw <= 18: perc = 30
    elif raw <= 22: perc = 40
    elif raw <= 26: perc = 55
    elif raw <= 30: perc = 70
    elif raw <= 32: perc = 85
    else: perc = 95

    # Tiempo
    if st.session_state.start_time and st.session_state.end_time:
        dt = st.session_state.end_time - st.session_state.start_time
        secs = int(dt.total_seconds())
    else:
        secs = 0

    return {
        "raw": raw,
        "pct": round(pct,1),
        "percentil": perc,
        "correct_vector": correct_list,
        "rule_stats": rule_stats,
        "diff_stats": diff_stats,
        "secs": secs
    }


# ---------------------------------------------------------------
# Callbacks seguros
# ---------------------------------------------------------------
def on_pick(idx: int, choice: int):
    ensure_bank(len(st.session_state.items))
    items = st.session_state.items
    idx = max(0, min(int(idx), len(items) - 1))
    st.session_state.answers[idx] = int(choice)

    # Avance seguro
    if idx < len(items) - 1:
        st.session_state.q_idx = idx + 1
        st.session_state.stage = "test"
    else:
        st.session_state.stage = "resultados"
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")

    st.session_state._needs_rerun = True


# ---------------------------------------------------------------
# Gráficos de resultados (Plotly opcional, usamos matplotlib para PDF)
# ---------------------------------------------------------------
def plot_rule_bars(rule_stats: Dict[str, Dict[str,int]]):
    if not HAS_MPL:
        return None
    rules = list(rule_stats.keys())
    ok = [rule_stats[r]["ok"] for r in rules]
    tot = [rule_stats[r]["tot"] for r in rules]
    acc = [ (ok[i]/tot[i]*100 if tot[i]>0 else 0) for i in range(len(rules)) ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=120)
    ypos = np.arange(len(rules))
    ax.barh(ypos, acc, color="#81B29A")
    ax.set_yticks(ypos)
    ax.set_yticklabels(rules, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exactitud (%)")
    ax.set_title("Exactitud por tipo de regla")
    for i, v in enumerate(acc):
        ax.text(v+1, i, f"{v:.1f}%", va='center', fontsize=8)
    fig.tight_layout()
    return fig


def plot_diff_bars(diff_stats: Dict[str, Dict[str,int]]):
    if not HAS_MPL:
        return None
    diffs = list(diff_stats.keys())
    ok = [diff_stats[d]["ok"] for d in diffs]
    tot = [diff_stats[d]["tot"] for d in diffs]
    acc = [ (ok[i]/tot[i]*100 if tot[i]>0 else 0) for i in range(len(diffs)) ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=120)
    ax.bar(diffs, acc, color="#E07A5F")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exactitud (%)")
    ax.set_title("Exactitud por dificultad")
    for i, v in enumerate(acc):
        ax.text(i, v+1, f"{v:.1f}%", ha='center', fontsize=8)
    fig.tight_layout()
    return fig


def plot_gauge(score: float, title: str = "Raven Score"):
    """Medidor semicircular sencillo con MPL."""
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=120, subplot_kw=dict(polar=True))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # fondo
    ax.bar(np.linspace(0, np.pi, 100), height=[100]*100, width=np.pi/100,
           color="#f1f1f1", edgecolor="#f1f1f1")
    # valor
    n = int(min(max(score, 0), 100))
    ax.bar(np.linspace(0, np.pi*(n/100), n), height=[100]*n, width=np.pi/100,
           color="#6D597A", edgecolor="#6D597A")
    # texto
    ax.text(np.pi/2, 60, f"{score:.1f}", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# Exportación: PDF (MPL) o HTML (fallback)
# ---------------------------------------------------------------
def build_pdf(items: List[RavenItem], answers: Dict[int,int], result: Dict) -> bytes:
    if not HAS_MPL:
        return b""

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada
        fig = plt.figure(figsize=(8.27,11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(.5,.95,"Informe Raven — Matrices Progresivas (Ocupacional)", ha='center', fontsize=20, fontweight='bold')
        ax.text(.5,.92,f"Fecha: {st.session_state.fecha}", ha='center', fontsize=11)

        # KPIs
        raw = result["raw"]; pct = result["pct"]; perc = result["percentil"]; secs = result["secs"]
        m, s = divmod(secs, 60)
        meta = [
            ("Ítems correctos", f"{raw} / {len(items)}"),
            ("% Acierto", f"{pct:.1f}%"),
            ("Percentil (aprox.)", f"{perc}"),
            ("Tiempo total", f"{m:02d}:{s:02d} min")
        ]
        y = .83
        for t, v in meta:
            ax.text(.15, y, f"{t}", fontsize=12)
            ax.text(.65, y, f"{v}", fontsize=14, fontweight="bold")
            y -= .05

        # Gauge
        g = plot_gauge(pct, "Exactitud general (0–100)")
        if g:
            fig_g = g
            # Insertar como imagen en la portada
            img_buf = BytesIO()
            fig_g.savefig(img_buf, format="png", dpi=140, bbox_inches="tight")
            plt.close(fig_g)
            img_buf.seek(0)
            arr = plt.imread(img_buf)
            ax.imshow(arr, extent=(.1, .9, .4, .7))
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Barras por regla
        f1 = plot_rule_bars(result["rule_stats"])
        if f1:
            pdf.savefig(f1, bbox_inches='tight'); plt.close(f1)

        # Barras por dificultad
        f2 = plot_diff_bars(result["diff_stats"])
        if f2:
            pdf.savefig(f2, bbox_inches='tight'); plt.close(f2)

        # Resumen tabla
        fig3 = plt.figure(figsize=(8.27,11.69))
        ax3 = fig3.add_axes([0,0,1,1]); ax3.axis('off')
        ax3.text(.5,.95,"Resumen por ítem", ha='center', fontsize=16, fontweight='bold')
        ax3.text(.08,.92,"(✓ correcto · ✗ incorrecto)", fontsize=10)

        # Lista de 36 en 2 columnas
        y0 = .88
        colx = [.08, .56]
        for i, it in enumerate(items):
            col = 0 if i < 18 else 1
            y = y0 - (i % 18) * .045
            ans = answers.get(it.idx, None)
            ok = (ans == it.correct_idx)
            mark = "✓" if ok else "✗"
            ax3.text(colx[col], y, f"{i+1:02d}. Regla: {it.rule} | Dif.: {it.difficulty}", fontsize=10)
            ax3.text(colx[col]+.38, y, mark, fontsize=12, fontweight='bold', color=("#2a9d8f" if ok else "#e76f51"))
        pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)

        # (Opcional) primeras 6 matrices miniatura
        fig4 = plt.figure(figsize=(8.27,11.69))
        for k in range(min(6, len(items))):
            try:
                ax4 = fig4.add_axes([.08+(k%3)*.3, .52-(k//3)*.46, .24, .36]); ax4.axis('off')
                ax4.set_title(f"Ítem {k+1}", fontsize=10)
                img = plt.imread(BytesIO(items[k].question_png))
                ax4.imshow(img)
            except Exception:
                pass
        fig4.suptitle("Muestra de matrices (miniaturas)", y=0.98, fontsize=16)
        pdf.savefig(fig4, bbox_inches='tight'); plt.close(fig4)

    buf.seek(0)
    return buf.read()


def build_html(items: List[RavenItem], answers: Dict[int,int], result: Dict) -> bytes:
    rows = ""
    for i, it in enumerate(items, start=1):
        ans = answers.get(it.idx, None)
        ok = (ans == it.correct_idx)
        rows += f"<tr><td>{i:02d}</td><td>{it.rule}</td><td>{it.difficulty}</td><td>{'✓' if ok else '✗'}</td></tr>"

    secs = result["secs"]; m, s = divmod(secs, 60)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>Informe Raven (HTML)</title>
<style>
body{{font-family:Inter,Arial; margin:24px; color:#111;}}
h1{{font-size:24px; margin:0 0 8px 0;}}
h3{{font-size:18px; margin:.8rem 0 .2rem 0;}}
table{{border-collapse:collapse; width:100%; margin-top:8px}}
th,td{{border:1px solid #eee; padding:8px; text-align:left;}}
.kpi-grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}}
.kpi{{border:1px solid #eee; border-radius:12px; padding:12px; background:#fff;}}
.kpi .label{{font-size:13px; opacity:.85}}
.kpi .value{{font-size:22px; font-weight:800}}
@media print{{ .no-print{{display:none}} }}
</style>
</head>
<body>
<h1>Informe Raven — Matrices Progresivas</h1>
<p>Fecha: <b>{st.session_state.fecha}</b></p>
<div class="kpi-grid">
  <div class="kpi"><div class="label">Ítems correctos</div><div class="value">{result['raw']} / {len(items)}</div></div>
  <div class="kpi"><div class="label">% Acierto</div><div class="value">{result['pct']:.1f}%</div></div>
  <div class="kpi"><div class="label">Percentil (aprox.)</div><div class="value">{result['percentil']}</div></div>
  <div class="kpi"><div class="label">Tiempo total</div><div class="value">{m:02d}:{s:02d} min</div></div>
</div>
<h3>Resumen por ítem</h3>
<table>
  <thead><tr><th>#</th><th>Regla</th><th>Dificultad</th><th>Resultado</th></tr></thead>
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
# Vistas
# ---------------------------------------------------------------
def view_inicio():
    st.markdown(
        """
        <div class="card">
          <div class="big-title">🧩 Test Raven — Matrices Progresivas (PRO)</div>
          <p class="small" style="margin:0;">Fondo blanco · Texto negro · Diseño profesional y responsivo</p>
        </div>
        """, unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">¿Qué mide?</h3>
              <p>Razonamiento analógico y capacidad de detección de patrones visuales. Se presenta una matriz 3×3 con una casilla vacía y 8 alternativas; debes elegir la que completa la regla implícita.</p>
              <ul>
                <li>36 ítems generados procedimentalmente (sin banco externo).</li>
                <li>Auto-avance al responder cada pregunta.</li>
                <li>Duración estimada: <b>20–30 min</b> (opcional).</li>
              </ul>
              <p class="small">Esta versión es orientativa para contextos educativos/ocupacionales.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">Cómo funciona</h3>
              <ol>
                <li>Verás una matriz con 1 casilla en blanco.</li>
                <li>Selecciona la opción correcta (1 a 8) — <b>avanza automáticamente</b>.</li>
                <li>Al finalizar, verás KPIs, gráficos y desglose por regla y dificultad, y podrás descargar un informe.</li>
              </ol>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🚀 Iniciar prueba", type="primary", use_container_width=True):
            st.session_state.stage = "test"
            st.session_state.items = generate_bank(st.session_state.get("seed", 2025), n_items=36)
            st.session_state.q_idx = 0
            st.session_state.answers = {}
            st.session_state.start_time = datetime.now()
            st.session_state.end_time = None
            st.session_state.fecha = None
            ensure_bank(36)
            st.rerun()


def view_test():
    ensure_bank(36)
    items: List[RavenItem] = st.session_state.items
    i = int(st.session_state.get("q_idx", 0))

    if i < 0 or i >= len(items):
        st.session_state.stage = "resultados"
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")
        st.session_state._needs_rerun = True
        st.stop()

    it = items[i]
    progress = (i+1) / len(items)
    st.progress(progress, text=f"Progreso: {i+1}/{len(items)}")

    st.markdown(f"""
    <div class="card">
      <h3 style="margin:.2rem 0">Ítem {i+1} de {len(items)}</h3>
      <div class="badge">Regla: {it.rule}</div>
      <div class="badge">Dificultad: {it.difficulty}</div>
    </div>
    """, unsafe_allow_html=True)

    # Pregunta (matriz)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if HAS_MPL and it.question_png:
            st.image(it.question_png, use_column_width=True)
        else:
            st.info("Instala matplotlib para ver las figuras de las matrices (imágenes).")
        st.markdown("</div>", unsafe_allow_html=True)

    # Opciones (8)
    st.markdown("### Alternativas")
    cols = st.columns(4)
    for k in range(8):
        col = cols[k % 4]
        with col:
            st.markdown("<div class='choice'>", unsafe_allow_html=True)
            if HAS_MPL and it.options_png[k]:
                st.image(it.options_png[k], use_column_width=True)
            st.markdown(f"<div class='num'>Opción {k+1}</div>", unsafe_allow_html=True)
            # Radio individual por opción no es práctico; usamos radio global:
            st.markdown("</div>", unsafe_allow_html=True)

    # Selector único (1..8) — autoavance
    prev = st.session_state.answers.get(i, None)
    prev_idx = None if prev is None else prev
    st.radio(
        "Selecciona tu respuesta",
        options=list(range(8)),
        index=prev_idx,
        format_func=lambda x: f"Opción {x+1}",
        key=f"resp_{i}",
        horizontal=True,
        label_visibility="collapsed",
        on_change=on_pick,
        args=(i, ),
        kwargs={"choice": st.session_state.get(f"resp_{i}", prev if prev is not None else 0)}
    )


def view_resultados():
    if st.session_state.end_time is None:
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")

    items = st.session_state.items
    result = compute_result(items, st.session_state.answers)

    # Encabezado
    st.markdown(
        f"""
        <div class="card">
          <div class="big-title">📊 Informe Raven — Resultados</div>
          <p class="small" style="margin:0;">Fecha: <b>{st.session_state.fecha}</b></p>
        </div>
        """, unsafe_allow_html=True
    )

    # KPIs
    m, s = divmod(result["secs"], 60)
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Ítems correctos</div><div class='value'>{result['raw']} / {len(items)}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>% Acierto</div><div class='value'>{result['pct']:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Percentil (aprox.)</div><div class='value'>{result['percentil']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Tiempo total</div><div class='value'>{m:02d}:{s:02d} min</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Medidor
    if HAS_MPL:
        g = plot_gauge(result["pct"], "Exactitud general (0–100)")
        st.pyplot(g, use_container_width=True)

    st.markdown("---")

    # Gráficos por regla y dificultad
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Exactitud por tipo de regla")
        if HAS_MPL:
            fig1 = plot_rule_bars(result["rule_stats"])
            st.pyplot(fig1, use_container_width=True)
        else:
            st.info("Instala matplotlib para ver este gráfico.")
    with c2:
        st.subheader("Exactitud por dificultad")
        if HAS_MPL:
            fig2 = plot_diff_bars(result["diff_stats"])
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("Instala matplotlib para ver este gráfico.")

    # Tabla resumen por ítem
    st.markdown("---")
    st.subheader("📋 Resumen por ítem")
    rows = []
    for i, it in enumerate(items, start=1):
        ans = st.session_state.answers.get(it.idx, None)
        ok = (ans == it.correct_idx)
        rows.append({
            "Ítem": i,
            "Regla": it.rule,
            "Dificultad": it.difficulty,
            "Respuesta": (f"Opción {ans+1}" if ans is not None else "—"),
            "Resultado": "✓" if ok else "✗"
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    if HAS_MPL:
        pdf_bytes = build_pdf(items, st.session_state.answers, result)
        st.download_button(
            "⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="Informe_Raven_PRO.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        html_bytes = build_html(items, st.session_state.answers, result)
        st.download_button(
            "⬇️ Descargar Reporte (HTML) — Imprime como PDF",
            data=html_bytes,
            file_name="Informe_Raven_PRO.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Abre el HTML y usa “Imprimir → Guardar como PDF”. (Si instalas matplotlib obtendrás PDF directo.)")

    st.markdown("---")
    if st.button("🔄 Nueva prueba", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.items = []
        st.session_state.q_idx = 0
        st.session_state.answers = {}
        st.session_state.start_time = None
        st.session_state.end_time = None
        st.session_state.fecha = None
        st.rerun()


# ---------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    ensure_bank(36)
    view_test()
else:
    view_resultados()

# Rerun si el callback lo marcó (elimina doble click)
if st.session_state._needs_rerun:
    st.session_state._needs_rerun = False
    st.rerun()
