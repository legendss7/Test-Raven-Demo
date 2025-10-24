# ================================================================
#  Raven PRO — Matrices Progresivas (rápida, robusta y responsiva)
#  - Carga instantánea (render diferido de imágenes por ítem)
#  - Cache de imágenes por ítem/opción
#  - Auto-avance, métricas, PDF/HTML
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from io import BytesIO
from datetime import datetime
import random
import functools

# ==========================================
# Config
# ==========================================
st.set_page_config(
    page_title="Raven PRO | Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

N_ITEMS = 24   # ⇦ Cambia a 36 si quieres más largo
SEED    = 2025

# Intento usar matplotlib (para imágenes/PDF). Si no está, usamos fallback HTML.
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, RegularPolygon
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ==========================================
# Estilos
# ==========================================
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
  box-shadow: 0 2px 0 rgba(0,0,0,0.03); padding:18px;
}
.big-title{
  font-size:clamp(2.1rem,4.5vw,3rem); font-weight:900; margin: .2rem 0 .6rem 0;
  animation: slideIn .3s ease-out both;
}
@keyframes slideIn{ from{ transform:translateY(6px); opacity:0; } to{ transform:translateY(0); opacity:1;} }
.kpi-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0;}
.kpi{ border:1px solid #eee; border-radius:14px; background:#fff; padding:16px; position:relative; overflow:hidden;}
.kpi .label{ font-size:.95rem; opacity:.85;}
.kpi .value{ font-size:2rem; font-weight:900; line-height:1;}
.choice{
  border:1px solid #eee; border-radius:12px; padding:10px; background:#fff; text-align:center;
}
.choice .num{ font-size:.85rem; opacity:.8; }
.choice img{ border-radius:8px; border:1px solid #eee; }
.small{ font-size:0.95rem; opacity:.9; }
hr{ border:none; border-top:1px solid #eee; margin:14px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Modelo de ítem (metadatos sin imágenes)
# ==========================================
@dataclass
class RavenItem:
    idx: int
    rule: str
    difficulty: str
    correct_tuple: Tuple[str, float, float]    # (shape, size, rot) de la casilla faltante
    options: List[Tuple[str, float, float]]    # 8 opciones como tuplas
    meta: dict = field(default_factory=dict)

# ==========================================
# Utilidades de figuras (solo cuando HAS_MPL)
# ==========================================
if HAS_MPL:
    def _fig_to_png(fig) -> bytes:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")  # dpi moderado
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _new_figure(w=520, h=520):
        fig = plt.figure(figsize=(w/96, h/96), dpi=96)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal'); ax.axis('off')
        return fig, ax

    def _draw_shape(ax, shape: str, xy: Tuple[float,float], size: float, rot: float, color: str="#111"):
        x, y = xy
        if shape == "square":
            s = size
            ax.add_patch(Rectangle((x-s/2, y-s/2), s, s, angle=np.degrees(rot),
                                   linewidth=1, edgecolor=color, facecolor='none'))
        elif shape == "circle":
            ax.add_patch(Circle((x, y), radius=size/2, linewidth=1, edgecolor=color, facecolor='none'))
        elif shape == "triangle":
            ax.add_patch(RegularPolygon((x, y), 3, radius=size/2, orientation=rot,
                                        linewidth=1, edgecolor=color, facecolor='none'))
        else:  # pentagon / default
            ax.add_patch(RegularPolygon((x, y), 5, radius=size/2, orientation=rot,
                                        linewidth=1, edgecolor=color, facecolor='none'))

    def _grid_positions():
        xs = [0.2, 0.5, 0.8]
        ys = [0.8, 0.5, 0.2]
        coords=[]
        for r in range(3):
            for c in range(3):
                coords.append((xs[c], ys[r]))
        return coords

# ==========================================
# Reglas (metadatos)
# ==========================================
SHAPES = ["square","circle","triangle","pentagon"]

def _rot_rule(seed:int)->Tuple[str, str, List[dict], Tuple[str,float,float], List[Tuple[str,float,float]]]:
    rng = random.Random(seed)
    rule_name = "Rotación progresiva"; difficulty="Media"
    shape = rng.choice(SHAPES)
    base_rot = rng.choice([0, np.pi/4, np.pi/6])
    step_row = rng.choice([np.pi/8, np.pi/6])
    step_col = rng.choice([np.pi/12, np.pi/8])

    cells=[]
    for r in range(3):
        for c in range(3):
            rot = base_rot + r*step_row + c*step_col
            cells.append({"shape":shape,"size":0.20,"rot":rot})
    correct = cells[8]
    # opciones
    opts=[(correct["shape"], correct["size"], correct["rot"])]
    used={ (correct["shape"], round(correct["size"],3), round(float(correct["rot"])%6.283,3) ) }
    for _ in range(7):
        delta = rng.choice([-1,1])*rng.choice([np.pi/12,np.pi/10,np.pi/8])
        rot2 = correct["rot"] + delta
        t = (shape, 0.20, round(float(rot2)%6.283,3))
        if t in used:
            rot2 += rng.choice([np.pi/16,-np.pi/16])
            t = (shape, 0.20, round(float(rot2)%6.283,3))
        used.add(t)
        opts.append((shape, 0.20, rot2))
    rng.shuffle(opts)
    return rule_name, difficulty, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def _size_rule(seed:int)->Tuple[str, str, List[dict], Tuple[str,float,float], List[Tuple[str,float,float]]]:
    rng = random.Random(seed)
    rule_name="Tamaño progresivo"; difficulty="Media"
    shape = rng.choice(SHAPES)
    base  = rng.choice([0.12, 0.14])
    drow  = rng.choice([0.02, 0.025])
    dcol  = rng.choice([0.02, 0.025])

    cells=[]
    for r in range(3):
        for c in range(3):
            size = base + r*drow + c*dcol
            cells.append({"shape":shape,"size":size,"rot":0.0})
    correct = cells[8]
    opts=[(correct["shape"], correct["size"], 0.0)]
    used={ (correct["shape"], round(correct["size"],3), 0.0) }
    for _ in range(7):
        delta = rng.choice([-1,1])*rng.choice([0.01,0.015,0.02])
        s2 = max(0.08, min(0.28, correct["size"]+delta))
        t = (shape, round(s2,3), 0.0)
        if t in used: s2 = max(0.08, min(0.28, s2 + rng.choice([0.005,-0.005]))); t=(shape, round(s2,3), 0.0)
        used.add(t)
        opts.append((shape, s2, 0.0))
    rng.shuffle(opts)
    return rule_name, difficulty, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def _shape_rule(seed:int)->Tuple[str, str, List[dict], Tuple[str,float,float], List[Tuple[str,float,float]]]:
    rng = random.Random(seed)
    rule_name="Cambio de forma"; difficulty="Baja"
    shapes_pick = rng.sample(SHAPES, k=len(SHAPES))
    row_step = rng.choice([1,2]); col_step = rng.choice([1,3]); base_idx = rng.randint(0,3)

    cells=[]
    for r in range(3):
        for c in range(3):
            idx = (base_idx + r*row_step + c*col_step) % len(SHAPES)
            shape = shapes_pick[idx]
            cells.append({"shape":shape,"size":0.20,"rot":0.0})
    correct = cells[8]
    opts=[(correct["shape"], 0.20, 0.0)]
    used={ (correct["shape"], 0.20, 0.0) }
    for _ in range(7):
        shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
        while (shape2, 0.20, 0.0) in used:
            shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
        used.add((shape2,0.20,0.0))
        opts.append((shape2, 0.20, 0.0))
    rng.shuffle(opts)
    return rule_name, difficulty, cells, (correct["shape"], correct["size"], correct["rot"]), opts

def _mix_rule(seed:int)->Tuple[str, str, List[dict], Tuple[str,float,float], List[Tuple[str,float,float]]]:
    rng = random.Random(seed)
    rule_name="Mezcla forma+rotación"; difficulty="Alta"
    base_idx = rng.randint(0,3); row_step = rng.choice([1,2]); col_rot_step = rng.choice([np.pi/12, np.pi/8])

    cells=[]
    for r in range(3):
        for c in range(3):
            shape = SHAPES[(base_idx + r*row_step) % len(SHAPES)]
            rot   = c*col_rot_step
            cells.append({"shape":shape,"size":0.20,"rot":rot})
    correct = cells[8]
    opts=[(correct["shape"], 0.20, correct["rot"])]
    used={ (correct["shape"], 0.20, round(float(correct["rot"])%6.283,3)) }
    for _ in range(7):
        if rng.random()<0.5:
            # cambia forma
            shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
            t = (shape2, 0.20, round(float(correct["rot"])%6.283,3))
            if t in used:
                shape2 = rng.choice([s for s in SHAPES if s != correct["shape"]])
                t = (shape2, 0.20, round(float(correct["rot"])%6.283,3))
            used.add(t)
            opts.append((shape2, 0.20, correct["rot"]))
        else:
            # cambia rot
            delta = rng.choice([np.pi/12, -np.pi/12, np.pi/8, -np.pi/8])
            rot2 = correct["rot"] + delta
            t = (correct["shape"], 0.20, round(float(rot2)%6.283,3))
            if t in used:
                rot2 += rng.choice([np.pi/16, -np.pi/16])
                t = (correct["shape"], 0.20, round(float(rot2)%6.283,3))
            used.add(t)
            opts.append((correct["shape"], 0.20, rot2))
    rng.shuffle(opts)
    return rule_name, difficulty, cells, (correct["shape"], correct["size"], correct["rot"]), opts

RULE_FUNCS = [_rot_rule, _size_rule, _shape_rule, _mix_rule]

# ==========================================
# Banco de ítems (METADATOS solamente)
# ==========================================
def build_item(idx:int, seed:int)->RavenItem:
    rng = random.Random(seed + idx*97)
    fn = rng.choice(RULE_FUNCS)
    rule_name, diff, cells, correct_tuple, opts = fn(seed + idx*137)
    return RavenItem(
        idx=idx,
        rule=rule_name,
        difficulty=diff,
        correct_tuple=correct_tuple,
        options=opts,
        meta={"cells":cells}   # guardamos las celdas SOLO para render del ítem actual
    )

@st.cache_data(show_spinner=False)
def build_bank(seed:int, n_items:int)->List[RavenItem]:
    return [build_item(i, seed) for i in range(n_items)]

# ==========================================
# Render diferido (solo el ítem actual) + cache de imágenes
# ==========================================
if HAS_MPL:
    @st.cache_data(show_spinner=False)
    def render_matrix_png(cells:List[dict])->bytes:
        fig, ax = _new_figure(480, 480)
        # marco y grid
        ax.add_patch(Rectangle((0.05,0.05), 0.90, 0.90, fill=False, ec="#111", lw=1.2))
        for i in [1/3, 2/3]:
            ax.plot([0.05,0.95],[0.05+i*0.90,0.05+i*0.90], color="#bbb", lw=.8)
            ax.plot([0.05+i*0.90,0.05+i*0.90],[0.05,0.95], color="#bbb", lw=.8)
        coords = _grid_positions()
        for idx, cell in enumerate(cells):
            if idx==8:  # casilla faltante
                ax.add_patch(Rectangle((coords[idx][0]-0.12, coords[idx][1]-0.12), 0.24, 0.24,
                                       fill=False, ec="#999", lw=2, ls="--"))
                continue
            _draw_shape(ax, cell["shape"], coords[idx], cell["size"], cell["rot"])
        return _fig_to_png(fig)

    @st.cache_data(show_spinner=False)
    def render_option_png(shape:str, size:float, rot:float)->bytes:
        fig, ax = _new_figure(160, 160)
        ax.add_patch(Rectangle((0.05,0.05), 0.90, 0.90, fill=False, ec="#ddd", lw=1.1))
        _draw_shape(ax, shape, (0.5,0.5), size, rot)
        return _fig_to_png(fig)

# ==========================================
# Estado
# ==========================================
if "stage" not in st.session_state: st.session_state.stage="inicio"   # inicio | test | resultados
if "seed"  not in st.session_state: st.session_state.seed=SEED
if "items" not in st.session_state: st.session_state.items=[]
if "q_idx" not in st.session_state: st.session_state.q_idx=0
if "answers" not in st.session_state: st.session_state.answers={}
if "start_time" not in st.session_state: st.session_state.start_time=None
if "end_time"   not in st.session_state: st.session_state.end_time=None
if "fecha"      not in st.session_state: st.session_state.fecha=None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun=False

def ensure_bank():
    if not st.session_state.items:
        st.session_state.items = build_bank(st.session_state.seed, N_ITEMS)
    st.session_state.q_idx = max(0, min(int(st.session_state.q_idx), len(st.session_state.items)-1))

# ==========================================
# Scoring
# ==========================================
def compute_result(items:List[RavenItem], answers:Dict[int,int])->Dict:
    total=len(items)
    ok_list=[]
    rule_stats={}
    diff_stats={"Baja":{"ok":0,"tot":0},"Media":{"ok":0,"tot":0},"Alta":{"ok":0,"tot":0}}
    for it in items:
        a = answers.get(it.idx, None)
        ok = (a is not None) and (it.options[a]==it.correct_tuple)
        ok_list.append(1 if ok else 0)
        rule_stats.setdefault(it.rule, {"ok":0,"tot":0})
        rule_stats[it.rule]["tot"] += 1
        if ok: rule_stats[it.rule]["ok"] += 1
        diff_stats[it.difficulty]["tot"] += 1
        if ok: diff_stats[it.difficulty]["ok"] += 1
    raw = int(sum(ok_list)); pct = (raw/total*100 if total>0 else 0.0)

    # percentil simple (orientativo)
    if raw <= 8: perc = 10
    elif raw <= 12: perc = 20
    elif raw <= 16: perc = 35
    elif raw <= 19: perc = 50
    elif raw <= 22: perc = 70
    elif raw <= 24: perc = 85
    else: perc = 95

    if st.session_state.start_time and st.session_state.end_time:
        secs = int((st.session_state.end_time - st.session_state.start_time).total_seconds())
    else:
        secs = 0

    return {"raw":raw,"pct":round(pct,1),"percentil":perc,"rule_stats":rule_stats,"diff_stats":diff_stats,"secs":secs}

# ==========================================
# Callbacks
# ==========================================
def on_pick(i:int):
    choice = st.session_state.get(f"resp_{i}")
    if choice is None: return
    st.session_state.answers[i] = int(choice)
    if i < len(st.session_state.items)-1:
        st.session_state.q_idx = i+1
        st.session_state.stage = "test"
    else:
        st.session_state.stage = "resultados"
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")
    st.session_state._needs_rerun = True

# ==========================================
# Gráficos (MPL)
# ==========================================
def mpl_gauge(score:float, title:str):
    if not HAS_MPL: return None
    fig, ax = plt.subplots(figsize=(5.6,3.0), dpi=110, subplot_kw=dict(polar=True))
    ax.set_theta_direction(-1); ax.set_theta_offset(np.pi)
    ax.set_ylim(0,100); ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.bar(np.linspace(0,np.pi,100), height=[100]*100, width=np.pi/100, color="#f1f1f1", edgecolor="#f1f1f1")
    n = int(min(max(score,0),100))
    ax.bar(np.linspace(0,np.pi*(n/100), n), height=[100]*n, width=np.pi/100, color="#6D597A", edgecolor="#6D597A")
    ax.text(np.pi/2, 60, f"{score:.1f}", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig

def mpl_bar_rules(rule_stats:Dict[str,Dict[str,int]]):
    if not HAS_MPL: return None
    rules = list(rule_stats.keys())
    acc = []
    for r in rules:
        ok = rule_stats[r]["ok"]; tot = rule_stats[r]["tot"]
        acc.append(ok/tot*100 if tot>0 else 0)
    fig, ax = plt.subplots(figsize=(6.0,3.4), dpi=110)
    y = np.arange(len(rules))
    ax.barh(y, acc, color="#81B29A")
    ax.set_yticks(y); ax.set_yticklabels(rules, fontsize=9)
    ax.set_xlim(0,100); ax.set_xlabel("Exactitud (%)"); ax.set_title("Exactitud por tipo de regla")
    for i,v in enumerate(acc): ax.text(v+1,i,f"{v:.1f}%",va="center",fontsize=8)
    fig.tight_layout(); return fig

def mpl_bar_diff(diff_stats:Dict[str,Dict[str,int]]):
    if not HAS_MPL: return None
    keys = list(diff_stats.keys())
    acc=[]
    for k in keys:
        ok=diff_stats[k]["ok"]; tot=diff_stats[k]["tot"]
        acc.append(ok/tot*100 if tot>0 else 0)
    fig, ax = plt.subplots(figsize=(6.0,3.4), dpi=110)
    ax.bar(keys, acc, color="#E07A5F")
    ax.set_ylim(0,100); ax.set_ylabel("Exactitud (%)"); ax.set_title("Exactitud por dificultad")
    for i,v in enumerate(acc): ax.text(i, v+1, f"{v:.1f}%", ha="center", fontsize=8)
    fig.tight_layout(); return fig

# ==========================================
# Export
# ==========================================
def build_pdf(items:List[RavenItem], answers:Dict[int,int], result:Dict)->bytes:
    if not HAS_MPL: return b""
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada + KPIs
        fig = plt.figure(figsize=(8.27,11.69)); ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(.5,.95,"Informe Raven — Matrices Progresivas (Ocupacional)", ha='center', fontsize=20, fontweight='bold')
        ax.text(.5,.92,f"Fecha: {st.session_state.fecha}", ha='center', fontsize=11)
        raw=result["raw"]; pct=result["pct"]; perc=result["percentil"]; secs=result["secs"]; m,s=divmod(secs,60)
        y=.83
        for t,v in [("Ítems correctos",f"{raw} / {len(items)}"),
                    ("% Acierto",f"{pct:.1f}%"),
                    ("Percentil (aprox.)",f"{perc}"),
                    ("Tiempo total", f"{m:02d}:{s:02d} min")]:
            ax.text(.15,y,t,fontsize=12); ax.text(.65,y,v,fontsize=14,fontweight="bold"); y-=.05
        g = mpl_gauge(pct, "Exactitud general (0–100)")
        if g:
            img = BytesIO(); g.savefig(img, format="png", dpi=140, bbox_inches="tight"); plt.close(g); img.seek(0)
            arr = plt.imread(img); ax.imshow(arr, extent=(.1,.9,.42,.72))
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        f1 = mpl_bar_rules(result["rule_stats"])
        if f1: pdf.savefig(f1, bbox_inches='tight'); plt.close(f1)
        f2 = mpl_bar_diff(result["diff_stats"])
        if f2: pdf.savefig(f2, bbox_inches='tight'); plt.close(f2)

        # Tabla por ítem (sin miniaturas para velocidad)
        fig3 = plt.figure(figsize=(8.27,11.69)); ax3 = fig3.add_axes([0,0,1,1]); ax3.axis('off')
        ax3.text(.5,.95,"Resumen por ítem", ha='center', fontsize=16, fontweight='bold')
        y0=.90
        for i,it in enumerate(items, start=1):
            ok = (answers.get(it.idx) is not None) and (it.options[answers[it.idx]]==it.correct_tuple)
            mark = "✓" if ok else "✗"
            ax3.text(.08, y0 - i*0.032, f"{i:02d}. Regla: {it.rule} | Dif.: {it.difficulty}", fontsize=10)
            ax3.text(.75, y0 - i*0.032, mark, fontsize=12, fontweight='bold', color=("#2a9d8f" if ok else "#e76f51"))
            if i==30:  # salto a 2da columna si fuera 36
                y0=.90
        pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)
    buf.seek(0)
    return buf.read()

def build_html(items:List[RavenItem], answers:Dict[int,int], result:Dict)->bytes:
    rows=""
    for i,it in enumerate(items, start=1):
        ok=(answers.get(it.idx) is not None) and (it.options[answers[it.idx]]==it.correct_tuple)
        rows += f"<tr><td>{i:02d}</td><td>{it.rule}</td><td>{it.difficulty}</td><td>{'✓' if ok else '✗'}</td></tr>"
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
  <div class="kpi"><div class="label">Ítems correctos</div><div class="value">{result['raw']} / {len(items)}</div></div>
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

# ==========================================
# Vistas
# ==========================================
def view_inicio():
    st.markdown("""
    <div class="card">
      <div class="big-title">🧩 Test Raven — Matrices Progresivas (PRO)</div>
      <p class="small" style="margin:0;">Carga ultra-rápida · Render de imágenes por ítem · Diseño responsivo</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown("""
        <div class="card">
          <h3 style="margin-top:0">¿Qué mide?</h3>
          <p>Razonamiento fluido: detectar patrones y reglas visuales en matrices 3×3 con una casilla faltante.</p>
          <ul>
            <li><b>Número de ítems:</b> {n} (cambia <code>N_ITEMS</code> si deseas 36).</li>
            <li><b>Auto-avance:</b> al seleccionar una opción, pasas al siguiente ítem.</li>
            <li><b>Resultados:</b> KPIs, exactitud por regla y dificultad, PDF/HTML.</li>
          </ul>
        </div>
        """.replace("{n}", str(N_ITEMS)), unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <h3 style="margin-top:0">Recomendaciones</h3>
          <ul>
            <li>Usa navegador reciente (Chrome/Edge/Firefox).</li>
            <li>Si ves cuadros en blanco, instala <code>matplotlib</code> en tu entorno.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Iniciar prueba", type="primary", use_container_width=True):
            st.session_state.stage="test"
            st.session_state.items=[]   # fuerza reconstrucción limpia
            st.session_state.items = build_bank(st.session_state.seed, N_ITEMS)
            st.session_state.q_idx=0
            st.session_state.answers={}
            st.session_state.start_time=datetime.now()
            st.session_state.end_time=None
            st.session_state.fecha=None
            st.rerun()

def view_test():
    ensure_bank()
    items = st.session_state.items
    i = st.session_state.q_idx
    it = items[i]

    st.progress((i+1)/len(items), text=f"Progreso: {i+1}/{len(items)}")
    st.markdown(f"""
    <div class="card">
      <h3 style="margin:.2rem 0">Ítem {i+1} de {len(items)}</h3>
      <div class="small">Regla: <b>{it.rule}</b> · Dificultad: <b>{it.difficulty}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Matriz (render diferido y cacheado)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if HAS_MPL:
            with st.spinner("Generando figura…"):
                q_png = render_matrix_png(it.meta["cells"])
                st.image(q_png, use_column_width=True)
        else:
            st.warning("Instala matplotlib para ver las figuras. (El test sigue funcionando con opciones listadas).")
        st.markdown("</div>", unsafe_allow_html=True)

    # Opciones (8) — imágenes por opción cacheadas
    st.markdown("### Alternativas")
    cols = st.columns(4)
    for k, tup in enumerate(it.options):
        col = cols[k%4]
        with col:
            st.markdown("<div class='choice'>", unsafe_allow_html=True)
            if HAS_MPL:
                with st.spinner(None):
                    op_png = render_option_png(*tup)
                    st.image(op_png, use_column_width=True)
            st.markdown(f"<div class='num'>Opción {k+1}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Selección única y auto-avance
    prev = st.session_state.answers.get(i)
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

    items = st.session_state.items
    result = compute_result(items, st.session_state.answers)

    st.markdown(f"""
    <div class="card">
      <div class="big-title">📊 Informe Raven — Resultados</div>
      <p class="small" style="margin:0;">Fecha: <b>{st.session_state.fecha}</b></p>
    </div>
    """, unsafe_allow_html=True)

    m,s = divmod(result["secs"],60)
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Ítems correctos</div><div class='value'>{result['raw']} / {len(items)}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>% Acierto</div><div class='value'>{result['pct']:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Percentil (aprox.)</div><div class='value'>{result['percentil']}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi'><div class='label'>Tiempo total</div><div class='value'>{m:02d}:{s:02d} min</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if HAS_MPL:
        g = mpl_gauge(result["pct"], "Exactitud general (0–100)")
        st.pyplot(g, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Exactitud por tipo de regla")
        if HAS_MPL:
            st.pyplot(mpl_bar_rules(result["rule_stats"]), use_container_width=True)
        else:
            st.info("Instala matplotlib para ver este gráfico.")
    with c2:
        st.subheader("Exactitud por dificultad")
        if HAS_MPL:
            st.pyplot(mpl_bar_diff(result["diff_stats"]), use_container_width=True)
        else:
            st.info("Instala matplotlib para ver este gráfico.")

    st.markdown("---")
    st.subheader("📋 Resumen por ítem")
    rows=[]
    for i,it in enumerate(items, start=1):
        ans = st.session_state.answers.get(it.idx)
        ok = ans is not None and (it.options[ans]==it.correct_tuple)
        rows.append({
            "Ítem": i,
            "Regla": it.rule,
            "Dificultad": it.difficulty,
            "Respuesta": f"Opción {ans+1}" if ans is not None else "—",
            "Resultado": "✓" if ok else "✗"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        st.caption("Para ver figuras y PDF nativo, instala matplotlib en tu entorno.")

    st.markdown("---")
    if st.button("🔄 Nueva prueba", type="primary", use_container_width=True):
        st.session_state.stage="inicio"
        st.session_state.items=[]
        st.session_state.q_idx=0
        st.session_state.answers={}
        st.session_state.start_time=None
        st.session_state.end_time=None
        st.session_state.fecha=None
        st.rerun()

# ==========================================
# Flujo principal
# ==========================================
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    view_resultados()

# Rerun único si un callback lo pidió
if st.session_state._needs_rerun:
    st.session_state._needs_rerun=False
    st.rerun()
