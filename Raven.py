# ===================================================================
#  Matrices No Verbales (estilo Raven) — PRO MAX
#  36 ítems • 6 reglas • auto-avance • KPIs • percentil • PDF
#  (Ítems generados por código; NO usa material propietario)
# ===================================================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import random, math

# --- PDF / imágenes
HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyBboxPatch, Wedge
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from PIL import Image, ImageDraw

# ---------------------------------------------------------------
# Config general
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Matrices No Verbales | Raven-style — PRO MAX",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Estilos (UI alineada a tu Big Five PRO)
# ---------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] { display:none !important; }

html, body, [data-testid="stAppViewContainer"]{
  background:#ffffff !important; color:#111 !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{ max-width:1200px; padding-top:0.8rem; padding-bottom:2rem; }

.card{
  border:1px solid #eee; border-radius:14px; background:#fff;
  box-shadow: 0 2px 0 rgba(0,0,0,0.03); padding:18px;
}
hr{ border:none; border-top:1px solid #eee; margin:16px 0; }
.small{ font-size:0.95rem; opacity:.9; }

/* Título grande animado */
.page-title{
  font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900; margin:.2rem 0 .6rem 0;
  animation: slideIn .3s ease-out both;
}
@keyframes slideIn{ from{transform:translateY(6px);opacity:0;} to{transform:translateY(0);opacity:1;} }

/* KPIs */
.kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.kpi{ border:1px solid #eee; border-radius:14px; background:#fff; padding:16px; position:relative; overflow:hidden; }
.kpi .label{ font-size:.95rem; opacity:.85; }
.kpi .value{ font-size:2.2rem; font-weight:900; line-height:1; }

/* Opciones */
.opt-grid{ display:grid; grid-template-columns:repeat(4, minmax(72px, 1fr)); gap:10px; }
.opt{ border:1px solid #eaeaea; border-radius:10px; padding:8px; background:#fff; cursor:pointer; }
.opt:hover{ box-shadow:0 2px 0 rgba(0,0,0,.06); }

/* Botones */
button[kind="primary"], button[kind="secondary"]{ width:100%; }

/* Tablas pequeñas */
[data-testid="stDataFrame"] div[role="grid"]{ font-size:0.95rem; }

/* Badges */
.badge{ display:inline-block; padding:.2rem .6rem; border:1px solid #eee; border-radius:999px; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Utilidades de dibujo (PIL)
# ---------------------------------------------------------------
def draw_shape(draw: ImageDraw.ImageDraw, cx, cy, size, shape="circle", angle=0, fill=0):
    s = size
    if shape == "circle":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], outline=0, width=4,
                     fill=None if fill==0 else 200)
    elif shape == "square":
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=0, width=4,
                       fill=None if fill==0 else 200)
    elif shape == "triangle":
        pts = []
        for k in range(3):
            a = math.radians(angle + 120*k - 90)
            pts.append((cx + s*math.cos(a), cy + s*math.sin(a)))
        draw.polygon(pts, outline=0, width=4, fill=None if fill==0 else 200)
    else:
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], outline=0, width=4,
                     fill=None if fill==0 else 200)

def render_cell(w=140, h=140, shape="circle", count=1, size=24, angle=0, fill=0):
    img = Image.new("L", (w, h), color=255)
    d = ImageDraw.Draw(img)
    cols = int(np.ceil(np.sqrt(max(1,count))))
    rows = int(np.ceil(max(1,count) / cols))
    margin = 18
    grid_w = w - 2*margin
    grid_h = h - 2*margin
    step_x = grid_w / max(1, cols-1) if cols>1 else 0
    step_y = grid_h / max(1, rows-1) if rows>1 else 0
    c = 0
    for r in range(rows):
        for ccol in range(cols):
            if c >= count: break
            cx = margin + ccol*step_x
            cy = margin + r*step_y
            draw_shape(d, cx, cy, size, shape=shape, angle=angle, fill=fill)
            c += 1
    return img

def compose_matrix(cells, missing_idx=(2,2), cell_size=140, gap=10):
    W = cell_size*3 + gap*4
    H = cell_size*3 + gap*4
    img = Image.new("L", (W, H), color=255)
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,W-1,H-1], outline=0, width=4)
    for i in range(3):
        for j in range(3):
            x = gap + j*(cell_size+gap)
            y = gap + i*(cell_size+gap)
            if (i,j) == missing_idx:
                d.rectangle([x,y,x+cell_size,y+cell_size], outline=0, width=3)
            else:
                img.paste(cells[i][j], (x,y))
    return img

# ---------------------------------------------------------------
# Generadores de reglas
# ---------------------------------------------------------------
def gen_rule_count(seed=None, difficulty=1):
    rng = random.Random(seed)
    base = rng.randint(1, 2 if difficulty==1 else (2 if difficulty==2 else 3))
    step_r = rng.randint(1, 2 if difficulty==1 else 3)
    step_c = rng.randint(0, 2 if difficulty==1 else 3)
    shape = rng.choice(["circle","square","triangle"])
    size  = rng.choice([18,22,26])
    fill  = rng.choice([0,1]) if difficulty>=2 else 0

    vals = [[base + i*step_r + j*step_c for j in range(3)] for i in range(3)]
    correct = vals[2][2]
    cells = [[render_cell(shape=shape, count=vals[i][j], size=size, fill=fill) for j in range(3)] for i in range(3)]

    correct_img = render_cell(shape=shape, count=correct, size=size, fill=fill)
    opts, used = [correct_img], {correct}
    while len(opts) < 8:
        delta = rng.choice([-3,-2,-1,1,2,3,4,-4])
        c = max(0, correct + delta)
        if c in used: continue
        used.add(c)
        opts.append(render_cell(shape=shape, count=c, size=size, fill=fill))
    rng.shuffle(opts)
    return dict(rule="count", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

def gen_rule_rotation(seed=None, difficulty=1):
    rng = random.Random(seed)
    step_r = rng.choice([15,30,45]) if difficulty==1 else rng.choice([30,45,60])
    step_c = rng.choice([15,30,45]) if difficulty<=2 else rng.choice([30,45,60,90])
    base   = rng.randint(0, 3)*15
    size   = rng.choice([20,24,28]); fill = 0

    angles = [[(base + i*step_r + j*step_c)%360 for j in range(3)] for i in range(3)]
    corr   = angles[2][2]
    cells  = [[render_cell(shape="triangle", count=1, size=size, angle=angles[i][j]) for j in range(3)] for i in range(3)]

    correct_img = render_cell(shape="triangle", count=1, size=size, angle=corr)
    opts, used = [correct_img], {corr}
    while len(opts)<8:
        delta = rng.choice([-90,-60,-45,-30,-15,15,30,45,60,90])
        a = (corr + delta) % 360
        if a in used: continue
        used.add(a)
        opts.append(render_cell(shape="triangle", count=1, size=size, angle=a))
    rng.shuffle(opts)
    return dict(rule="rotation", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

def gen_rule_size(seed=None, difficulty=1):
    rng = random.Random(seed)
    base = rng.choice([10,12,14])
    step_r = rng.choice([3,4,5]) if difficulty==1 else rng.choice([4,5,6,7])
    step_c = rng.choice([2,3,4]) if difficulty<=2 else rng.choice([3,4,5,6])
    shape = rng.choice(["circle","square"]); fill = rng.choice([0,1]) if difficulty==3 else 0

    sizes = [[base + i*step_r + j*step_c for j in range(3)] for i in range(3)]
    corr = sizes[2][2]
    cells = [[render_cell(shape=shape, count=1, size=sizes[i][j], fill=fill) for j in range(3)] for i in range(3)]

    correct_img = render_cell(shape=shape, count=1, size=corr, fill=fill)
    opts, used = [correct_img], {corr}
    while len(opts)<8:
        delta = rng.choice([-6,-4,-3,-2,2,3,4,6])
        s = max(6, corr + delta)
        if s in used: continue
        used.add(s)
        opts.append(render_cell(shape=shape, count=1, size=s, fill=fill))
    rng.shuffle(opts)
    return dict(rule="size", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

def gen_rule_shape_alternation(seed=None, difficulty=1):
    """Alterna forma por filas y columnas (p.ej. círculo→cuadrado→triángulo)."""
    rng = random.Random(seed)
    seq = ["circle","square","triangle"]
    shift_r = rng.randint(0,2)
    shift_c = rng.randint(0,2)
    size = rng.choice([18,22,26]); fill = rng.choice([0,1]) if difficulty>=2 else 0
    # patrón forma = seq[(i+shift_r + j+shift_c) % 3]
    forms = [[seq[(i+shift_r + j+shift_c) % 3] for j in range(3)] for i in range(3)]
    cells = [[render_cell(shape=forms[i][j], count=1, size=size, fill=fill) for j in range(3)] for i in range(3)]
    correct_shape = forms[2][2]
    correct_img = render_cell(shape=correct_shape, count=1, size=size, fill=fill)

    opts = [correct_img]
    others = [s for s in seq if s != correct_shape]
    for s in others:
        opts.append(render_cell(shape=s, count=1, size=size, fill=fill))
    # completar hasta 8 con variantes de tamaño para ruido
    while len(opts)<8:
        s = rng.choice(seq)
        sz = size + rng.choice([-4,-2,2,4])
        opts.append(render_cell(shape=s, count=1, size=max(10,sz), fill=fill))
    rng.shuffle(opts)
    return dict(rule="shape_alt", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

def gen_rule_fill_toggle(seed=None, difficulty=1):
    """El relleno alterna (vacío/lleno) por filas y columnas."""
    rng = random.Random(seed)
    shape = rng.choice(["circle","square","triangle"])
    base_fill = rng.choice([0,1])
    step_r = rng.choice([1])  # toggle fila
    step_c = rng.choice([1])  # toggle columna
    size = rng.choice([20,24,28])
    fills = [[(base_fill + i*step_r + j*step_c) % 2 for j in range(3)] for i in range(3)]
    cells = [[render_cell(shape=shape, count=1, size=size, fill=fills[i][j]) for j in range(3)] for i in range(3)]
    correct_fill = fills[2][2]
    correct_img = render_cell(shape=shape, count=1, size=size, fill=correct_fill)

    # opciones = mismo shape, distintos fills
    opts = [correct_img,
            render_cell(shape=shape, count=1, size=size, fill=1-correct_fill)]
    # completar con pequeñas variaciones de tamaño
    while len(opts)<8:
        sz = size + rng.choice([-4,-2,2,4])
        f = rng.choice([0,1])
        opts.append(render_cell(shape=shape, count=1, size=max(10,sz), fill=f))
    rng.shuffle(opts)
    return dict(rule="fill_toggle", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

def gen_rule_combo_count_x_rotation(seed=None, difficulty=1):
    """Combinada: cantidad + rotación se suman por fila/columna."""
    rng = random.Random(seed)
    base_count = rng.randint(1, 2 if difficulty==1 else 3)
    step_rc = rng.randint(1, 2 if difficulty==1 else 3)
    base_angle = rng.choice([0,15,30])
    step_ra = rng.choice([15,30,45]) if difficulty<=2 else rng.choice([30,45,60])
    size = rng.choice([18,22,26])
    counts = [[base_count + i*step_rc + j*step_rc for j in range(3)] for i in range(3)]
    angles = [[(base_angle + i*step_ra + j*step_ra)%360 for j in range(3)] for i in range(3)]
    shape  = "triangle"
    cells  = [[render_cell(shape=shape, count=counts[i][j], size=size, angle=angles[i][j]) for j in range(3)] for i in range(3)]
    corr_c = counts[2][2]; corr_a = angles[2][2]
    correct_img = render_cell(shape=shape, count=corr_c, size=size, angle=corr_a)

    opts = [correct_img]
    used = set()
    while len(opts)<8:
        dc = rng.choice([-2,-1,1,2,3,-3])
        da = rng.choice([-60,-45,-30,-15,15,30,45,60])
        cc = max(0, corr_c + dc)
        aa = (corr_a + da) % 360
        key = (cc,aa)
        if key in used or (cc==corr_c and aa==corr_a): continue
        used.add(key)
        opts.append(render_cell(shape=shape, count=cc, size=size, angle=aa))
    rng.shuffle(opts)
    return dict(rule="count_x_rot", difficulty=difficulty,
                matrix=compose_matrix(cells), options=opts, answer=opts.index(correct_img))

# Banco de 36 ítems
ALL_RULES = [
    gen_rule_count,
    gen_rule_rotation,
    gen_rule_size,
    gen_rule_shape_alternation,
    gen_rule_fill_toggle,
    gen_rule_combo_count_x_rotation,
]

def generate_bank(seed=2025, n_items=36):
    rng = random.Random(seed)
    items = []
    # 12 fáciles, 12 medios, 12 difíciles
    diffs = ([1]*12 + [2]*12 + [3]*12)[:n_items]
    for i in range(n_items):
        g = ALL_RULES[i % len(ALL_RULES)]
        items.append(g(seed=rng.randint(1,10**9), difficulty=diffs[i]))
    rng.shuffle(items)
    return items

# ---------------------------------------------------------------
# Estado
# ---------------------------------------------------------------
if "stage" not in st.session_state: st.session_state.stage = "inicio"  # inicio | test | resultados
if "seed" not in st.session_state: st.session_state.seed = 2025
if "items" not in st.session_state: st.session_state.items = generate_bank(st.session_state.seed, 36)
if "q_idx" not in st.session_state: st.session_state.q_idx = 0
if "answers" not in st.session_state: st.session_state.answers = {}  # idx -> 0..7
if "start_time" not in st.session_state: st.session_state.start_time = None
if "end_time" not in st.session_state: st.session_state.end_time = None
if "fecha" not in st.session_state: st.session_state.fecha = None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun = False

# ---------------------------------------------------------------
# Avance
# ---------------------------------------------------------------
def on_pick(idx:int, choice:int):
    st.session_state.answers[idx] = choice
    if idx < len(st.session_state.items)-1:
        st.session_state.q_idx = idx + 1
    else:
        st.session_state.stage = "resultados"
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")
    st.session_state._needs_rerun = True

# ---------------------------------------------------------------
# Scoring + percentil (ponderado por dificultad)
# ---------------------------------------------------------------
def norm_percentile(weighted_score):
    """
    Estimación de percentil sobre supuesta distribución ~N(μ=60, σ=15) del score ponderado.
    """
    z = (weighted_score - 60.0) / 15.0
    phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0, min(1.0, phi)) * 100.0

def score_results():
    items = st.session_state.items
    answers = st.session_state.answers
    correct_flags, diffs, rules = [], [], []
    weights = {1:1.0, 2:1.4, 3:1.8}
    weight_vals, weight_hits = [], []

    for i, it in enumerate(items):
        user = answers.get(i, None)
        ok = (user is not None) and (int(user) == int(it["answer"]))
        correct_flags.append(1 if ok else 0)
        d = it["difficulty"]
        diffs.append(d); rules.append(it["rule"])
        w = weights[d]
        weight_vals.append(w)
        weight_hits.append(w if ok else 0.0)

    total = sum(correct_flags)
    perc = total / len(items) * 100.0

    # ponderado
    weighted_score = (sum(weight_hits) / sum(weight_vals)) * 100.0
    percentile = norm_percentile(weighted_score)

    df = pd.DataFrame({"ok":correct_flags,"diff":diffs,"rule":rules})
    by_diff = df.groupby("diff")["ok"].mean().reindex([1,2,3]).fillna(0.0)*100
    by_rule = df.groupby("rule")["ok"].mean().reindex(
        ["count","rotation","size","shape_alt","fill_toggle","count_x_rot"]
    ).fillna(0.0)*100

    if st.session_state.start_time and st.session_state.end_time:
        sec = (st.session_state.end_time - st.session_state.start_time).total_seconds()
    else:
        sec = 0.0

    return dict(
        total_correct=total,
        total_items=len(items),
        accuracy=round(perc,1),
        weighted=round(float(weighted_score),1),
        percentile=round(float(percentile),1),
        by_diff=by_diff,
        by_rule=by_rule,
        per_item_ok=correct_flags,
        seconds=int(sec)
    )

# ---------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------
def gauge_plotly(value: float, title: str = "", color="#6D597A"):
    v = max(0, min(100, float(value)))
    bounds = [0, 25, 50, 75, 100]
    colors = ["#fde2e1", "#fff0c2", "#e7f3ff", "#e4f5ea"]
    vals = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=vals, hole=0.6, rotation=180, direction="clockwise",
        textinfo="none", marker=dict(colors=colors, line=dict(color="#fff", width=1)),
        hoverinfo="skip", showlegend=False, sort=False
    ))
    theta = (180 * (v/100.0))
    r = 0.95; x0, y0 = 0.5, 0.5
    xe = x0 + r*math.cos(math.radians(180 - theta))
    ye = y0 + r*math.sin(math.radians(180 - theta))
    fig.add_shape(type="line", x0=x0, y0=y0, x1=xe, y1=ye, line=dict(color=color, width=4))
    fig.add_shape(type="circle", x0=x0-0.02, y0=y0-0.02, x1=x0+0.02, y1=y0+0.02,
                  line=dict(color=color), fillcolor=color)
    fig.update_layout(
        annotations=[
            dict(text=f"<b>{v:.1f}%</b>", x=0.5, y=0.32, showarrow=False, font=dict(size=24, color="#111")),
            dict(text=title, x=0.5, y=0.16, showarrow=False, font=dict(size=13, color="#333")),
        ],
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False, height=220
    )
    return fig

def plot_bar(series: pd.Series, title: str):
    palette = ["#81B29A","#F2CC8F","#E07A5F","#9EC1CF","#A593E0","#88D8B0"]
    fig = go.Figure()
    colors = [palette[i%len(palette)] for i in range(len(series))]
    fig.add_trace(go.Bar(
        x=series.index.astype(str), y=series.values,
        marker=dict(color=colors), text=[f"{v:.1f}%" for v in series.values],
        textposition="outside"
    ))
    fig.update_layout(height=320, template="plotly_white",
                      yaxis=dict(range=[0,105], title="Aciertos (%)"),
                      xaxis_title="", title=title)
    return fig

# ---------------------------------------------------------------
# PDF
# ---------------------------------------------------------------
def pil_to_array(im): return np.array(im.convert("RGB"))

def build_pdf(results, fecha):
    items = st.session_state.items
    answers = st.session_state.answers

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Portada + KPIs + gauge simple
        fig = plt.figure(figsize=(8.27,11.69))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(.5,.95,"Informe — Matrices No Verbales (Raven-style)", ha='center', fontsize=18, fontweight='bold')
        ax.text(.5,.92,f"Fecha: {fecha}", ha='center', fontsize=10)

        def card(ax, x,y,w,h,title,val):
            r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               edgecolor="#dddddd", facecolor="#ffffff")
            ax.add_patch(r)
            ax.text(x+w*0.06, y+h*0.60, title, fontsize=10, color="#333")
            ax.text(x+w*0.06, y+h*0.25, f"{val}", fontsize=18, fontweight='bold')

        Y0=.82; H=.10; W=.40; GAP=.02
        card(ax, .06, Y0,       W, H, "Ítems correctos", f"{results['total_correct']}/{results['total_items']}")
        card(ax, .54, Y0,       W, H, "Precisión", f"{results['accuracy']:.1f}%")
        card(ax, .06, Y0-(H+GAP), W, H, "Score ponderado", f"{results['weighted']:.1f}%")
        card(ax, .54, Y0-(H+GAP), W, H, "Percentil estimado", f"P{int(round(results['percentile']))}")

        # Gauge (semicírculo básico)
        axg = fig.add_axes([.30,.70,.40,.10]); axg.axis('off')
        base = Wedge((0.5,0), 1.0, 0, 180, facecolor="#f5f5f5", edgecolor="#e1e1e1")
        axg.add_patch(base)
        val = max(0,min(100, float(results["accuracy"])))
        angle = 180 * (val/100.0)
        needle = Wedge((0.5,0), 1.0, 0, angle, facecolor="#cfe8d8", edgecolor="#cfe8d8")
        axg.add_patch(needle)
        axg.text(0.5, -0.35, f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
        axg.set_xlim(-0.1,1.1); axg.set_ylim(-0.6,1.1)

        # Barras por dificultad
        ax2 = fig.add_axes([.10,.47,.80,.18])
        x = list(results["by_diff"].index.astype(int)); y = list(results["by_diff"].values)
        ax2.bar(x, y, color=["#81B29A","#F2CC8F","#E07A5F"])
        ax2.set_ylim(0,100); ax2.set_xlabel("Dificultad"); ax2.set_ylabel("Aciertos (%)")
        ax2.set_title("Rendimiento por dificultad")
        for xi, yi in zip(x,y): ax2.text(xi, yi+1, f"{yi:.1f}%", ha="center", fontsize=9)

        # Barras por regla
        ax3 = fig.add_axes([.10,.22,.80,.18])
        xr = list(range(len(results["by_rule"])))
        y2 = list(results["by_rule"].values)
        ax3.bar(xr, y2, color=["#a5c9a1","#c9b29f","#9fbcd1","#d7c6f3","#ffcfae","#a7e3c3"])
        ax3.set_ylim(0,100); ax3.set_xticks(xr); ax3.set_xticklabels(list(results["by_rule"].index), rotation=10)
        ax3.set_ylabel("Aciertos (%)")
        ax3.set_title("Rendimiento por tipo de regla")
        for xi, yi in zip(xr,y2): ax3.text(xi, yi+1, f"{yi:.1f}%", ha="center", fontsize=9)

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Páginas de detalle (6 ítems por página)
        per_page = 6
        pages = int(np.ceil(len(items)/per_page))
        for p in range(pages):
            fig = plt.figure(figsize=(8.27,11.69))
            ax = fig.add_axes([0,0,1,1]); ax.axis('off')
            ax.text(.5,.96,f"Detalle de ítems ({p+1}/{pages})", ha='center', fontsize=14, fontweight='bold')
            rows, cols = 3, 2
            for k in range(per_page):
                idx = p*per_page + k
                if idx >= len(items): break
                r = k // cols; c = k % cols
                left = .07 + c*0.46; top = .86 - r*0.29
                sub = fig.add_axes([left, top-0.25, 0.42, 0.22]); sub.axis('off')
                sub.set_title(f"Ítem {idx+1} · Regla: {items[idx]['rule']} · Dif: {items[idx]['difficulty']}", fontsize=9)
                sub.imshow(pil_to_array(items[idx]["matrix"]))
                usr = st.session_state.answers.get(idx, None)
                ok  = items[idx]["answer"]
                info = f"Tu resp.: {('-' if usr is None else usr+1)}   |   Correcta: {ok+1}"
                sub.text(0.5, -0.08, info, transform=sub.transAxes, ha='center', fontsize=9)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------
def view_inicio():
    st.markdown(f"""
    <div class="card">
      <div class="page-title">🧩 Matrices No Verbales (Raven-style) — PRO MAX</div>
      <p class="small">36 ítems • 6 reglas • auto-avance • KPIs • percentil • PDF • diseño responsivo</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0">Formato</h3>
            <p>Matrices 3×3 con una celda faltante y 8 alternativas. Reglas lógicas:</p>
            <ul>
              <li><b>Cantidad</b> (suma por filas/columnas)</li>
              <li><b>Rotación</b> (ángulos que progresan)</li>
              <li><b>Tamaño</b> (incrementos por fila/columna)</li>
              <li><b>Alternancia de forma</b> (círculo/cuadrado/triángulo)</li>
              <li><b>Toggle de relleno</b> (vacío/lleno)</li>
              <li><b>Combinada</b> <i>(cantidad × rotación)</i></li>
            </ul>
            <p class="small">Los ítems se generan por código con su clave exacta; no se usan láminas oficiales.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0">Corrección y métricas</h3>
            <ul>
              <li>Puntaje total y precisión global</li>
              <li><b>Score ponderado</b> por dificultad (más valor a ítems difíciles)</li>
              <li><b>Percentil estimado</b> sobre normas simuladas (μ=60, σ=15)</li>
              <li>Rendimiento por <b>dificultad</b> y por <b>regla</b></li>
              <li>PDF con KPIs, gauge, barras y miniaturas por ítem</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Iniciar prueba", type="primary", use_container_width=True):
            st.session_state.stage = "test"
            st.session_state.items = generate_bank(st.session_state.seed, n_items=36)
            st.session_state.q_idx = 0
            st.session_state.answers = {}
            st.session_state.start_time = datetime.now()
            st.session_state.end_time = None
            st.session_state.fecha = None
            st.rerun()

def view_test():
    i = st.session_state.q_idx
    items = st.session_state.items
    it = items[i]
    total = len(items)
    st.progress((i+1)/total, text=f"Progreso: {i+1}/{total}")
    st.markdown(f"<div class='page-title'>Ítem {i+1} de {total}</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Regla: {it['rule']}</span> &nbsp; <span class='badge'>Dificultad: {it['difficulty']}</span>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2,1])
    with col1:
        st.image(it["matrix"], caption="Matriz 3×3 (selecciona la pieza faltante)", use_container_width=True)

    with col2:
        st.markdown("#### Alternativas (elige una)")
        grid_cols = st.columns(4)
        for k in range(8):
            with grid_cols[k%4]:
                st.image(it["options"][k].resize((140,140)), use_container_width=True)
                if st.button(f"Elegir #{k+1}", key=f"pick_{i}_{k}", use_container_width=True):
                    on_pick(i, k)

        chosen = st.session_state.answers.get(i, None)
        if chosen is not None:
            st.info(f"Seleccionaste la opción #{chosen+1}. Puedes cambiarla; el avance es automático.")

def view_resultados():
    res = score_results()
    st.markdown(f"""
    <div class="card">
      <div class="page-title">📊 Resultados — Matrices No Verbales</div>
      <p class="small">Fecha: <b>{st.session_state.fecha or datetime.now().strftime("%d/%m/%Y %H:%M")}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs + gauges
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.markdown("<div class='kpi'><div class='label'>Ítems correctos</div>"
                    f"<div class='value'>{res['total_correct']}/{res['total_items']}</div></div>", unsafe_allow_html=True)
    with c1:
        st.markdown("<div class='kpi'><div class='label'>Precisión global</div>"
                    f"<div class='value'>{res['accuracy']:.1f}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><div class='label'>Score ponderado</div>"
                    f"<div class='value'>{res['weighted']:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><div class='label'>Percentil estimado</div>"
                    f"<div class='value'>P{int(round(res['percentile']))}</div></div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_plotly(res["accuracy"], title="Precisión global"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_plotly(res["weighted"], title="Score ponderado"), use_container_width=True)

    st.markdown("---")
    c3_, c4_ = st.columns(2)
    with c3_:
        st.plotly_chart(plot_bar(res["by_diff"], "Rendimiento por dificultad"), use_container_width=True)
    with c4_:
        st.plotly_chart(plot_bar(res["by_rule"], "Rendimiento por tipo de regla"), use_container_width=True)

    # Tabla detalle
    st.markdown("---")
    st.subheader("📋 Detalle por ítem")
    rows = []
    for i,it in enumerate(st.session_state.items):
        user = st.session_state.answers.get(i, None)
        ok = it["answer"]
        rows.append(dict(
            Item=i+1, Regla=it["rule"], Dificultad=it["difficulty"],
            Respuesta=("—" if user is None else user+1), Correcta=ok+1,
            Acierto="✅" if (user is not None and user==ok) else "❌"
        ))
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Exportar
    st.markdown("---")
    st.subheader("📥 Exportar informe")
    if HAS_MPL:
        pdf_bytes = build_pdf(res, st.session_state.fecha or datetime.now().strftime("%d/%m/%Y %H:%M"))
        st.download_button(
            "⬇️ Descargar PDF (KPIs + gauge + miniaturas)",
            data=pdf_bytes,
            file_name="Informe_Matrices_No_Verbales_PROMAX.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    else:
        st.info("Instala `matplotlib` para obtener el PDF directo con KPIs e ítems.")

    st.markdown("---")
    if st.button("🔄 Nueva prueba", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {}
        st.session_state.items = generate_bank(st.session_state.seed, n_items=36)
        st.session_state.start_time = None
        st.session_state.end_time = None
        st.session_state.fecha = None
        st.rerun()

# ---------------------------------------------------------------
# FLUJO PRINCIPAL
# ---------------------------------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    if st.session_state.end_time is None:
        st.session_state.end_time = datetime.now()
        st.session_state.fecha = st.session_state.end_time.strftime("%d/%m/%Y %H:%M")
    view_resultados()

# Rerun único para evitar doble click
if st.session_state._needs_rerun:
    st.session_state._needs_rerun = False
    st.rerun()
