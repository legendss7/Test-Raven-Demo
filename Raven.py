# app.py
# ------------------------------------------------------------
# Matrices lógicas minimalistas (estilo Raven) con miniaturas
# Requisitos: pip install streamlit pillow
# ------------------------------------------------------------
import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import random
import math
from datetime import datetime

st.set_page_config(page_title="Matrices Lógicas Minimal", layout="wide")

TOTAL_ITEMS = 60
OPTIONS = list("ABCDEFGH")

# ------------------ GENERADOR DE ÍTEMS (PROCEDURAL) ------------------
# Cada celda es un vector de 3 atributos:
#  - shape: 0=círculo, 1=cuadrado, 2=triángulo
#  - fill : 0=contorno, 1=relleno
#  - rot  : 0, 45, 90, 135 (solo visible para triángulo)
#
# Regla base en cada fila: v3 = (v1 + v2) mod base (componente a componente)
# La celda faltante es la (2,2) (fila3, col3). Se generan distractores cercanos.
#
CANVAS = 360           # tamaño del tablero grande
THUMB  = 160           # tamaño de miniatura
MARGIN = 20
GRID   = 3

def _rnd(seed):
    rnd = random.Random(seed)
    return rnd

def draw_shape(draw: ImageDraw.Draw, cx, cy, size, shape, fill, rot):
    # colores
    fg = (0, 0, 0)
    bg = (255, 255, 255)
    w = max(2, size // 18)  # grosor

    x0, y0 = cx - size//2, cy - size//2
    x1, y1 = cx + size//2, cy + size//2

    if shape == 0:  # círculo
        if fill:
            draw.ellipse([x0, y0, x1, y1], outline=fg, fill=fg, width=w)
        else:
            draw.ellipse([x0, y0, x1, y1], outline=fg, width=w)

    elif shape == 1:  # cuadrado
        if fill:
            draw.rectangle([x0, y0, x1, y1], outline=fg, fill=fg, width=w)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=fg, width=w)

    else:  # triángulo (rot admite 0,45,90,135 => 0..3)
        # triángulo isósceles rotado en 90° * k + 45° * m (m=0 o 1)
        import math as _m
        angle = [0, 45, 90, 135][rot] * _m.pi / 180.0
        r = size * 0.52
        pts = []
        for j in range(3):
            a = angle + 2 * _m.pi * j / 3
            pts.append((cx + r * _m.cos(a), cy + r * _m.sin(a)))
        if fill:
            draw.polygon(pts, outline=fg, fill=fg)
        else:
            draw.polygon(pts, outline=fg)

def render_board(cells, missing=(2, 2), size=CANVAS, show_qmark=True):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    gap = MARGIN
    cell = (size - 2 * gap) // GRID
    # líneas de la grilla
    fg = (0, 0, 0)
    for i in range(1, GRID):
        d.line([(gap, gap + i*cell), (size-gap, gap + i*cell)], fill=fg, width=1)
        d.line([(gap + i*cell, gap), (gap + i*cell, size-gap)], fill=fg, width=1)
    # borde
    d.rectangle([gap, gap, size-gap, size-gap], outline=fg, width=1)

    # dibujar shapes
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == missing:
                if show_qmark:
                    # signo ? minimalista
                    cx = gap + c*cell + cell//2
                    cy = gap + r*cell + cell//2
                    s = int(cell*0.45)
                    d.text((cx-8, cy-18), "?", fill=fg)
                continue
            shape, fill, rot = cells[r][c]
            cx = gap + c*cell + cell//2
            cy = gap + r*cell + cell//2
            size_shape = int(cell*0.62)
            draw_shape(d, cx, cy, size_shape, shape, fill, rot)
    return img

def vec_add(v1, v2):
    s = ( (v1[0]+v2[0])%3, (v1[1]+v2[1])%2, (v1[2]+v2[2])%4 )
    return s

def make_item(seed: int):
    rnd = _rnd(seed)
    # base aleatoria para las dos primeras celdas de cada fila
    # fila 0..2, col 0..1 definidas; col2 se calcula; fila2,col2 será missing
    cells = [[None]*3 for _ in range(3)]
    for r in range(3):
        for c in range(2):
            cells[r][c] = (
                rnd.randrange(0, 3),  # shape
                rnd.randrange(0, 2),  # fill
                rnd.randrange(0, 4),  # rot
            )
        cells[r][2] = vec_add(cells[r][0], cells[r][1])

    # ahora aplicamos coherencia por columnas sutil:
    # ajustar primera columna con una suma fija para dificultad
    adj = (rnd.randrange(0,3), rnd.randrange(0,2), rnd.randrange(0,4))
    for r in range(3):
        cells[r][0] = vec_add(cells[r][0], adj)
        cells[r][1] = cells[r][1]  # sin cambio
        cells[r][2] = vec_add(cells[r][2], adj)

    # respuesta correcta = valor en (2,2)
    correct = cells[2][2]

    # quitar (2,2) para el tablero con ?
    board = [row[:] for row in cells]
    board[2][2] = None

    # generar 7 distractores cercanos
    def mutate(v):
        s,f,t = v
        choice = rnd.choice([0,1,2])
        if choice == 0: s = (s + rnd.choice([1,2])) % 3
        elif choice == 1: f = 1 - f
        else: t = (t + rnd.choice([1,2])) % 4
        return (s,f,t)

    opts = [correct]
    while len(opts) < 8:
        cand = mutate(correct)
        if cand not in opts:
            opts.append(cand)
    rnd.shuffle(opts)
    correct_idx = opts.index(correct)

    # render opciones en imágenes (cuadros individuales)
    opt_imgs = []
    opt_size = 160
    for v in opts:
        im = Image.new("RGB", (opt_size, opt_size), "white")
        d = ImageDraw.Draw(im)
        s = int(opt_size*0.65)
        draw_shape(d, opt_size//2, opt_size//2, s, v[0], v[1], v[2])
        d.rectangle([4,4,opt_size-4,opt_size-4], outline=(0,0,0), width=1)
        opt_imgs.append(im)

    # render tablero grande y miniatura
    board_big = render_board(board, missing=(2,2), size=CANVAS, show_qmark=True)
    board_thumb = board_big.copy()
    board_thumb.thumbnail((THUMB, THUMB))

    return board_big, board_thumb, opt_imgs, correct_idx

# caches
@st.cache_data(show_spinner=False, max_entries=256)
def get_item_assets(idx: int):
    # idx es 0..59; seed fijo para reproducibilidad
    seed = 10_000 + idx
    board_big, board_thumb, opt_imgs, correct_idx = make_item(seed)
    # devolver bytes (para que el cache sea más liviano)
    def to_bytes(pil_img):
        b = BytesIO(); pil_img.save(b, format="PNG", optimize=True); return b.getvalue()
    return to_bytes(board_big), to_bytes(board_thumb), [to_bytes(x) for x in opt_imgs], correct_idx

# ------------------ ESTADO ------------------
if "idx" not in st.session_state: st.session_state.idx = 0
if "resp" not in st.session_state: st.session_state.resp = {}   # {i: 0..7}
if "score" not in st.session_state: st.session_state.score = 0
if "start_ts" not in st.session_state: st.session_state.start_ts = datetime.now().isoformat()

def goto(i: int):
    st.session_state.idx = i

def pick_option(i: int, k: int, correct_idx: int):
    st.session_state.resp[i] = k
    if k == correct_idx:
        # nota: si cambia la respuesta, recalculamos al finalizar
        pass
    # auto-avance
    if i < TOTAL_ITEMS-1:
        st.session_state.idx = i+1
        st.rerun()
    else:
        st.session_state.view = "resultados"
        st.rerun()

# ------------------ UI MINIMAL ------------------
if "view" not in st.session_state: st.session_state.view = "test"

# barra superior: miniaturas clicables (muy sobrio)
with st.container():
    cols = st.columns(6, gap="small")
    for i in range(TOTAL_ITEMS):
        big, thumb, _, _ = get_item_assets(i)
        with cols[i % 6]:
            st.image(thumb, use_container_width=True, caption=str(i+1))
            st.button(f"{i+1}", key=f"goto_{i}", on_click=goto, args=(i,))

st.divider()

if st.session_state.view == "test":
    i = st.session_state.idx
    big, _, opt_imgs, correct_idx = get_item_assets(i)

    # panel principal: tablero + opciones A–H
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(f"**Ítem {i+1}/{TOTAL_ITEMS}**")
        st.image(big, use_container_width=True)

    with right:
        st.markdown("**Elige la alternativa que completa la matriz**")
        # mostrar opciones en grilla minimal (4x2)
        rows = 4; cols = 2
        k = 0
        for r in range(rows):
            cs = st.columns(cols, gap="small")
            for c in range(cols):
                if k >= 8: break
                with cs[c]:
                    st.image(opt_imgs[k], use_container_width=True, caption=OPTIONS[k])
                    st.button(f"Elegir {OPTIONS[k]}",
                              key=f"pick_{i}_{k}",
                              on_click=pick_option,
                              args=(i, k, correct_idx))
                k += 1

        # marca de lo ya respondido
        if i in st.session_state.resp:
            st.caption(f"Seleccionaste: **{OPTIONS[st.session_state.resp[i]]}**")
        else:
            st.caption("Selecciona una opción para continuar →")

        # botón finalizar visible sólo al final
        if i == TOTAL_ITEMS-1:
            if st.button("Finalizar", type="primary"):
                st.session_state.view = "resultados"
                st.rerun()

elif st.session_state.view == "resultados":
    # recomputar score
    correct = 0
    data = []
    for i in range(TOTAL_ITEMS):
        _, _, _, correct_idx = get_item_assets(i)
        sel = st.session_state.resp.get(i, None)
        ok = (sel == correct_idx)
        correct += int(ok)
        data.append({"item": i+1, "respuesta": OPTIONS[sel] if sel is not None else None,
                     "correcta": OPTIONS[correct_idx], "acierto": ok})

    st.markdown("### Resultado")
    st.metric("Aciertos", f"{correct} / {TOTAL_ITEMS}")
    st.dataframe(data, use_container_width=True, hide_index=True)

    # exportar CSV
    import pandas as pd
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv,
                       file_name=f"matrices_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")

    if st.button("Reiniciar"):
        st.session_state.clear()
        st.rerun()
