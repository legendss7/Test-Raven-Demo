# streamlit_app_logic.py
# ================================================================
#  Test de Lógica Matemática – Analista (estructura Big Five PRO)
#  - Misma arquitectura y UX que el archivo Big Five:
#    stage: inicio | test | resultados
#    q_idx, answers, fecha, _needs_rerun
#    view_inicio(), view_test(), view_resultados()
#    auto-avance con radio.on_change + _needs_rerun
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# ---------------------------------------------------------------
# Config general (igual estilo base)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Lógica Matemática | Evaluación de Analista",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Estilos (reutiliza paleta/estructura visual + mejoras)
# ---------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"]{ display:none !important; }
html, body, [data-testid="stAppViewContainer"]{
  background:#ffffff !important; color:#111 !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{ max-width:1200px; padding-top:0.8rem; padding-bottom:2rem; }
.card{ border:1px solid #eee; border-radius:14px; background:#fff; box-shadow:0 2px 0 rgba(0,0,0,0.03); padding:18px; }
.dim-title{ font-size:clamp(2.0rem, 5vw, 3.0rem); font-weight:900; letter-spacing:.2px; line-height:1.12; margin:.2rem 0 .6rem 0; }
.dim-desc{ margin:.1rem 0 1rem 0; opacity:.9; }
.kpi-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; margin:10px 0 6px 0; }
.kpi{ border:1px solid #eee; border-radius:14px; background:#fff; padding:16px; position:relative; overflow:hidden; }
.kpi .label{ font-size:.95rem; opacity:.85; }
.kpi .value{ font-size:2.0rem; font-weight:900; line-height:1; }
.small{ font-size:.95rem; opacity:.9; }
.tag{ display:inline-block; padding:.2rem .6rem; border:1px solid #eee; border-radius:999px; font-size:.82rem; }
hr{ border:none; border-top:1px solid #eee; margin:16px 0; }

/* Bloque imprimible: optimiza márgenes y oculta botones en PDF */
@media print{
  [data-testid="stToolbar"], .stButton, .stDownloadButton, .stDownloadButton>button { display:none !important; }
  .block-container{ padding:0 !important; }
  .card{ box-shadow:none !important; border:1px solid #ddd; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Banco de preguntas (60) con competencias
# ---------------------------------------------------------------
# Estructura de cada pregunta:
# {"key": "LM1", "texto": "...", "alternativas": ["...","...","...","...","..."], "correcta": "B",
#  "competencia": "Razonamiento cuantitativo"}

COMPETENCIAS = [
    "Razonamiento cuantitativo",
    "Cálculo aplicado",
    "Análisis de datos",
    "Lógica combinatoria",
    "Proporciones y razón",
]

# Generadores simples para construir 60 ítems variados
from math import comb

def gen_series(i, base, step):
    seq = [base + step*k for k in range(5)]
    correcto = seq[-1] + step
    alts = sorted({correcto, correcto+step, max(1, correcto-step), correcto+2*step, correcto+3})
    letras = ["A","B","C","D","E"]
    correcta = letras[alts.index(correcto)]
    return {
        "key": f"LM{i}",
        "texto": f"Completa la serie: {', '.join(map(str, seq))}. ¿Cuál sigue?",
        "alternativas": [str(a) for a in alts],
        "correcta": correcta,
        "competencia": COMPETENCIAS[0],
    }

def gen_pct(i, precio, pct):
    inc = round(precio*pct/100)
    total = precio + inc
    alts = sorted({total, total-10, total+10, total-5, total+5})
    letras = ["A","B","C","D","E"]
    correcta = letras[alts.index(total)]
    return {
        "key": f"LM{i}",
        "texto": f"Un artículo cuesta ${precio}. Aumenta {pct}% ¿Cuál es el nuevo precio?",
        "alternativas": [f"${a}" for a in alts],
        "correcta": correcta,
        "competencia": COMPETENCIAS[1],
    }

def gen_tabla(i, a=120, b=150, c=90, d=140):
    prom = round((a+b+c+d)/4)
    alts = sorted({prom, prom+5, prom-5, prom+10, prom-10})
    letras = ["A","B","C","D","E"]
    correcta = letras[alts.index(prom)]
    return {
        "key": f"LM{i}",
        "texto": "Según la tabla (A=120, B=150, C=90, D=140), ¿cuál es el promedio de ventas?",
        "alternativas": [str(x) for x in alts],
        "correcta": correcta,
        "competencia": COMPETENCIAS[2],
    }

def gen_comb(i, n, k):
    val = comb(n, k)
    alts = sorted({val, max(1, val-1), val+1, max(1, val-2), val+2})
    letras = ["A","B","C","D","E"]
    correcta = letras[alts.index(val)]
    return {
        "key": f"LM{i}",
        "texto": f"¿Cuántos comités distintos de {k} personas pueden formarse a partir de {n} candidatos? (sin orden)",
        "alternativas": [str(x) for x in alts],
        "correcta": correcta,
        "competencia": COMPETENCIAS[3],
    }

def gen_prop(i, a, b, x):
    y = int(round((b*x)/a))
    alts = sorted({y, max(1, y-1), y+1, max(1, y-2), y+2})
    letras = ["A","B","C","D","E"]
    correcta = letras[alts.index(y)]
    return {
        "key": f"LM{i}",
        "texto": f"Si {a} es a {b} como {x} es a ¿y?",
        "alternativas": [str(v) for v in alts],
        "correcta": correcta,
        "competencia": COMPETENCIAS[4],
    }

# Construye 60 preguntas (12 de cada tipo)
QUESTIONS = []
idx = 1
for base, step in [(2,3),(5,4),(10,2),(3,5),(7,6),(1,7),(4,4),(8,3),(6,5),(9,4),(11,2),(12,3)]:
    QUESTIONS.append(gen_series(idx, base, step)); idx += 1
for precio, pct in [(100,10),(250,15),(400,20),(120,5),(300,12),(180,18),(90,25),(220,8),(560,7),(800,9),(150,30),(360,22)]:
    QUESTIONS.append(gen_pct(idx, precio, pct)); idx += 1
for _ in range(12):
    QUESTIONS.append(gen_tabla(idx)); idx += 1
for n,k in [(6,2),(7,3),(8,2),(9,3),(10,2),(7,4),(8,3),(9,4),(10,3),(6,3),(8,4),(10,4)]:
    QUESTIONS.append(gen_comb(idx, n, k)); idx += 1
for a,b,x in [(2,10,5),(3,12,7),(4,20,9),(5,25,6),(6,18,7),(7,21,9),(8,24,10),(9,27,6),(10,30,7),(3,15,5),(4,16,7),(5,35,8)]:
    QUESTIONS.append(gen_prop(idx, a, b, x)); idx += 1

KEYS = [q["key"] for q in QUESTIONS]
KEY2IDX = {k:i for i,k in enumerate(KEYS)}

# ---------------------------------------------------------------
# Estado (calcado del patrón Big Five)
# ---------------------------------------------------------------
if "stage" not in st.session_state: st.session_state.stage = "inicio"  # inicio | test | resultados
if "q_idx" not in st.session_state: st.session_state.q_idx = 0
if "answers" not in st.session_state: st.session_state.answers = {k: None for k in KEYS}  # guarda letra A-E
if "fecha" not in st.session_state: st.session_state.fecha = None
if "_needs_rerun" not in st.session_state: st.session_state._needs_rerun = False

# ---------------------------------------------------------------
# Callbacks (auto-avance)
# ---------------------------------------------------------------
def on_answer_change(qkey:str):
    val = st.session_state.get(f"resp_{qkey}")  # p.ej. "A: 42"
    letra = None
    if isinstance(val, str) and ":" in val:
        letra = val.split(":",1)[0].strip()
    elif isinstance(val, str) and len(val)==1:
        letra = val
    if letra:
        st.session_state.answers[qkey] = letra
        idx = KEY2IDX[qkey]
        if idx < len(QUESTIONS)-1:
            st.session_state.q_idx = idx + 1
        else:
            st.session_state.stage = "resultados"
            st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state._needs_rerun = True

# ---------------------------------------------------------------
# Scoring y reportes
# ---------------------------------------------------------------
def compute_scores(answers:dict)->dict:
    total = len(QUESTIONS)
    aciertos = 0
    errores = 0
    por_comp = {c:{"aciertos":0,"total":0} for c in COMPETENCIAS}
    for q in QUESTIONS:
        por_comp[q["competencia"]]["total"] += 1
        sel = answers.get(q["key"])
        if sel is None:
            continue
        if sel == q["correcta"]:
            aciertos += 1
            por_comp[q["competencia"]]["aciertos"] += 1
        else:
            errores += 1
    omitidas = total - (aciertos + errores)
    porcentaje = round(100*aciertos/total, 1)
    return {
        "total": total,
        "aciertos": aciertos,
        "errores": errores,
        "omitidas": omitidas,
        "porcentaje": porcentaje,
        "por_comp": por_comp,
    }

# ---------------------------------------------------------------
# Gráfico simple por competencia + gauge de puntaje
# ---------------------------------------------------------------
def plot_competencias(por_comp:dict):
    labels = []
    values = []
    for k,v in por_comp.items():
        labels.append(k)
        pct = 0 if v["total"]==0 else round(100*v["aciertos"]/v["total"],1)
        values.append(pct)
    fig = go.Figure(go.Bar(x=labels, y=values, text=[f"{x}%" for x in values], textposition="outside"))
    fig.update_layout(template="plotly_white", height=420, yaxis=dict(range=[0,105], title="% acierto"), xaxis=dict(title="Competencia"))
    return fig

def plot_gauge_percent(percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'thickness': 0.35},
            'steps': [
                {'range': [0,60], 'color': '#ffe5e5'},
                {'range': [60,80], 'color': '#fff5cc'},
                {'range': [80,100], 'color': '#e6ffea'},
            ],
        },
    ))
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=30,r=30,t=20,b=20))
    return fig

# ---------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------

def view_inicio():
    st.markdown(
        """
        <div class="card">
          <h1 style="margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;">
            📐 Test de Lógica Matemática — Analista
          </h1>
          <p class="tag" style="margin:0;">Fondo blanco · Texto negro · Diseño profesional y responsivo</p>
        </div>
        """, unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.35,1])
    with c1:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">¿Qué mide?</h3>
              <ul style="line-height:1.6">
                <li><b>Razonamiento cuantitativo</b>: series y patrones numéricos.</li>
                <li><b>Cálculo aplicado</b>: porcentajes, descuentos, reglas simples.</li>
                <li><b>Análisis de datos</b>: tablas pequeñas, promedios.</li>
                <li><b>Lógica combinatoria</b>: conteos sin orden.</li>
                <li><b>Proporciones</b>: regla de tres y razón.</li>
              </ul>
              <p class="small">60 ítems · Autoavance · Duración estimada: <b>20–30 min</b>.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3 style="margin-top:0">Cómo funciona</h3>
              <ol style="line-height:1.6">
                <li>Ves 1 pregunta por pantalla y eliges una opción (A–E).</li>
                <li>Al elegir, avanzas automáticamente a la siguiente.</li>
                <li>Resultados con KPIs, tabla, desglose por competencia y descarga CSV.</li>
              </ol>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🚀 Iniciar evaluación", type="primary", use_container_width=True):
            st.session_state.stage = "test"
            st.session_state.q_idx = 0
            st.session_state.answers = {k: None for k in KEYS}
            st.session_state.fecha = None
            st.rerun()


def view_test():
    i = st.session_state.q_idx
    q = QUESTIONS[i]
    p = (i+1)/len(QUESTIONS)
    st.progress(p, text=f"Progreso: {i+1}/{len(QUESTIONS)}")
    st.markdown(f"<div class='dim-title'>🧮 Lógica Matemática</div>", unsafe_allow_html=True)
    st.markdown("<div class='dim-desc'>Selecciona la alternativa correcta.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### {i+1}. {q['texto']}")

    # Radio estilo Big Five: opciones tipo "A: valor"
    letras = ["A","B","C","D","E"]
    opciones = [f"{letras[j]}: {q['alternativas'][j]}" for j in range(len(q['alternativas']))]

    prev = st.session_state.answers.get(q["key"])  # letra
    prev_idx = None
    if prev in letras:
        prev_idx = letras.index(prev)

    st.radio(
        "Selecciona una opción",
        options=opciones,
        index=prev_idx,
        key=f"resp_{q['key']}",
        label_visibility="collapsed",
        on_change=on_answer_change,
        args=(q["key"],),
        horizontal=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def view_resultados():
    res = compute_scores(st.session_state.answers)

    st.markdown(
        f"""
        <div class=\"card\">
          <h1 style=\"margin:0 0 6px 0; font-size:clamp(2.2rem,3.8vw,3rem); font-weight:900;\">📊 Informe Lógica Matemática — Resultados</h1>
          <p class=\"small\" style=\"margin:0;\">Fecha: <b>{st.session_state.fecha}</b></p>
        </div>
        """, unsafe_allow_html=True
    )

    # KPIs + Gauge de puntaje
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Total de ítems</div><div class='value'>{res['total']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Aciertos</div><div class='value'>{res['aciertos']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Errores</div><div class='value'>{res['errores']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Omitidas</div><div class='value'>{res['omitidas']}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.subheader("Puntaje total (%)")
        st.plotly_chart(plot_gauge_percent(res['porcentaje']), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 % de acierto por competencia")
    st.plotly_chart(plot_competencias(res["por_comp"]), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detalle de respuestas")
    rows = []
    for q in QUESTIONS:
        sel = st.session_state.answers.get(q["key"]) or "—"
        ok = (sel == q["correcta"]) if sel != "—" else False
        rows.append({
            "ID": q["key"],
            "Competencia": q["competencia"],
            "Pregunta": q["texto"],
            "Respuesta": sel,
            "Correcta": q["correcta"],
            "Acierto": int(ok),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Descarga CSV
    from io import StringIO
    csv_io = StringIO(); df.to_csv(csv_io, index=False)
    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=csv_io.getvalue(),
        file_name="Resultados_Logica_Analista.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

    # Botón Imprimir/Guardar como PDF (usa diálogo del navegador)
    st.markdown("""
        <div style='display:flex; gap:8px; margin-top:8px;'>
          <button onclick='window.print()' style='padding:10px 14px; border:1px solid #ddd; border-radius:10px; background:#111; color:#fff; cursor:pointer;'>🖨️ Imprimir / Guardar PDF</button>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Nueva evaluación", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {k: None for k in KEYS}
        st.session_state.fecha = None
        st.rerun()"🔄 Nueva evaluación", type="primary", use_container_width=True):
        st.session_state.stage = "inicio"
        st.session_state.q_idx = 0
        st.session_state.answers = {k: None for k in KEYS}
        st.session_state.fecha = None
        st.rerun()

# ---------------------------------------------------------------
# FLUJO PRINCIPAL (calcado al Big Five)
# ---------------------------------------------------------------
if st.session_state.stage == "inicio":
    view_inicio()
elif st.session_state.stage == "test":
    view_test()
else:
    if st.session_state.fecha is None:
        st.session_state.fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    view_resultados()

# Rerun único si el callback lo marcó
if st.session_state._needs_rerun:
    st.session_state._needs_rerun = False
    st.rerun()
