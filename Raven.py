# test_logica_analista.py
# ---------------------------------------------
# App de evaluación: Test de Lógica Matemática – Analista
# Estructura inspirada en app Big Five (inicio → test_activo → resultados)
# - StateManager con initialize()
# - Navegación por st.session_state.stage y flag navigation_flag
# - Auto-avance al elegir alternativa
# - Forzar scroll al top en cada cambio de pregunta
# - Cronómetro y límite de tiempo
# - Scoring y desglose por competencias con gráfico Plotly
# - Exportación CSV
# ---------------------------------------------

from __future__ import annotations
import time
from datetime import timedelta
import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, List
import io

# ----------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Test Lógica Matemática - Analista",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# CONSTANTES
# ----------------------------
N_PREGUNTAS = 60
TIEMPO_LIMITE_MIN = 35  # límite total del test

# ----------------------------
# UTILIDADES DE ESTILO
# ----------------------------
CSS_BASE = """
<style>
/* Títulos y tipografía limpia */
h1, h2, h3, h4 { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }

/***** Contenedores *****/
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

/* Cartas livianas */
.card {
  background: #ffffff;
  border: 1px solid #e9ecef;
  border-radius: 14px;
  padding: 1rem 1.2rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

/* Botones de alternativas */
.alt-btn {
  width: 100%;
  padding: 0.9rem 0.8rem;
  border-radius: 12px;
  border: 1px solid #dfe3e6;
  background: #fff;
  cursor: pointer;
}
.alt-btn:hover { background: #f7f9fb; }
.alt-btn.selected { background: #ecf3ff; border-color: #74a7ff; }

/* Barra de progreso simple */
.progress-wrap{width:100%;height:10px;background:#f1f3f5;border-radius:999px;overflow:hidden}
.progress-bar{height:100%;background:#3b82f6}

/* Tabla de resultados */
.table-compact td, .table-compact th { padding: 0.35rem 0.5rem; font-size: 0.93rem; }
</style>
"""

st.markdown(CSS_BASE, unsafe_allow_html=True)

# ----------------------------
# STATE MANAGER
# ----------------------------
class StateManager:
    @staticmethod
    def initialize():
        if 'stage' not in st.session_state:
            st.session_state.stage = 'inicio'  # 'inicio' | 'test_activo' | 'resultados'
        if 'navigation_flag' not in st.session_state:
            st.session_state.navigation_flag = False
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0  # 0-based
        if 'answers' not in st.session_state:
            st.session_state.answers: Dict[str, str] = {}
        if 'started_at' not in st.session_state:
            st.session_state.started_at = None
        if 'finished_at' not in st.session_state:
            st.session_state.finished_at = None
        if 'user_info' not in st.session_state:
            st.session_state.user_info = {'nombre': '', 'email': ''}
        if 'scroll_key' not in st.session_state:
            st.session_state.scroll_key = 0
        if 'questions' not in st.session_state:
            st.session_state.questions = build_questions()
        if 'time_limit_min' not in st.session_state:
            st.session_state.time_limit_min = TIEMPO_LIMITE_MIN

# ----------------------------
# FORZAR SCROLL AL TOPE
# ----------------------------
try:
    # Streamlit >=1.32 suele exponer components.v1.html como import directo
    from streamlit.components.v1 import html as _comp_html
except Exception:
    from streamlit import components as _components
    _comp_html = _components.v1.html


def forzar_scroll_al_top():
    """Scroll al tope inyectando <script> vía markdown para máxima compatibilidad.
    Evita streamlit.components y usa un placeholder dinámico.
    """
    script = (
        "<script>
"
        "  setTimeout(function(){
"
        "    try {
"
        "      var root = window.parent || window;
"
        "      if (root && root.scrollTo) { root.scrollTo({top:0, behavior:'auto'}); }
"
        "      var c = root && root.document ? root.document.querySelector('[data-testid=\\"stAppViewContainer\\"]') : null;
"
        "      if (c && c.scrollTo) { c.scrollTo({top:0, behavior:'auto'}); }
"
        "    } catch(e) { }
"
        "  }, 60);
"
        "</script>"
    )
    # Placeholder para re-inyectar el script en cada paso
    if 'scroll_placeholder' not in st.session_state:
        st.session_state.scroll_placeholder = st.empty()
    st.session_state.scroll_placeholder.markdown(script, unsafe_allow_html=True)

# ----------------------------
# GENERACIÓN DE PREGUNTAS
# ----------------------------
# Cada pregunta: {id, competencia, dificultad, enunciado, alternativas[list], correcta(letter), explicacion}

COMP_Q = {
    'series': 'Razonamiento cuantitativo',
    'arit': 'Cálculo aplicado',
    'tabla': 'Análisis de datos',
    'comb': 'Lógica combinatoria',
    'prop': 'Proporciones y razón'
}


def series_item(idx: int, base: int, step: int, n_alts: int = 5):
    seq = [base + step * k for k in range(6)]
    correcto = seq[-1] + step
    alternativas = sorted({correcto, correcto+step, correcto-step, correcto+2*step, correcto//2 if correcto%2==0 else correcto+3})
    alternativas = alternativas[:n_alts]
    letras = ['A','B','C','D','E'][:len(alternativas)]
    correcta = letras[alternativas.index(correcto)]
    return {
        'id': f'LM{idx}',
        'competencia': COMP_Q['series'],
        'dificultad': 'facil' if step <= 5 else 'media',
        'enunciado': f"Completa la serie: {', '.join(map(str, seq))}, ¿cuál sigue?",
        'alternativas': [str(x) for x in alternativas],
        'correcta': correcta,
        'explicacion': f"La serie aumenta de {step} en {step}. El siguiente es {correcto}."
    }


def arit_item(idx: int, precio: int, pct: int):
    inc = round(precio * pct/100)
    total = precio + inc
    alts = sorted({total, total-10, total+10, total-5, total+5})
    letras = ['A','B','C','D','E']
    correcta = letras[alts.index(total)]
    return {
        'id': f'LM{idx}',
        'competencia': COMP_Q['arit'],
        'dificultad': 'facil' if pct<=20 else 'media',
        'enunciado': f"Un artículo cuesta ${precio}. Aumenta {pct}% ¿Cuál es el nuevo precio?",
        'alternativas': [f"${a}" for a in alts],
        'correcta': correcta,
        'explicacion': f"{precio} x {pct}% = {inc}; {precio}+{inc} = {total}."
    }


def tabla_item(idx: int):
    # Mini tabla implícita en texto
    # Ventas por canal: A=120, B=150, C=90, D=140
    datos = {'A':120,'B':150,'C':90,'D':140}
    pregunta = "Según la tabla (A=120, B=150, C=90, D=140), ¿cuál es el promedio de ventas?"
    prom = round(sum(datos.values())/len(datos))
    alts = sorted({prom, prom+5, prom-5, prom+10, prom-10})
    letras = ['A','B','C','D','E']
    correcta = letras[alts.index(prom)]
    return {
        'id': f'LM{idx}',
        'competencia': COMP_Q['tabla'],
        'dificultad': 'media',
        'enunciado': pregunta,
        'alternativas': [str(a) for a in alts],
        'correcta': correcta,
        'explicacion': f"Promedio = (120+150+90+140)/4 = {prom}."
    }


def comb_item(idx: int, n: int, k: int):
    # Combinatoria simple: de n personas, formar comités de k (sin orden): C(n,k)
    from math import comb
    correcto = comb(n,k)
    alts = sorted({correcto, correcto+1, max(1,correcto-1), correcto+2, max(1,correcto-2)})
    letras = ['A','B','C','D','E']
    correcta = letras[alts.index(correcto)]
    return {
        'id': f'LM{idx}',
        'competencia': COMP_Q['comb'],
        'dificultad': 'media' if n<=8 else 'dificil',
        'enunciado': f"¿Cuántos comités distintos de {k} personas pueden formarse a partir de {n} candidatos? (sin orden)",
        'alternativas': [str(a) for a in alts],
        'correcta': correcta,
        'explicacion': f"Se usa C(n,k) = n!/(k!(n-k)!). Aquí C({n},{k}) = {correcto}."
    }


def prop_item(idx: int, a: int, b: int, x: int):
    # Regla de tres: a/b = x/y → y = (b*x)/a
    from math import isclose
    y = int(round((b*x)/a))
    alts = sorted({y, y+1, max(1,y-1), y+2, max(1,y-2)})
    letras = ['A','B','C','D','E']
    correcta = letras[alts.index(y)]
    return {
        'id': f'LM{idx}',
        'competencia': COMP_Q['prop'],
        'dificultad': 'facil' if a<=10 and b<=50 else 'media',
        'enunciado': f"Si {a} es a {b} como {x} es a ¿y?",
        'alternativas': [str(v) for v in alts],
        'correcta': correcta,
        'explicacion': f"y = (b*x)/a = ({b}*{x})/{a} = {y}."
    }


def build_questions() -> List[Dict]:
    qs: List[Dict] = []
    idx = 1
    # 1) Series numéricas (12)
    series_params = [(2,3),(5,4),(10,2),(3,5),(7,6),(1,7),(4,4),(8,3),(6,5),(9,4),(11,2),(12,3)]
    for base, step in series_params:
        qs.append(series_item(idx, base, step))
        idx += 1
    # 2) Aritmética aplicada (12)
    arit_params = [(100,10),(250,15),(400,20),(120,5),(300,12),(180,18),(90,25),(220,8),(560,7),(800,9),(150,30),(360,22)]
    for precio, pct in arit_params:
        qs.append(arit_item(idx, precio, pct))
        idx += 1
    # 3) Tablas/Promedios (12 variaciones)
    for _ in range(12):
        qs.append(tabla_item(idx))
        idx += 1
    # 4) Combinatoria simple (12)
    comb_params = [(6,2),(7,3),(8,2),(9,3),(10,2),(7,4),(8,3),(9,4),(10,3),(6,3),(8,4),(10,4)]
    for n,k in comb_params:
        qs.append(comb_item(idx, n, k))
        idx += 1
    # 5) Proporciones (12)
    prop_params = [(2,10,5),(3,12,7),(4,20,9),(5,25,6),(6,18,7),(7,21,9),(8,24,10),(9,27,6),(10,30,7),(3,15,5),(4,16,7),(5,35,8)]
    for a,b,x in prop_params:
        qs.append(prop_item(idx, a, b, x))
        idx += 1
    # Asegura N_PREGUNTAS
    return qs[:N_PREGUNTAS]

# ----------------------------
# CRONÓMETRO Y TIEMPO
# ----------------------------

def segundos_transcurridos() -> int:
    if st.session_state.started_at is None:
        return 0
    return int(time.time() - st.session_state.started_at)


def formatear_tiempo(seg: int) -> str:
    return str(timedelta(seconds=seg))[:-3]  # mm:ss (o hh:mm:ss)


def excede_limite() -> bool:
    limite = st.session_state.time_limit_min * 60
    return segundos_transcurridos() >= limite

# ----------------------------
# RENDER DE PROGRESO
# ----------------------------

def render_progreso():
    total = len(st.session_state.questions)
    idx = st.session_state.current_index + 1
    pct = int((idx/total)*100)
    st.markdown(
        f"**Pregunta {idx} de {total}** — Progreso: {pct}%"
    )
    st.markdown(
        f'<div class="progress-wrap"><div class="progress-bar" style="width:{pct}%"></div></div>',
        unsafe_allow_html=True
    )

# ----------------------------
# NAVEGACIÓN
# ----------------------------

def avanzar():
    total = len(st.session_state.questions)
    if st.session_state.current_index < total - 1:
        st.session_state.current_index += 1
        st.session_state.navigation_flag = True
    else:
        st.session_state.stage = 'resultados'
        st.session_state.finished_at = time.time()
        st.session_state.navigation_flag = True


def retroceder():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.navigation_flag = True

# ----------------------------
# SCORING
# ----------------------------

def calcular_score(answers: Dict[str,str], questions: List[Dict]):
    total = len(questions)
    aciertos = 0
    por_competencia = {}
    for q in questions:
        comp = q['competencia']
        por_competencia.setdefault(comp, {'aciertos':0,'total':0})
        por_competencia[comp]['total'] += 1
        rid = q['id']
        if rid in answers and answers[rid] == q['correcta']:
            aciertos += 1
            por_competencia[comp]['aciertos'] += 1
    errores = sum(1 for q in questions if q['id'] in answers and answers[q['id']] != q['correcta'])
    omitidas = total - (aciertos + errores)
    porcentaje = round(100 * aciertos / total, 1)
    tiempo_total = 0
    if st.session_state.started_at is not None:
        fin = st.session_state.finished_at or time.time()
        tiempo_total = int(fin - st.session_state.started_at)
    return {
        'total': total,
        'aciertos': aciertos,
        'errores': errores,
        'omitidas': omitidas,
        'porcentaje': porcentaje,
        'por_competencia': por_competencia,
        'tiempo_total': tiempo_total,
    }

# ----------------------------
# VISTAS
# ----------------------------

def vista_inicio():
    st.title("Test de Lógica Matemática – Analista")
    st.write("""
    Evaluación breve de razonamiento cuantitativo aplicada a un rol de Analista. 
    Instrucciones:
    - Selecciona una alternativa por pregunta.
    - Avanza automáticamente al elegir (puedes usar Anterior/Siguiente).
    - Tiempo límite: **{} minutos**.
    - Al finalizar verás tu resumen y podrás descargar un CSV con tus respuestas.
    """.format(st.session_state.time_limit_min))

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_info['nombre'] = st.text_input("Nombre (opcional)", value=st.session_state.user_info.get('nombre',''))
        with col2:
            st.session_state.user_info['email'] = st.text_input("Email (opcional)", value=st.session_state.user_info.get('email',''))

        st.markdown("---")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Comenzar Test", type="primary"):
                st.session_state.answers = {}
                st.session_state.current_index = 0
                st.session_state.started_at = time.time()
                st.session_state.finished_at = None
                st.session_state.stage = 'test_activo'
                st.session_state.navigation_flag = True
        with c2:
            if st.session_state.answers:
                if st.button("Continuar donde quedé"):
                    st.session_state.stage = 'test_activo'
                    st.session_state.navigation_flag = True
        st.markdown('</div>', unsafe_allow_html=True)



def render_pregunta_old(q: Dict):
    # Progreso + Cronómetro
    render_progreso()
    trans = segundos_transcurridos()
    limite_seg = st.session_state.time_limit_min * 60
    restante = max(0, limite_seg - trans)
    st.caption(f"Tiempo transcurrido: {formatear_tiempo(trans)}  ·  Restante: {formatear_tiempo(restante)}")

    if excede_limite():
        st.warning("Se alcanzó el tiempo límite. Pasando a resultados…")
        st.session_state.stage = 'resultados'
        st.session_state.finished_at = time.time()
        st.session_state.navigation_flag = True
        return

    # Card de pregunta
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(q['id'] + "  ·  " + q['competencia'])
    st.write(f"**Dificultad:** {q['dificultad'].title()}")
    st.markdown("---")
    st.write(q['enunciado'])
    st.markdown("---")

    # Alternativas como botones para auto-avance
    alternativas = q['alternativas']
    letras = ['A','B','C','D','E'][:len(alternativas)]
    current_sel = st.session_state.answers.get(q['id'])  # letra seleccionada si existe

    cols = st.columns(len(alternativas)) if len(alternativas) <= 5 else st.columns(5)
    def click_choice(letter: str):
        st.session_state.answers[q['id']] = letter
        forzar_scroll_al_top()
        avanzar()

    for i, letra in enumerate(letras):
        label = f"{letra}"
        body = alternativas[i]
        is_sel = (current_sel == letra)
        with cols[i % len(cols)]:
            if st.button(label, key=f"btn_{q['id']}_{letra}"):
                click_choice(letra)
            st.write(body)
            if is_sel:
                st.caption("Seleccionada")

    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("⟵ Anterior", use_container_width=True):
            forzar_scroll_al_top()
            retroceder()
    with c2:
        if st.button("Siguiente ⟶", use_container_width=True):
            forzar_scroll_al_top()
            avanzar()



def render_pregunta(q: Dict):
    """Render de una pregunta con **estructura visual igual** al Big Five:
    - Encabezado compacto (competencia + dificultad)
    - Enunciado
    - Selector tipo `st.radio` (como el Likert del Big Five) con autoavance
    - Progreso y cronómetro arriba
    - Botonera inferior igual (Anterior / Siguiente)
    """
    # Progreso + Cronómetro (mismo bloque superior que Big Five)
    render_progreso()
    trans = segundos_transcurridos()
    limite_seg = st.session_state.time_limit_min * 60
    restante = max(0, limite_seg - trans)
    st.caption(f"Tiempo transcurrido: {formatear_tiempo(trans)}  ·  Restante: {formatear_tiempo(restante)}")

    if excede_limite():
        st.warning("Se alcanzó el tiempo límite. Pasando a resultados…")
        st.session_state.stage = 'resultados'
        st.session_state.finished_at = time.time()
        st.session_state.navigation_flag = True
        return

    # Card principal (idéntica estructura visual)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(q['id'] + "  ·  " + q['competencia'])
    st.write(f"**Dificultad:** {q['dificultad'].title()}")
    st.markdown("---")
    st.write(q['enunciado'])
    st.markdown("---")

    # === Selector estilo Big Five (radio) ===
    alternativas = q['alternativas']
    letras = ['A','B','C','D','E'][:len(alternativas)]
    opciones_radio = [f"{letras[i]}: {alternativas[i]}" for i in range(len(alternativas))]

    # Selección previa
    prev_letra = st.session_state.answers.get(q['id'])
    if prev_letra in letras:
        default_index = letras.index(prev_letra)
    else:
        default_index = None

    def _on_change():
        value = st.session_state.get(f"radio_{q['id']}")
        sel_letter = value.split(":",1)[0].strip() if isinstance(value, str) else None
        if sel_letter:
            st.session_state.answers[q['id']] = sel_letter
            forzar_scroll_al_top()
            avanzar()

    st.radio(
        label="Selecciona una alternativa",
        options=opciones_radio,
        index=default_index,
        key=f"radio_{q['id']}",
        on_change=_on_change,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Botonera inferior igual al Big Five
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("⟵ Anterior", use_container_width=True):
            forzar_scroll_al_top()
            retroceder()
    with c2:
        if st.button("Siguiente ⟶", use_container_width=True):
            forzar_scroll_al_top()
            avanzar()


def vista_test_activo():
    total = len(st.session_state.questions)
    idx = st.session_state.current_index
    q = st.session_state.questions[idx]
    forzar_scroll_al_top()
    render_pregunta(q)



def vista_resultados():
    st.title("Resultados del Test")
    resultados = calcular_score(st.session_state.answers, st.session_state.questions)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Aciertos", resultados['aciertos'])
    colB.metric("Errores", resultados['errores'])
    colC.metric("Omitidas", resultados['omitidas'])
    colD.metric("Puntaje %", resultados['porcentaje'])

    st.markdown("---")
    st.subheader("Desglose por competencia")
    comp_rows = []
    for comp, vals in resultados['por_competencia'].items():
        tasa = round(100*vals['aciertos']/vals['total'],1)
        comp_rows.append({'competencia': comp, 'aciertos': vals['aciertos'], 'total': vals['total'], '% acierto': tasa})
    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp, use_container_width=True)
    if not df_comp.empty:
        fig = px.bar(df_comp, x='competencia', y='% acierto', title='Tasa de acierto por competencia')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Detalle de respuestas")
    detalle = []
    for q in st.session_state.questions:
        rid = q['id']
        sel = st.session_state.answers.get(rid, '')
        ok = (sel == q['correcta']) if sel else False
        detalle.append({
            'id': rid,
            'competencia': q['competencia'],
            'dificultad': q['dificultad'],
            'respuesta_usuario': sel or '—',
            'respuesta_correcta': q['correcta'],
            'acierto': 1 if ok else 0,
            'explicacion': q['explicacion']
        })
    df_det = pd.DataFrame(detalle)
    st.dataframe(df_det.drop(columns=['explicacion']), use_container_width=True)

    # Exportar CSV (incluye explicación para revisión)
    csv_buf = io.StringIO()
    df_det.to_csv(csv_buf, index=False)
    st.download_button("Descargar resultados (CSV)", data=csv_buf.getvalue(), file_name="resultados_test_logica.csv", mime="text/csv")

    # Recomendación simple
    st.markdown("---")
    st.subheader("Recomendaciones")
    if resultados['porcentaje'] >= 80:
        st.success("Desempeño sobresaliente. Recomendado para tareas analíticas complejas y toma de decisiones cuantitativas.")
    elif resultados['porcentaje'] >= 60:
        st.info("Buen desempeño. Recomendable con refuerzo en áreas específicas identificadas en el desglose.")
    else:
        st.warning("Desempeño por debajo del esperado. Sugerido entrenamiento focalizado en fundamentos de cálculo y análisis de datos.")

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Volver al inicio", use_container_width=True):
            st.session_state.stage = 'inicio'
            st.session_state.navigation_flag = True
    with c2:
        if st.button("Reiniciar test", use_container_width=True):
            st.session_state.answers = {}
            st.session_state.current_index = 0
            st.session_state.started_at = time.time()
            st.session_state.finished_at = None
            st.session_state.stage = 'test_activo'
            st.session_state.navigation_flag = True


# ----------------------------
# MAIN
# ----------------------------

def main():
    StateManager.initialize()

    # Manejo de navegación centralizada
    if st.session_state.navigation_flag:
        st.session_state.navigation_flag = False
        st.rerun()

    # Sidebar con info del test
    with st.sidebar:
        st.header("Configuración")
        st.write(f"Preguntas: {len(st.session_state.questions)}")
        st.write(f"Tiempo límite: {st.session_state.time_limit_min} min")
        if st.session_state.started_at:
            st.caption(f"Transcurrido: {formatear_tiempo(segundos_transcurridos())}")
        st.markdown("---")
        st.caption("Test de Lógica Matemática – Analista · Demo educativa")

    if st.session_state.stage == 'inicio':
        vista_inicio()
    elif st.session_state.stage == 'test_activo':
        vista_test_activo()
    elif st.session_state.stage == 'resultados':
        vista_resultados()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align:center; font-size: 12px; color: grey;'>
        🧠 Test Lógica Matemática – Analista (Demo) · Los resultados son orientativos y no constituyen diagnóstico oficial.<br>
        © 2025 – Desarrollado en Streamlit | Arquitectura modular profesional
        </p>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
