# app.py
# ---------------------------------------------------------
# Test de Raven - App liviana, profesional y escalable
# Autor: tú :)
# Requisitos: streamlit, pillow, requests, pandas
# pip install streamlit pillow requests pandas
# ---------------------------------------------------------

import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Test de Raven - Demo",
    layout="centered",
    page_icon="🧠",
    initial_sidebar_state="collapsed",
)

# ---------------------- Configuración ----------------------

TOTAL_ITEMS = 60
OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Genera URLs RAW directas a tu repo (no uses /blob/)
def build_image_url(i: int) -> str:
    return f"https://raw.githubusercontent.com/legendss7/Test-Raven-Demo/main/raven_pagina_{i:03}.png"

# Cachea la descarga de imágenes por URL
@st.cache_data(show_spinner=False, max_entries=256)
def fetch_image_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.content

# Utilidad: forzar scroll al top tras cada avance
def scroll_to_top():
    # NOTA: evitar f-strings para no chocar con llaves JS
    js_code = """
    <script>
        setTimeout(function(){
            try {
                window.parent.scrollTo({ top: 0, behavior: 'auto' });
                var mainContent = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (mainContent) {
                    mainContent.scrollTo({ top: 0, behavior: 'auto' });
                }
            } catch(e) {}
        }, 120);
    </script>
    """
    st.components.v1.html(js_code, height=0, scrolling=False)

# ---------------------- Estado Global ----------------------

if "stage" not in st.session_state:
    st.session_state.stage = "inicio"  # inicio | test | resultados

if "idx" not in st.session_state:
    st.session_state.idx = 0  # 0..59

if "responses" not in st.session_state:
    # dict: { index (0-based): "A".."H" }
    st.session_state.responses = {}

if "start_ts" not in st.session_state:
    st.session_state.start_ts = None

# ---------------------- Callbacks ----------------------

def cb_select_option():
    """Se dispara al elegir alternativa: guarda => avanza => resultados al final."""
    i = st.session_state.idx
    key = f"resp_{i}"
    selected = st.session_state.get(key, None)
    if selected is not None:
        st.session_state.responses[i] = selected

        # Avanzar
        if i < TOTAL_ITEMS - 1:
            st.session_state.idx = i + 1
            scroll_to_top()
            st.rerun()
        else:
            st.session_state.stage = "resultados"
            st.rerun()

def cb_start_test():
    st.session_state.stage = "test"
    st.session_state.idx = 0
    st.session_state.responses = {}
    st.session_state.start_ts = datetime.now().isoformat()
    st.rerun()

def cb_restart():
    for k in list(st.session_state.keys()):
        if k.startswith("resp_"):
            del st.session_state[k]
    st.session_state.idx = 0
    st.session_state.responses = {}
    st.session_state.stage = "inicio"
    st.rerun()

# ---------------------- Vistas ----------------------

def vista_inicio():
    st.title("🧠 Test de Matrices Progresivas (Raven) – Demo")
    st.markdown(
        """
**Objetivo**: presentar ítems tipo matriz (1 a 60), registrar respuesta por alternativa **A–H** y entregar un resumen final.

**Notas importantes**  
- Las imágenes se leen **directamente** desde tu repositorio GitHub (raw) para máxima ligereza.  
- Para un resultado “100% confiable” necesitas **cargar la clave de respuestas** (CSV) que **tú** provees.  
- Esta app **no incluye** la clave ni normas por temas de derechos.  
- Puedes exportar las respuestas en CSV y calcular métricas adicionales si lo deseas.
        """.strip()
    )
    st.button("Comenzar", type="primary", use_container_width=True, on_click=cb_start_test)

    with st.expander("⚙️ Configuración técnica"):
        st.code(
            "Repo esperado: https://github.com/legendss7/Test-Raven-Demo\n"
            "Imágenes: raven_pagina_001.png ... raven_pagina_060.png\n"
            "URL RAW: https://raw.githubusercontent.com/legendss7/Test-Raven-Demo/main/raven_pagina_001.png",
            language="text"
        )


def vista_test():
    i = st.session_state.idx
    n_display = i + 1
    progress = (n_display) / TOTAL_ITEMS

    st.markdown(f"#### Ítem {n_display} de {TOTAL_ITEMS}")
    st.progress(progress)

    # Cargar imagen actual (on-demand)
    url = build_image_url(n_display)
    try:
        img_bytes = fetch_image_bytes(url)
        img = Image.open(BytesIO(img_bytes))
        st.image(img, use_container_width=True, caption=f"Ítem {n_display}")
    except Exception as e:
        st.error(f"No se pudo cargar la imagen del ítem {n_display}. Verifica el nombre/URL.\n{e}")

    st.markdown("**Selecciona la alternativa correcta:**")

    # Valor actual si ya respondió
    prefill = st.session_state.responses.get(i, None)

    # Radio con callback de auto-avance
    st.radio(
        label="Alternativas",
        options=OPTIONS,
        index=OPTIONS.index(prefill) if prefill in OPTIONS else None,
        horizontal=True,
        key=f"resp_{i}",
        on_change=cb_select_option,
    )

    # Ayudas visuales
    with st.expander("Ver respuestas registradas", expanded=False):
        if st.session_state.responses:
            df_prev = pd.DataFrame(
                [{"Ítem": k + 1, "Respuesta": v} for k, v in sorted(st.session_state.responses.items())]
            )
            st.dataframe(df_prev, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay respuestas registradas.")

    st.caption("Al elegir una alternativa, pasarás automáticamente al siguiente ítem.")

def vista_resultados():
    st.success("¡Has completado el test!")
    end_ts = datetime.now().isoformat()

    # Construir DataFrame de respuestas
    data = []
    for i in range(TOTAL_ITEMS):
        data.append({
            "item": i + 1,
            "respuesta": st.session_state.responses.get(i, None)
        })
    df_resp = pd.DataFrame(data)

    st.subheader("Resumen de respuestas")
    st.dataframe(df_resp, use_container_width=True, hide_index=True)

    # Carga opcional de clave de corrección (CSV)
    st.markdown("### ✅ (Opcional) Cargar clave de respuestas")
    st.caption(
        "Sube un CSV con columnas: **item** (1..60) y **correcta** (A..H). "
        "Ejemplo de filas: `1,A` `2,D` …"
    )
    key_file = st.file_uploader("Clave de respuestas (CSV)", type=["csv"])

    total_correct = None
    if key_file is not None:
        try:
            key_df = pd.read_csv(key_file)
            # Normalización básica
            key_df["item"] = key_df["item"].astype(int)
            key_df["correcta"] = key_df["correcta"].str.strip().str.upper()

            merged = df_resp.merge(key_df, on="item", how="left")
            merged["correcto"] = merged["respuesta"].str.upper() == merged["correcta"]
            total_correct = int(merged["correcto"].sum())

            st.subheader("Resultados")
            st.metric(label="Aciertos (puntaje directo)", value=f"{total_correct} / {TOTAL_ITEMS}")

            with st.expander("Detalle de corrección"):
                st.dataframe(merged, use_container_width=True, hide_index=True)

            st.caption(
                "Para percentiles o equivalencias normativas se requiere una tabla de normas por edad/forma (no incluida)."
            )
        except Exception as e:
            st.error(f"Error al procesar la clave: {e}")

    # Exportar CSV con respuestas
    st.markdown("### 📥 Exportar respuestas")
    csv_bytes = df_resp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar respuestas (CSV)",
        data=csv_bytes,
        file_name=f"raven_respuestas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Metadatos
    st.markdown("### Metadatos")
    st.json(
        {
            "inicio": st.session_state.start_ts,
            "fin": end_ts,
            "total_items": TOTAL_ITEMS,
            "total_respondidos": int(df_resp["respuesta"].notna().sum()),
            "aciertos": total_correct if total_correct is not None else "—"
        }
    )

    st.button("🔁 Reiniciar", use_container_width=True, on_click=cb_restart)

# ---------------------- Router ----------------------

if st.session_state.stage == "inicio":
    vista_inicio()
elif st.session_state.stage == "test":
    vista_test()
elif st.session_state.stage == "resultados":
    vista_resultados()
else:
    st.session_state.stage = "inicio"
    st.rerun()
