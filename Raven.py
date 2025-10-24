import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import time
import fitz  # PyMuPDF

# --- CONFIGURACIÓN Y DATOS DEL TEST ---
# Respuestas correctas del Test de Raven (Standard Progressive Matrices - SPM)
# Esto se basa en la estructura estándar A1-A12, B1-B12, C1-C12, D1-D12, E1-E12
# NOTA: Debes verificar la numeración de las opciones en tu PDF, ya que el OCR puede ser imperfecto.
RESPUESTAS_CORRECTAS = {
    # SERIE A (12 preguntas)
    'A1': 4, 'A2': 5, 'A3': 1, 'A4': 2, 'A5': 6, 'A6': 3, 'A7': 6, 'A8': 2, 'A9': 1, 'A10': 3, 'A11': 5, 'A12': 6,
    # SERIE B (12 preguntas)
    'B1': 2, 'B2': 6, 'B3': 5, 'B4': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 6, 'B9': 1, 'B10': 2, 'B11': 3, 'B12': 4,
    # SERIE C (12 preguntas)
    'C1': 8, 'C2': 2, 'C3': 3, 'C4': 8, 'C5': 7, 'C6': 3, 'C7': 5, 'C8': 6, 'C9': 4, 'C10': 3, 'C11': 7, 'C12': 2,
    # SERIE D (12 preguntas)
    'D1': 3, 'D2': 4, 'D3': 3, 'D4': 7, 'D5': 8, 'D6': 4, 'D7': 2, 'D8': 5, 'D9': 1, 'D10': 6, 'D11': 5, 'D12': 4,
    # SERIE E (12 preguntas)
    'E1': 7, 'E2': 6, 'E3': 8, 'E4': 2, 'E5': 1, 'E6': 5, 'E7': 6, 'E8': 7, 'E9': 4, 'E10': 3, 'E11': 8, 'E12': 1,
}

TODAS_PREGUNTAS = list(RESPUESTAS_CORRECTAS.keys())
TOTAL_PREGUNTAS = len(TODAS_PREGUNTAS)

# --- CONFIGURACIÓN DE PÁGINAS DEL PDF (Asumiendo que las imágenes se extraen por página) ---
# Mapea Pregunta (A1, A2...) a la Página del PDF (1-based index)
MAPA_PAGINAS = {
    'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 'A6': 6, 'A7': 7, 'A8': 8, 'A9': 9, 'A10': 10, 'A11': 11, 'A12': 12,
    'B1': 13, 'B2': 14, 'B3': 15, 'B4': 16, 'B5': 17, 'B6': 18, 'B7': 19, 'B8': 20, 'B9': 21, 'B10': 22, 'B11': 23, 'B12': 24,
    'C1': 25, 'C2': 26, 'C3': 27, 'C4': 28, 'C5': 29, 'C6': 30, 'C7': 31, 'C8': 32, 'C9': 33, 'C10': 34, 'C11': 35, 'C12': 36,
    'D1': 37, 'D2': 38, 'D3': 39, 'D4': 40, 'D5': 41, 'D6': 42, 'D7': 43, 'D8': 44, 'D9': 45, 'D10': 46, 'D11': 47, 'D12': 48,
    'E1': 49, 'E2': 50, 'E3': 51, 'E4': 52, 'E5': 53, 'E6': 54, 'E7': 55, 'E8': 56, 'E9': 57, 'E10': 58, 'E11': 59, 'E12': 60,
}

# La mayoría de las preguntas tienen 6 opciones (A1-A12 y B1-B12), C, D y E tienen 8.
OPCIONES_POR_PREGUNTA = {
    'A': 6, 'B': 6, 'C': 8, 'D': 8, 'E': 8
}

# --- FUNCIONES DE ESTADO Y FLUJO ---

def inicializar_sesion():
    """Inicializa variables de sesión si no existen."""
    if 'test_iniciado' not in st.session_state:
        st.session_state.test_iniciado = False
    if 'pregunta_actual_idx' not in st.session_state:
        st.session_state.pregunta_actual_idx = 0
    if 'respuestas' not in st.session_state:
        st.session_state.respuestas = {}
    if 'tiempo_inicio' not in st.session_state:
        st.session_state.tiempo_inicio = None
    if 'pdf_subido' not in st.session_state:
        st.session_state.pdf_subido = False
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None

def siguiente_pregunta(seleccion):
    """Guarda la respuesta y avanza a la siguiente pregunta."""
    pregunta_id = TODAS_PREGUNTAS[st.session_state.pregunta_actual_idx]
    st.session_state.respuestas[pregunta_id] = seleccion

    # Asegura la transición automática a la siguiente pregunta o al resultado
    if st.session_state.pregunta_actual_idx < TOTAL_PREGUNTAS - 1:
        st.session_state.pregunta_actual_idx += 1
    else:
        st.session_state.test_iniciado = 'finalizado'

def reiniciar_test():
    """Resetea el estado para comenzar de nuevo."""
    st.session_state.test_iniciado = False
    st.session_state.pregunta_actual_idx = 0
    st.session_state.respuestas = {}
    st.session_state.tiempo_inicio = None
    st.session_state.pdf_subido = False
    st.session_state.pdf_content = None

# --- MANEJO DE IMÁGENES DEL PDF CON PYMUPDF ---

@st.cache_resource
def obtener_imagen_pagina(pdf_content, pagina_numero):
    """Extrae la imagen de una página específica del PDF con PyMuPDF."""
    try:
        # Abrir el documento desde el contenido binario
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Número de página es 1-based, PyMuPDF es 0-based
        pagina = doc.load_page(pagina_numero - 1)
        
        # Crear un píxmap para renderizar la página
        # El parámetro 'dpi' (e.g., 300) puede mejorar la calidad de la imagen
        zoom = 2  # Factor de zoom (e.g., 2 = 200 DPI si el PDF es 72 DPI)
        matriz = fitz.Matrix(zoom, zoom)
        pix = pagina.get_pixmap(matrix=matriz, dpi=300)
        
        # Convertir a imagen PIL
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error al procesar la página {pagina_numero}: {e}")
        return None

# --- GENERACIÓN DE INFORME PDF CON PYMUPDF ---

def generar_informe_pdf(df_resultados, df_categorias, puntaje_total, tiempo_transcurrido):
    """Genera un informe PDF profesional con los resultados y gráficos."""
    doc = fitz.open()  # Crear un nuevo documento PDF
    ancho, alto = fitz.paper_size('a4')
    pagina = doc.new_page(width=ancho, height=alto)
    
    # Fuentes y Posiciones
    x_margen = 50
    y_pos = 50
    line_height = 20
    
    # Título Principal
    pagina.insert_text((x_margen, y_pos), "INFORME COMPLETO - TEST DE RAVEN (SPM)", fontsize=18, fontname="helv-bold")
    y_pos += 2 * line_height
    
    # 1. Resumen General (KPIs)
    pagina.insert_text((x_margen, y_pos), "1. Resumen de Resultados", fontsize=14, fontname="helv-bold")
    y_pos += line_height
    
    # KPI 1: Puntaje Total
    pagina.insert_text((x_margen, y_pos), f"Puntaje Total Obtenido:", fontsize=12, fontname="helv-bold")
    pagina.insert_text((x_margen + 200, y_pos), f"{puntaje_total} / {TOTAL_PREGUNTAS}", fontsize=12, fontname="helv")
    y_pos += line_height
    
    # KPI 2: Tiempo Total
    pagina.insert_text((x_margen, y_pos), f"Tiempo Total Empleado:", fontsize=12, fontname="helv-bold")
    pagina.insert_text((x_margen + 200, y_pos), f"{tiempo_transcurrido:.2f} segundos", fontsize=12, fontname="helv")
    y_pos += 2 * line_height

    # 2. Resultados por Serie (Tabla)
    pagina.insert_text((x_margen, y_pos), "2. Desglose de Puntajes por Serie", fontsize=14, fontname="helv-bold")
    y_pos += line_height
    
    # Cabecera de la tabla
    headers = ["Serie", "Respuestas Correctas", "Porcentaje (%)"]
    col_widths = [100, 200, 200]
    
    current_x = x_margen
    for i, header in enumerate(headers):
        pagina.insert_text((current_x, y_pos), header, fontsize=10, fontname="helv-bold")
        current_x += col_widths[i]
    y_pos += line_height
    
    # Datos de la tabla
    for index, row in df_categorias.iterrows():
        current_x = x_margen
        pagina.insert_text((current_x, y_pos), row['Serie'], fontsize=10, fontname="helv")
        current_x += col_widths[0]
        pagina.insert_text((current_x, y_pos), str(row['Correctas']), fontsize=10, fontname="helv")
        current_x += col_widths[1]
        pagina.insert_text((current_x, y_pos), f"{row['Porcentaje']:.2f}%", fontsize=10, fontname="helv")
        y_pos += line_height
    
    y_pos += line_height
    
    # 3. Detalle de Respuestas
    pagina.insert_text((x_margen, y_pos), "3. Detalle de Respuestas", fontsize=14, fontname="helv-bold")
    y_pos += line_height
    
    # Agregar detalle pregunta por pregunta (solo las primeras 10 para concisión)
    detalle_headers = ["Pregunta", "Tu Respuesta", "Respuesta Correcta", "Resultado"]
    col_widths_detalle = [80, 100, 150, 100]
    
    current_x = x_margen
    for i, header in enumerate(detalle_headers):
        pagina.insert_text((current_x, y_pos), header, fontsize=10, fontname="helv-bold")
        current_x += col_widths_detalle[i]
    y_pos += line_height
    
    # Datos del detalle
    for index, row in df_resultados.head(10).iterrows():
        current_x = x_margen
        color = (0, 0.5, 0) if row['Resultado'] == 'Correcto' else (0.5, 0, 0)
        
        pagina.insert_text((current_x, y_pos), row['Pregunta'], fontsize=10, fontname="helv")
        current_x += col_widths_detalle[0]
        pagina.insert_text((current_x, y_pos), str(row['Tu Respuesta']), fontsize=10, fontname="helv")
        current_x += col_widths_detalle[1]
        pagina.insert_text((current_x, y_pos), str(row['Respuesta Correcta']), fontsize=10, fontname="helv")
        current_x += col_widths_detalle[2]
        pagina.insert_text((current_x, y_pos), row['Resultado'], fontsize=10, fontname="helv-bold", color=color)
        y_pos += line_height
        
        if y_pos > alto - 50:
            pagina = doc.new_page(width=ancho, height=alto)
            y_pos = 50

    # Guardar el PDF en un buffer
    pdf_buffer = io.BytesIO()
    doc.save(pdf_buffer)
    doc.close()
    return pdf_buffer.getvalue()

# --- VISTAS DE STREAMLIT ---

def mostrar_pantalla_inicio():
    """Pantalla inicial para subir el PDF y comenzar el test."""
    st.title("🧩 Test de Matrices Progresivas de Raven (SPM)")
    st.header("Herramienta de Evaluación Digital")
    st.markdown("Esta aplicación simula el Test de Raven utilizando un archivo PDF proporcionado por el usuario, procesando cada página como una pregunta.")

    if not st.session_state.pdf_subido:
        # Subida del archivo
        uploaded_file = st.file_uploader(
            "Sube el archivo PDF del 'Test de Raven_Mejorar_OCR.pdf' para comenzar.", 
            type="pdf"
        )
        
        if uploaded_file is not None:
            st.session_state.pdf_content = uploaded_file.read()
            st.session_state.pdf_subido = True
            st.experimental_rerun()
    else:
        st.success("PDF cargado correctamente. Listo para comenzar el test.")
        if st.button("Iniciar Test de Raven (60 Preguntas)", help="Comenzar la evaluación cronometrada."):
            st.session_state.test_iniciado = 'en_curso'
            st.session_state.tiempo_inicio = time.time()
            st.experimental_rerun()

def mostrar_test_en_curso():
    """Muestra la pregunta actual y maneja la lógica de avance."""
    st.title("🧠 Evaluación en Curso")

    # Calculo de tiempo transcurrido
    tiempo_actual = time.time()
    tiempo_transcurrido = tiempo_actual - st.session_state.tiempo_inicio
    
    # Header e Indicador de progreso
    col1, col2 = st.columns([3, 1])
    pregunta_id = TODAS_PREGUNTAS[st.session_state.pregunta_actual_idx]
    
    col1.header(f"Pregunta {st.session_state.pregunta_actual_idx + 1} de {TOTAL_PREGUNTAS}: {pregunta_id}")
    col2.metric("Tiempo Transcurrido", f"{tiempo_transcurrido:.0f} s")

    st.progress((st.session_state.pregunta_actual_idx + 1) / TOTAL_PREGUNTAS)
    
    # 1. Obtener y mostrar la imagen de la pregunta
    pagina_num = MAPA_PAGINAS.get(pregunta_id)
    if st.session_state.pdf_content and pagina_num:
        # Usar la función de caché para la imagen
        img = obtener_imagen_pagina(st.session_state.pdf_content, pagina_num)
        if img:
            st.image(img, use_column_width=True, caption=f"Matriz de la pregunta {pregunta_id}")
        else:
            st.error("No se pudo cargar la imagen de la pregunta.")
    else:
        st.error("Error: Contenido del PDF no disponible o ID de pregunta inválida.")

    # 2. Botones de Respuesta (Opciones)
    serie = pregunta_id[0]
    num_opciones = OPCIONES_POR_PREGUNTA.get(serie, 6) # Valor por defecto 6
    opciones = list(range(1, num_opciones + 1))

    st.subheader("Selecciona la pieza que completa la matriz:")
    
    # Crear botones de respuesta en filas
    cols = st.columns(num_opciones if num_opciones <= 8 else 6)
    for i, opcion in enumerate(opciones):
        if cols[i % len(cols)].button(f"Opción {opcion}", key=f"btn_{pregunta_id}_{opcion}"):
            siguiente_pregunta(opcion)
            st.experimental_rerun() # Fuerza el avance automático
            
    st.markdown("---")
    # Botón para saltar (opcional, para desarrollo/test)
    if st.button("Saltar Pregunta (No Recomendado en un Test Real)", key="skip_btn"):
        siguiente_pregunta('Saltada')
        st.experimental_rerun()

def mostrar_resultados():
    """Calcula y muestra los resultados detallados y la opción de descarga."""
    st.title("🎉 Test Finalizado - Resultados Detallados")
    
    tiempo_fin = time.time()
    tiempo_transcurrido = tiempo_fin - st.session_state.tiempo_inicio
    
    # 1. Procesamiento de Resultados
    df_respuestas = pd.DataFrame(st.session_state.respuestas.items(), columns=['Pregunta', 'Tu Respuesta'])
    df_respuestas['Respuesta Correcta'] = df_respuestas['Pregunta'].map(RESPUESTAS_CORRECTAS)
    
    # Excluir preguntas saltadas/no respondidas (cuyo valor es 'Saltada')
    df_resultados = df_respuestas[df_respuestas['Tu Respuesta'] != 'Saltada'].copy()
    
    df_resultados['Correcto'] = df_resultados['Tu Respuesta'] == df_resultados['Respuesta Correcta']
    df_resultados['Resultado'] = df_resultados['Correcto'].apply(lambda x: 'Correcto' if x else 'Incorrecto')
    
    puntaje_total = df_resultados['Correcto'].sum()
    
    # 2. Análisis por Serie
    df_resultados['Serie'] = df_resultados['Pregunta'].str.extract(r'([A-E])')
    df_categorias = df_resultados.groupby('Serie')['Correcto'].sum().reset_index(name='Correctas')
    df_categorias['Total'] = 12
    df_categorias['Porcentaje'] = (df_categorias['Correctas'] / df_categorias['Total']) * 100
    
    # --- 3. Despliegue de KPIs y Métricas ---
    st.header("Resumen General")
    col1, col2, col3 = st.columns(3)
    
    # KPI 1: Puntaje Total
    col1.metric(
        label="Puntaje Total", 
        value=f"{puntaje_total} / {TOTAL_PREGUNTAS}", 
        delta=f"{(puntaje_total / TOTAL_PREGUNTAS) * 100:.1f}% de acierto"
    )
    
    # KPI 2: Tiempo Transcurrido
    col2.metric(
        label="Tiempo Total Empleado", 
        value=f"{tiempo_transcurrido:.2f} segundos", 
        delta=f"Tiempo promedio por pregunta: {tiempo_transcurrido / puntaje_total:.2f} s" if puntaje_total > 0 else "N/A"
    )
    
    # KPI 3: Preguntas Saltadas
    saltadas = (df_respuestas['Tu Respuesta'] == 'Saltada').sum()
    col3.metric(
        label="Preguntas Saltadas/Omitidas",
        value=saltadas
    )

    # --- 4. Gráfico de Resultados por Serie (Usando Streamlit nativo) ---
    st.header("Análisis de Resultados por Serie (A-E)")
    st.dataframe(df_categorias, hide_index=True)
    
    st.bar_chart(df_categorias.set_index('Serie')['Porcentaje'])
    
    # --- 5. Detalle de Respuestas y Descarga ---
    st.header("Detalle de Respuestas")
    st.dataframe(df_resultados[['Pregunta', 'Tu Respuesta', 'Respuesta Correcta', 'Resultado']], 
                 hide_index=True, 
                 use_container_width=True)
                 
    # 6. Generación de PDF
    st.subheader("Descargar Informe Completo (PDF)")
    
    # Generar el contenido binario del PDF
    pdf_data = generar_informe_pdf(df_resultados, df_categorias, puntaje_total, tiempo_transcurrido)
    
    # Botón de descarga con el archivo binario
    st.download_button(
        label="📥 Descargar Informe PDF",
        data=pdf_data,
        file_name="Informe_Test_Raven_Completo.pdf",
        mime="application/pdf"
    )
    
    st.button("Comenzar Nuevo Test", on_click=reiniciar_test)

# --- FUNCIÓN PRINCIPAL DE LA APLICACIÓN ---

def main():
    st.set_page_config(
        page_title="Test de Raven Completo",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inicializar_sesion()
    
    # Lógica de navegación entre pantallas
    if not st.session_state.test_iniciado:
        mostrar_pantalla_inicio()
    elif st.session_state.test_iniciado == 'en_curso':
        mostrar_test_en_curso()
    elif st.session_state.test_iniciado == 'finalizado':
        mostrar_resultados()

if __name__ == "__main__":
    main()
