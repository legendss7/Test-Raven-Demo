import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import random
import json
from datetime import datetime
import base64
import time

# Configuración inicial de la página
st.set_page_config(
    page_title="Test de Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización del estado de la sesión
def init_session_state():
    if 'init' not in st.session_state:
        st.session_state.init = True
        st.session_state.test_started = False
        st.session_state.test_completed = False
        st.session_state.current_matrix = None
        st.session_state.user_data = None
        st.session_state.current_section = 'A'
        st.session_state.current_question = 0
        st.session_state.score = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        st.session_state.start_time = None
        st.session_state.needs_new_matrix = True

init_session_state()

def create_pattern(size, difficulty, pattern_type):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    if pattern_type == 'dots':
        num_dots = int(3 + difficulty * 5)
        spacing = size[0] // (num_dots + 1)
        for i in range(num_dots):
            for j in range(num_dots):
                x = spacing * (i + 1)
                y = spacing * (j + 1)
                radius = 5
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='black')
    
    elif pattern_type == 'lines':
        num_lines = int(2 + difficulty * 4)
        for _ in range(num_lines):
            start_x = random.randint(0, size[0])
            start_y = random.randint(0, size[1])
            end_x = random.randint(0, size[0])
            end_y = random.randint(0, size[1])
            draw.line([(start_x, start_y), (end_x, end_y)], fill='black', width=2)
    
    else:  # shapes
        shapes = ['rectangle', 'circle', 'triangle']
        num_shapes = int(2 + difficulty * 3)
        for _ in range(num_shapes):
            shape = random.choice(shapes)
            shape_size = random.randint(30, 100)
            x = random.randint(0, size[0] - shape_size)
            y = random.randint(0, size[1] - shape_size)
            
            if shape == 'rectangle':
                draw.rectangle([x, y, x+shape_size, y+shape_size], outline='black', width=2)
            elif shape == 'circle':
                draw.ellipse([x, y, x+shape_size, y+shape_size], outline='black', width=2)
            else:
                points = [(x, y+shape_size), (x+shape_size//2, y), (x+shape_size, y+shape_size)]
                draw.polygon(points, outline='black', width=2)
    
    return image

def generate_matrix(difficulty):
    size = (400, 400)
    matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == 2 and j == 2:  # Última celda vacía
                pattern = Image.new('RGB', size, 'white')
            else:
                pattern_type = random.choice(['dots', 'lines', 'shapes'])
                pattern = create_pattern(size, difficulty, pattern_type)
            row.append(pattern)
        matrix.append(row)
    return matrix

def generate_options(difficulty):
    size = (400, 400)
    options = []
    for _ in range(6):
        pattern_type = random.choice(['dots', 'lines', 'shapes'])
        option = create_pattern(size, difficulty, pattern_type)
        options.append(option)
    return options

def calculate_difficulty(section, question):
    sections = ['A', 'B', 'C', 'D', 'E']
    section_factor = sections.index(section) / (len(sections) - 1)
    question_factor = question / 12
    return 0.2 + section_factor * 0.4 + question_factor * 0.4

def handle_start_button():
    name = st.session_state.name
    age = st.session_state.age
    education = st.session_state.education
    
    if name and age:
        st.session_state.user_data = {
            'name': name,
            'age': age,
            'education': education,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.test_started = True
        st.session_state.start_time = time.time()
    else:
        st.error("Por favor complete todos los campos requeridos")

def handle_option_click(option_number):
    if option_number == st.session_state.current_matrix['correct']:
        st.session_state.score[st.session_state.current_section] += 1
    
    st.session_state.current_question += 1
    st.session_state.needs_new_matrix = True
    
    if st.session_state.current_question >= 12:
        if st.session_state.current_section == 'E':
            st.session_state.test_completed = True
        else:
            st.session_state.current_section = chr(ord(st.session_state.current_section) + 1)
            st.session_state.current_question = 0
            st.session_state.start_time = time.time()

def main():
    st.title("Test de Matrices Progresivas")

    if not st.session_state.test_started:
        st.write("### Información del participante")
        st.text_input("Nombre completo", key="name")
        st.number_input("Edad", min_value=5, max_value=100, key="age")
        st.selectbox("Nivel de educación", 
            ["Primaria", "Secundaria", "Universidad", "Postgrado"],
            key="education")
        
        st.button("Comenzar test", on_click=handle_start_button)
    
    elif not st.session_state.test_completed:
        # Verificar tiempo
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            if elapsed > 300:  # 5 minutos
                if st.session_state.current_section == 'E':
                    st.session_state.test_completed = True
                    st.rerun()
                else:
                    st.session_state.current_section = chr(ord(st.session_state.current_section) + 1)
                    st.session_state.current_question = 0
                    st.session_state.start_time = time.time()
                    st.session_state.needs_new_matrix = True
        
        st.write(f"### Sección {st.session_state.current_section}")
        st.write(f"Pregunta {st.session_state.current_question + 1} de 12")
        
        remaining = 300 - (time.time() - st.session_state.start_time)
        st.progress(remaining / 300)
        st.write(f"Tiempo restante: {int(remaining)} segundos")
        
        if st.session_state.needs_new_matrix:
            difficulty = calculate_difficulty(
                st.session_state.current_section, 
                st.session_state.current_question
            )
            matrix = generate_matrix(difficulty)
            options = generate_options(difficulty)
            st.session_state.current_matrix = {
                'matrix': matrix,
                'options': options,
                'correct': random.randint(0, 5)
            }
            st.session_state.needs_new_matrix = False
        
        # Mostrar matriz
        cols = st.columns(3)
        for i, row in enumerate(st.session_state.current_matrix['matrix']):
            for j, cell in enumerate(row):
                with cols[j]:
                    st.image(cell, use_column_width=True)
        
        # Mostrar opciones
        st.write("### Seleccione la opción correcta:")
        option_cols = st.columns(6)
        for i, option in enumerate(st.session_state.current_matrix['options']):
            with option_cols[i]:
                st.button(
                    f"Opción {i+1}", 
                    key=f"opt_{i}",
                    on_click=handle_option_click,
                    args=(i,)
                )
    
    else:
        st.write("### Resultados del Test")
        total_score = sum(st.session_state.score.values())
        max_score = 60  # 12 preguntas * 5 secciones
        percentile = (total_score / max_score) * 100
        
        st.write(f"Puntaje total: {total_score} de {max_score}")
        st.write(f"Percentil: {percentile:.1f}")
        
        # Clasificación
        if percentile >= 95:
            classification = "Capacidad intelectual superior"
        elif percentile >= 75:
            classification = "Por encima del promedio"
        elif percentile >= 25:
            classification = "Promedio"
        elif percentile >= 5:
            classification = "Por debajo del promedio"
        else:
            classification = "Necesita atención especial"
        
        st.write(f"Clasificación: {classification}")
        
        # Resultados por sección
        st.write("### Resultados por sección:")
        for section, score in st.session_state.score.items():
            st.write(f"Sección {section}: {score} de 12")
        
        # Preparar resultados para descargar
        results = {
            **st.session_state.user_data,
            'total_score': total_score,
            'percentile': percentile,
            'classification': classification,
            'section_scores': st.session_state.score
        }
        
        # Botón para descargar resultados
        if st.button("Descargar resultados"):
            json_str = json.dumps(results, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="resultados_test.json">Descargar JSON</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
