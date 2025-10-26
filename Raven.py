import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random
import json
from datetime import datetime
import os
import base64
import io
import time

# Configuración inicial de la página
st.set_page_config(
    page_title="Test de Matrices Progresivas",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PatternGenerator:
    def __init__(self):
        self.image_size = (400, 400)
        self.patterns = {
            'basic': ['dots', 'lines', 'shapes'],
            'intermediate': ['rotation', 'size', 'number'],
            'advanced': ['combination', 'progression', 'symmetry']
        }
        
    def create_pattern(self, difficulty, pattern_type):
        image = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(image)
        
        if pattern_type == 'dots':
            self._draw_dot_pattern(draw, difficulty)
        elif pattern_type == 'lines':
            self._draw_line_pattern(draw, difficulty)
        elif pattern_type == 'shapes':
            self._draw_shape_pattern(draw, difficulty)
            
        return image
    
    def _draw_dot_pattern(self, draw, difficulty):
        num_dots = int(3 + difficulty * 5)
        spacing = self.image_size[0] // (num_dots + 1)
        for i in range(num_dots):
            for j in range(num_dots):
                x = spacing * (i + 1)
                y = spacing * (j + 1)
                radius = 5
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='black')

    def _draw_line_pattern(self, draw, difficulty):
        num_lines = int(2 + difficulty * 4)
        for _ in range(num_lines):
            start_x = random.randint(0, self.image_size[0])
            start_y = random.randint(0, self.image_size[1])
            end_x = random.randint(0, self.image_size[0])
            end_y = random.randint(0, self.image_size[1])
            draw.line([(start_x, start_y), (end_x, end_y)], fill='black', width=2)

    def _draw_shape_pattern(self, draw, difficulty):
        shapes = ['rectangle', 'circle', 'triangle']
        num_shapes = int(2 + difficulty * 3)
        for _ in range(num_shapes):
            shape = random.choice(shapes)
            size = random.randint(30, 100)
            x = random.randint(0, self.image_size[0] - size)
            y = random.randint(0, self.image_size[1] - size)
            
            if shape == 'rectangle':
                draw.rectangle([x, y, x+size, y+size], outline='black', width=2)
            elif shape == 'circle':
                draw.ellipse([x, y, x+size, y+size], outline='black', width=2)
            elif shape == 'triangle':
                points = [(x, y+size), (x+size//2, y), (x+size, y+size)]
                draw.polygon(points, outline='black', width=2)

class MatrixGenerator:
    def __init__(self):
        self.pattern_generator = PatternGenerator()
        self.matrix_size = (3, 3)
        
    def generate_matrix(self, difficulty, rule_type):
        images = []
        for i in range(self.matrix_size[0]):
            row = []
            for j in range(self.matrix_size[1]):
                if i == self.matrix_size[0]-1 and j == self.matrix_size[1]-1:
                    # Dejar la última celda vacía
                    image = Image.new('RGB', self.pattern_generator.image_size, 'white')
                else:
                    pattern_type = self._get_pattern_type(difficulty)
                    modified_difficulty = self._modify_difficulty(difficulty, i, j)
                    image = self.pattern_generator.create_pattern(modified_difficulty, pattern_type)
                row.append(image)
            images.append(row)
        return images
    
    def _get_pattern_type(self, difficulty):
        if difficulty < 0.3:
            return random.choice(self.pattern_generator.patterns['basic'])
        elif difficulty < 0.7:
            return random.choice(self.pattern_generator.patterns['intermediate'])
        else:
            return random.choice(self.pattern_generator.patterns['advanced'])
            
    def _modify_difficulty(self, base_difficulty, row, col):
        # Ajustar la dificultad según la posición en la matriz
        position_factor = (row + col) / (self.matrix_size[0] + self.matrix_size[1] - 2)
        return min(1.0, base_difficulty * (1 + position_factor * 0.5))

class RavenTest:
    def __init__(self):
        self.matrix_generator = MatrixGenerator()
        self.current_section = 'A'
        self.sections = ['A', 'B', 'C', 'D', 'E']
        self.questions_per_section = 12
        self.current_question = 0
        self.score = {section: 0 for section in self.sections}
        self.time_limits = {
            'A': 300,  # 5 minutos
            'B': 300,
            'C': 300,
            'D': 300,
            'E': 300
        }
        self.start_time = None
        
    def start_section(self):
        self.start_time = time.time()
        self.current_question = 0
        
    def get_time_remaining(self):
        if self.start_time is None:
            return self.time_limits[self.current_section]
        elapsed = time.time() - self.start_time
        remaining = self.time_limits[self.current_section] - elapsed
        return max(0, remaining)
    
    def generate_question(self):
        difficulty = self._calculate_difficulty()
        rule_type = self._determine_rule_type()
        matrix = self.matrix_generator.generate_matrix(difficulty, rule_type)
        options = self._generate_options(matrix, difficulty, rule_type)
        correct_answer = random.randint(0, 5)
        return {
            'matrix': matrix,
            'options': options,
            'correct': correct_answer,
            'section': self.current_section,
            'number': self.current_question + 1
        }
    
    def _calculate_difficulty(self):
        base_difficulty = 0.2
        section_factor = self.sections.index(self.current_section) / (len(self.sections) - 1)
        question_factor = self.current_question / self.questions_per_section
        return min(1.0, base_difficulty + section_factor * 0.4 + question_factor * 0.4)
    
    def _determine_rule_type(self):
        if self.current_section in ['A', 'B']:
            return 'basic'
        elif self.current_section in ['C', 'D']:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _generate_options(self, matrix, difficulty, rule_type):
        options = []
        # Generar opciones incluyendo la correcta
        for i in range(6):
            option = self.pattern_generator.create_pattern(difficulty, self._get_pattern_type(difficulty))
            options.append(option)
        return options

    def check_answer(self, user_answer, correct_answer):
        is_correct = user_answer == correct_answer
        if is_correct:
            self.score[self.current_section] += 1
        
        self.current_question += 1
        if self.current_question >= self.questions_per_section:
            self._advance_section()
            
        return is_correct
    
    def _advance_section(self):
        current_index = self.sections.index(self.current_section)
        if current_index < len(self.sections) - 1:
            self.current_section = self.sections[current_index + 1]
            self.current_question = 0
            self.start_time = None
        else:
            self.test_completed = True
    
    def get_results(self):
        total_score = sum(self.score.values())
        max_score = len(self.sections) * self.questions_per_section
        percentile = self._calculate_percentile(total_score)
        
        return {
            'total_score': total_score,
            'max_score': max_score,
            'percentile': percentile,
            'section_scores': self.score,
            'classification': self._get_classification(percentile)
        }
    
    def _calculate_percentile(self, score):
        # Implementar cálculo de percentil basado en normas
        return (score / (len(self.sections) * self.questions_per_section)) * 100
    
    def _get_classification(self, percentile):
        if percentile >= 95:
            return "Capacidad intelectual superior"
        elif percentile >= 75:
            return "Por encima del promedio"
        elif percentile >= 25:
            return "Promedio"
        elif percentile >= 5:
            return "Por debajo del promedio"
        else:
            return "Necesita atención especial"

class TestInterface:
    def __init__(self):
        if 'test' not in st.session_state:
            st.session_state.test = RavenTest()
            st.session_state.current_matrix = None
            st.session_state.user_data = None
            st.session_state.test_started = False
            st.session_state.test_completed = False
    
    def render(self):
        if not st.session_state.test_started:
            self._render_start_screen()
        elif not st.session_state.test_completed:
            self._render_test()
        else:
            self._render_results()
    
    def _render_start_screen(self):
        st.title("Test de Matrices Progresivas")
        
        with st.form("user_data"):
            st.write("### Información del participante")
            name = st.text_input("Nombre completo")
            age = st.number_input("Edad", min_value=5, max_value=100)
            education = st.selectbox("Nivel de educación", [
                "Primaria", "Secundaria", "Universidad", "Postgrado"
            ])
            
            if st.form_submit_button("Comenzar test"):
                if name and age:
                    st.session_state.user_data = {
                        'name': name,
                        'age': age,
                        'education': education,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.test_started = True
                    st.session_state.test.start_section()
                    st.experimental_rerun()
                else:
                    st.error("Por favor complete todos los campos requeridos")
    
    def _render_test(self):
        test = st.session_state.test
        
        # Mostrar información de la sección
        st.write(f"### Sección {test.current_section}")
        st.write(f"Pregunta {test.current_question + 1} de {test.questions_per_section}")
        
        # Mostrar tiempo restante
        remaining_time = test.get_time_remaining()
        st.progress(remaining_time / test.time_limits[test.current_section])
        st.write(f"Tiempo restante: {int(remaining_time)} segundos")
        
        if remaining_time <= 0:
            test._advance_section()
            if not test.test_completed:
                st.experimental_rerun()
            else:
                st.session_state.test_completed = True
                st.experimental_rerun()
            return
        
        # Generar nueva matriz si es necesario
        if st.session_state.current_matrix is None:
            st.session_state.current_matrix = test.generate_question()
        
        # Mostrar matriz
        matrix = st.session_state.current_matrix['matrix']
        cols = st.columns(3)
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                with cols[j]:
                    st.image(cell, use_column_width=True)
        
        # Mostrar opciones
        st.write("### Seleccione la opción correcta:")
        option_cols = st.columns(6)
        selected_option = None
        
        for i, option in enumerate(st.session_state.current_matrix['options']):
            with option_cols[i]:
                if st.button(f"Opción {i+1}", key=f"opt_{i}"):
                    selected_option = i
        
        if selected_option is not None:
            test.check_answer(selected_option, st.session_state.current_matrix['correct'])
            st.session_state.current_matrix = None
            
            if test.test_completed:
                st.session_state.test_completed = True
            
            st.experimental_rerun()
    
    def _render_results(self):
        results = st.session_state.test.get_results()
        
        st.title("Resultados del Test")
        st.write(f"### Puntaje total: {results['total_score']} de {results['max_score']}")
        st.write(f"### Percentil: {results['percentile']:.1f}")
        st.write(f"### Clasificación: {results['classification']}")
        
        # Mostrar resultados por sección
        st.write("### Resultados por sección:")
        for section, score in results['section_scores'].items():
            st.write(f"Sección {section}: {score} de {st.session_state.test.questions_per_section}")
        
        # Guardar resultados
        self._save_results(results)
        
        # Opción para descargar resultados
        if st.button("Descargar resultados"):
            self._download_results(results)
    
    def _save_results(self, results):
        results_data = {
            **st.session_state.user_data,
            **results
        }
        
        if not os.path.exists('results'):
            os.makedirs('results')
            
        filename = f"results/{st.session_state.user_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f)
    
    def _download_results(self, results):
        results_data = {
            **st.session_state.user_data,
            **results
        }
        
        json_str = json.dumps(results_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="resultados_test.json">Descargar JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

def main():
    interface = TestInterface()
    interface.render()

if __name__ == "__main__":
    main()
