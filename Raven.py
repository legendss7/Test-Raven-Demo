import streamlit as st
import json
import random
import time
import pandas as pd
from datetime import datetime
import base64
import math
import numpy as np

# Configuración inicial de la página
st.set_page_config(
    page_title="Test de Lógica Matemática - Analistas",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Banco de preguntas
QUESTIONS_BANK = {
    "secuencias": [
        {
            "question": "¿Qué número continúa la secuencia? 2, 6, 12, 20, 30, __",
            "options": ["42", "40", "38", "44"],
            "correct": "42",
            "explanation": "Cada número aumenta en una serie que crece de 4 en 4 (+4, +6, +8, +10, +12)",
            "difficulty": "media",
            "time_limit": 60
        },
        {
            "question": "Complete la serie: 1, 1, 2, 3, 5, 8, 13, __",
            "options": ["21", "20", "22", "24"],
            "correct": "21",
            "explanation": "Serie Fibonacci: cada número es la suma de los dos anteriores",
            "difficulty": "fácil",
            "time_limit": 45
        },
        {
            "question": "Serie: 3, 6, 12, 24, 48, __",
            "options": ["96", "72", "86", "94"],
            "correct": "96",
            "explanation": "Cada número se multiplica por 2",
            "difficulty": "fácil",
            "time_limit": 30
        }
    ],
    "logica": [
        {
            "question": "Si todos los A son B, y algunos B son C, entonces:",
            "options": [
                "Todos los A son C",
                "Algunos A podrían ser C",
                "Ningún A es C",
                "Todos los C son A"
            ],
            "correct": "Algunos A podrían ser C",
            "explanation": "Es un problema de lógica proposicional y conjuntos",
            "difficulty": "difícil",
            "time_limit": 90
        },
        {
            "question": "En una oficina, el 60% de los empleados habla inglés y el 40% habla francés. Si el 20% habla ambos idiomas, ¿qué porcentaje no habla ninguno de los dos idiomas?",
            "options": ["20%", "15%", "25%", "30%"],
            "correct": "20%",
            "explanation": "Usando teoría de conjuntos: Total = Inglés + Francés - Ambos + Ninguno -> 100 = 60 + 40 - 20 + x",
            "difficulty": "media",
            "time_limit": 120
        }
    ],
    "analisis_numerico": [
        {
            "question": "Si un proyecto tiene una tasa de retorno del 15% anual, ¿cuántos años se necesitan para duplicar la inversión inicial?",
            "options": ["4.7 años", "5.0 años", "6.7 años", "7.3 años"],
            "correct": "5.0 años",
            "explanation": "Usando la regla del 72: 72/15 ≈ 5 años",
            "difficulty": "difícil",
            "time_limit": 90
        },
        {
            "question": "En una base de datos, el tiempo de búsqueda crece logarítmicamente. Si con 1000 registros tarda 2 segundos, ¿cuánto tardará aproximadamente con 8000 registros?",
            "options": ["16 segundos", "6 segundos", "4 segundos", "8 segundos"],
            "correct": "4 segundos",
            "explanation": "log₂(8000/1000) = 3, entonces 2 * (3/2) = 4 segundos",
            "difficulty": "difícil",
            "time_limit": 120
        }
    ]
}

def init_session_state():
    """Inicializa el estado de la sesión"""
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.answers = []
        st.session_state.start_time = None
        st.session_state.questions = []
        st.session_state.user_data = None
        st.session_state.test_completed = False

def prepare_test():
    """Prepara las preguntas del test"""
    questions = []
    # Seleccionar preguntas de cada categoría
    for category in QUESTIONS_BANK.values():
        questions.extend(random.sample(category, min(2, len(category))))
    random.shuffle(questions)
    return questions

def calculate_score(answers, total_questions):
    """Calcula el puntaje y genera recomendaciones"""
    correct_answers = sum(1 for ans in answers if ans['is_correct'])
    score_percentage = (correct_answers / total_questions) * 100
    
    # Análisis por categoría
    category_scores = {
        'secuencias': [],
        'logica': [],
        'analisis_numerico': []
    }
    
    recommendations = []
    
    if score_percentage < 60:
        recommendations.append("Se recomienda reforzar conceptos básicos de lógica matemática.")
    elif score_percentage < 80:
        recommendations.append("Buen desempeño. Enfocarse en mejorar velocidad de resolución.")
    else:
        recommendations.append("Excelente desempeño. Nivel adecuado para el puesto.")
    
    return {
        'score_percentage': score_percentage,
        'correct_answers': correct_answers,
        'total_questions': total_questions,
        'recommendations': recommendations,
        'category_scores': category_scores
    }

def export_results_to_csv(results):
    """Exporta los resultados a CSV"""
    df = pd.DataFrame([results])
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resultados_test.csv">Descargar resultados (CSV)</a>'
    return href

def main():
    init_session_state()
    
    st.title("Test de Lógica Matemática para Analistas")
    
    if not st.session_state.test_started:
        st.write("### Bienvenido al Test de Lógica Matemática")
        st.write("""
        Este test evaluará sus habilidades en:
        - Secuencias numéricas
        - Lógica proposicional
        - Análisis numérico
        
        Duración aproximada: 30 minutos
        """)
        
        # Formulario de registro
        st.write("### Información del candidato")
        with st.form("registro"):
            nombre = st.text_input("Nombre completo")
            email = st.text_input("Email")
            experiencia = st.selectbox(
                "Años de experiencia",
                ["0-1 años", "1-3 años", "3-5 años", "5+ años"]
            )
            
            if st.form_submit_button("Comenzar Test"):
                if nombre and email:
                    st.session_state.user_data = {
                        "nombre": nombre,
                        "email": email,
                        "experiencia": experiencia,
                        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.test_started = True
                    st.session_state.questions = prepare_test()
                    st.session_state.start_time = time.time()
                    st.experimental_rerun()
                else:
                    st.error("Por favor complete todos los campos")
    
    elif not st.session_state.test_completed:
        # Mostrar pregunta actual
        question = st.session_state.questions[st.session_state.current_question]
        
        # Mostrar tiempo restante
        time_limit = question.get('time_limit', 60)
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            remaining = max(0, time_limit - elapsed)
            st.progress(remaining / time_limit)
            st.write(f"Tiempo restante: {int(remaining)} segundos")
            
            if remaining <= 0:
                st.session_state.answers.append({
                    'question': question['question'],
                    'answer': None,
                    'correct': question['correct'],
                    'is_correct': False,
                    'time_taken': time_limit
                })
                st.session_state.current_question += 1
                st.session_state.start_time = time.time()
                if st.session_state.current_question >= len(st.session_state.questions):
                    st.session_state.test_completed = True
                st.experimental_rerun()
        
        st.write(f"### Pregunta {st.session_state.current_question + 1} de {len(st.session_state.questions)}")
        st.write(question["question"])
        
        # Mostrar opciones
        answer = st.radio("Seleccione una respuesta:", question["options"], key=f"q_{st.session_state.current_question}")
        
        if st.button("Siguiente"):
            time_taken = time.time() - st.session_state.start_time
            is_correct = answer == question["correct"]
            
            st.session_state.answers.append({
                'question': question['question'],
                'answer': answer,
                'correct': question['correct'],
                'is_correct': is_correct,
                'time_taken': time_taken
            })
            
            if is_correct:
                st.session_state.score += 1
            
            st.session_state.current_question += 1
            st.session_state.start_time = time.time()
            
            if st.session_state.current_question >= len(st.session_state.questions):
                st.session_state.test_completed = True
            
            st.experimental_rerun()
    
    else:
        # Mostrar resultados
        st.write("### Resultados del Test")
        results = calculate_score(st.session_state.answers, len(st.session_state.questions))
        
        st.write(f"Puntaje: {results['score_percentage']:.1f}%")
        st.write(f"Respuestas correctas: {results['correct_answers']} de {results['total_questions']}")
        
        st.write("### Recomendaciones:")
        for rec in results['recommendations']:
            st.write(f"- {rec}")
        
        # Mostrar respuestas
        st.write("### Detalle de respuestas:")
        for i, answer in enumerate(st.session_state.answers):
            with st.expander(f"Pregunta {i+1}"):
                st.write(f"Pregunta: {answer['question']}")
                st.write(f"Su respuesta: {answer['answer'] if answer['answer'] else 'Sin responder'}")
                st.write(f"Respuesta correcta: {answer['correct']}")
                st.write(f"Tiempo: {answer['time_taken']:.1f} segundos")
                
                if answer['is_correct']:
                    st.success("¡Correcto!")
                else:
                    st.error("Incorrecto")
        
        # Exportar resultados
        st.write("### Exportar resultados")
        results_data = {
            **st.session_state.user_data,
            **results
        }
        st.markdown(export_results_to_csv(results_data), unsafe_allow_html=True)
        
        if st.button("Realizar nuevo test"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == "__main__":
    main()
