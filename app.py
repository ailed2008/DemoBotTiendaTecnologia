import streamlit as st
import sqlite3  # o la librería de tu BD
import pandas as pd
import pickle  # si tienes un modelo entrenado
import random

# Título
st.title("Chatbot con Python y Predicciones")

# Entrada del usuario
user_input = st.text_input("Escribe tu pregunta:")
API_KEY="AIzaSyBnFrra0uqqQSdKHskygMN6kGk29QBRSPE"
# Conexión a la base de datos (ejemplo con SQLite)
def consulta_bd(query):
    # Simulación: en vez de conectar a tu BD real, devuelvo datos falsos
    datos = {
        "ventas": [100, 150, 200],
        "mes": ["Enero", "Febrero", "Marzo"]
    }
    return pd.DataFrame(datos)

# Cargar modelo (si tienes uno entrenado)
# with open("modelo.pkl", "rb") as f:
#     model = pickle.load(f)

# Respuesta del chatbot
if st.button("Enviar"):
    if user_input:
        # Ejemplo de consulta
        df = consulta_bd(user_input)
        respuesta = f"Encontré estos datos: {df.to_dict()}"
        
        # Si tuvieras un modelo predictivo:
        # pred = model.predict([[feature1, feature2]])
        # respuesta = f"La predicción es: {pred[0]}"
        
        st.write(respuesta)
    else:
        st.warning("Por favor escribe algo antes de enviar.")
