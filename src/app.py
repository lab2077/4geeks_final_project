import streamlit as st
import numpy as np
import pickle

# Cargar el modelo LSTM
with open('lstm_tornado.pkl', 'rb') as file:
    lstm_model = pickle.load(file)

# Título
st.title('Predicción de Magnitud de Tornado 🌪️')

# Inputs del usuario
st.header('Datos del Tornado')
month = st.selectbox('Mes', list(range(1, 13)))
start_latitude = st.number_input('Latitud de Inicio', value=35.0, step=0.1)
start_longitude = st.number_input('Longitud de Inicio', value=-97.0, step=0.1)

# Preparar entrada (ajustar según cómo espera tu modelo los datos)
input_data = np.array([[month, start_latitude, start_longitude]])
input_data = np.expand_dims(input_data, axis=0)  # LSTM espera 3D [samples, timesteps, features]

# Botón para predecir
if st.button('Predecir Magnitud'):
    prediction = lstm_model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Si es clasificación
    st.success(f'Magnitud Predicha: {predicted_class}')
