import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_excel('../data/processed/df_train.csv')
df_test = pd.read_excel('../data/processed/df_test.csv')


st.title('Concrete Data Set Navigator')


st.markdown('**Streamlit** is a good *tool*!')

st.text('''En este proyecto desarrollaremos un modelo de aprendizaje automático utilizando redes neuronales LSTM (Long Short-Term Memory) para predecir la magnitud de tornados en Estados Unidos, basándonos en un conjunto de datos históricos llamado us_tornado_dataset_1950_2021.csv. Este dataset incluye información relevante sobre cada evento de tornado, como la fecha, el estado afectado, número de heridos y fallecidos, coordenadas de inicio y fin, longitud y ancho del tornado.

El objetivo principal es construir un modelo capaz de aprender patrones temporales y espaciales a partir de estos datos, permitiendo realizar predicciones más precisas sobre la magnitud de futuros tornados. Las redes LSTM son especialmente útiles para este tipo de problemas, ya que están diseñadas para trabajar con secuencias de datos y capturar relaciones de largo plazo, lo que resulta ideal en contextos donde la evolución temporal de los eventos es relevante.''')



st.dataframe()

plt.figure(figsize=(8,8))
plt.scatter(x=df['cement'],y=df['compressive_strength'])
plt.title('Compressive Strength vs Concrete Density')
st.pyplot(plt)