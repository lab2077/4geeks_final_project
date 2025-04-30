import streamlit as st
import pandas as pd


df_train = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_train.csv')
df_test = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_test.csv')
df_val = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_val.csv')
df_raw = pd.read_csv('/workspaces/4geeks_final_project/data/raw/us_tornado_dataset_1950_2021.csv')


#------------- TITLE

st.title('🌪️ Proyecto de Predicción de Magnitud de Tornados')
st.markdown('''

---------
            
- Fuente: kaggle.com
- URL: https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021
- Fecha: 30 de Abril del 2025

---------''')



#------------- OBJETIVO


st.markdown('''## 1. 🎯 Objetivo del Proyecto
       

Desarrollar un modelo de **Machine Learning** para predecir la magnitud de tornados en EE.UU. basándonos en un conjunto de datos históricos llamado "US Tornado Dataset 1950 2021 (CSV)".

---------     
        ''')


#------------- INTRO


st.markdown(''' ## 2. Conceptos (Marco teórico)
 
            
#### 🌪️ Tornados: Definición, Clasificación e Impacto
---------     
##### **❓ ¿Qué es un tornado?**

**Un tornado** es una columna de aire que gira violentamente desde una tormenta eléctrica hasta la superficie terrestre.  

---------            

###### **- Características principales:**  

- Forma: A menudo de embudo característica  
- Vientos: Pueden oscilar entre 105 km/h y más de 322 km/h (65 a más de 200 mph), dependiendo de la intensidad del tornado.  
- Tamaño: Puede variar desde unos pocos metros hasta más de 2 kilómetros de ancho.
- Duración: Generalmente dura desde unos pocos segundos hasta más de una hora.
- Movimiento: Normalmente se desplaza a velocidades de 30 a 70 km/h, aunque puede variar.

            
###### **- Formación:**
              
- Aire cálido y húmedo en la superficie se encuentra con aire frío y seco en niveles superiores.
- Se desarrolla una tormenta supercelda con cizalladura del viento (cambios en la velocidad o dirección del viento con la altura).
- El aire ascendente (corrientes de ascenso) comienza a rotar debido a la cizalladura del viento.
- Esta rotación se estrecha y alarga, formando un embudo visible de nubes.
- Si el embudo toca el suelo, se convierte en un tornado.            

            
###### **- Línea de Tiempo de la Taxonomía**

Los tornados han sido clasificados de diferentes maneras a lo largo del tiempo. Desde 1958, han existido múltiples taxonomías que han evolucionado desde descripciones visuales simples hasta clasificaciones científicas más precisas. A lo largo de las décadas, se han introducido nuevas categorías y refinado definiciones para reflejar mejor la diversidad y formación de los tornados.            

            
- Antes de 1958: Sin sistema formal; basado en apariencia/daños.
- 1958: 3 especies — Embudo, Trombón de agua, Diablo de polvo.
- 2000: Se añadió Tromba terrestre (Landspout); definiciones refinadas.
- 2009: Cambio a 2 tipos — Supercelda y No Supercelda.
- 2013: 3 tipos — Supercelda, No Supercelda, Híbrido.


---------

#### **📊 Clasificaciónes de Tornados**       

Antes de 1971, no existía un sistema formal para clasificar tornados. En 1971, se introdujo la Escala Fujita (F), y en 2007 fue reemplazada en EE.UU. por la Escala Mejorada Fujita (EF), diseñada para evaluar con mayor precisión el daño y estimar mejor la velocidad del viento. El cambio se realizó porque la escala original sobreestimaba algunas velocidades del viento. Ambas escalas se basan en daños observados, pero la EF también considera el tipo de estructura afectada.
            
**Otras variables consideradas en la clasificación incluyen:**

- Velocidad estimada del viento
-Tipo de edificaciones o vegetación dañadas
-Trayectoria y duración del tornado
-Ancho del recorrido de daño            

---------

            
##### *****- Escala Fujita Original (F) (1971)*****

| Categoría | Velocidad Viento | Daños |
|-----------|------------------|-------|
| **F0** | 40-72 mph (64-116 km/h) | Leves (ramas rotas, señales dobladas) |
| **F1** | 73-112 mph (117-180 km/h) | Moderados (tejas voladas, caravanas volcadas) |
| **F2** | 113-157 mph (181-253 km/h) | Considerables (techos arrancados, árboles arrancados) |
| **F3** | 158-206 mph (254-332 km/h) | Graves (estructuras débiles destruidas) |
| **F4** | 207-260 mph (333-418 km/h) | Devastadores (casas niveladas, autos lanzados) |
| **F5** | 261-318 mph (419-512 km/h) | Increíbles (estructuras arrasadas, deformación del terreno) |
            
##### *****- Escala Mejorada Fujita (EF) (2007 – EE.UU.)*****
| Categoría | Velocidad Viento | Daños |
|-----------|------------------|-------|
| **EF0** | 65-85 mph (105-137 km/h) | Leves (ramas rotas) |
| **EF1** | 86-110 mph (138-177 km/h) | Moderados (tejados dañados) |
| **EF2** | 111-135 mph (178-217 km/h) | Considerables (árboles arrancados) |
| **EF3** | 136-165 mph (218-266 km/h) | Graves (paredes derrumbadas) |
| **EF4** | 166-200 mph (267-322 km/h) | Devastadores (casas destruidas) |
| **EF5** | >200 mph (322+ km/h) | Increíbles (estructuras niveladas) |
            

---------
            
#### **💰 Impacto Económico**
##### **- Factores clave:**  
- **Categoría del tornado** (EF3+ = mayor destrucción)  
- **Ubicación** (zonas urbanas = mayor costo)  

##### **- Áreas afectadas:**  
- 🏠 Viviendas e infraestructuras  
- 🌾 Agricultura (cosechas/ganado)  
- 🏢 Negocios (interrupciones operativas)  
- 🚑 Respuesta de emergencia  

##### **- Ejemplos destacados (EE.UU.)**
| Evento | Año | Daños (USD) | Categoría |
|--------|-----|-------------|-----------|
| Joplin, MO | 2011 | $2.8 mil millones | EF5 |
| Moore, OK | 2013 | $2 mil millones | EF5 |
| **Promedio anual (1996-2023)** | - | **$1.1 mil millones** | EF3-EF4 |  


> **Dato crítico:** Un EF2 en zona urbana puede costar más que un EF4 en área rural.

            
---------

''')
            


#------------- DATASET INFO



st.markdown('''

## 3. Dataset
-----------
            
### 🗂️ US Tornado Dataset 1950-2021
Utilizamos un conjunto de datos obtenido de Kaggle.com, que recopila información sobre tornados ocurridos en Estados Unidos desde 1950 hasta 2021. Este dataset incluye variables como la fecha, ubicación, intensidad, longitud, ancho, y daños causados por cada tornado reportado. Gracias a su amplitud temporal y nivel de detalle, nos permitió analizar tendencias históricas y comparar la severidad de los eventos a lo largo del tiempo.

-----------

###

|Nombre|Descripción|Tipo|
|----|-----------|----|
|year|Año con 4 dígitos|Int|
|month|Mes (1-12)|Int|
|day|Día del mes|Int|
|date|Objeto datetime (ej. 2011-01-01)|Date|
|state|Estado donde se originó el tornado; abreviatura de 2 letras|String|
|magnitude|Escala Fujita mejorada para clasificar tornados|Int|
|injuries|Número de heridos durante el tornado|Int|
|fatalities|Número de fallecidos durante el tornado|Int|
|start_latitude|Latitud inicial en grados decimales|Float|
|start_longitude|Longitud inicial en grados decimales|Float|
|end_latitude|Latitud final en grados decimales|Float|
|end_longitude|Longitud final en grados decimales|Float|
|length|Longitud de la trayectoria en millas|Float|
|width|Ancho en yardas|Float|

            











''')






#st.dataframe()

#plt.figure(figsize=(8,8))
#plt.scatter(x=df['cement'],y=df['compressive_strength'])
#plt.title('Compressive Strength vs Concrete Density')
#st.pyplot(plt)