import geopandas as gpd
import shapely
import streamlit as st
from shapely.geometry import Point, Polygon, MultiPolygon
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import MarkerCluster



df_train = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_train.csv')
df_test = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_test.csv')
df_val = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df_val.csv')
df = pd.read_csv('/workspaces/4geeks_final_project/data/processed/df.csv')
df_raw = pd.read_csv('/workspaces/4geeks_final_project/data/raw/us_tornado_dataset_1950_2021.csv')
length_average_df = pd.read_csv('/workspaces/4geeks_final_project/data/processed/state_month_length_avg_2015_to_2021.csv')
width_average_df = pd.read_csv('/workspaces/4geeks_final_project/data/processed/state_month_width_avg_2015_to_2021.csv')



#------------- TITLE

st.title('🌪️ Proyecto de Predicción de Magnitud de Tornados')
st.markdown('''

---------
            
- Fuente: kaggle.com
- URL: https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021
- Fecha: 30 de Abril del 2025

---------''')



#------------- OBJETIVO


st.markdown('''## 1. Objetivo del Proyecto
       

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

### ***Diccionario de Datos***

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

            
-----------

''')

#------------------ HYPOTHESIS

st.markdown('''
            
## 4. HIPÓTESIS
----------            
                        
La intensidad de un tornado puede predecirse con precisión utilizando un modelo de aprendizaje automático entrenado con variables categóricas como el mes, el estado, la región y la estación del año, así como variables numéricas como la latitud y longitud de inicio, la longitud del trayecto y el ancho del tornado. Se espera que entre todas estas, la época del año y la ubicación geográfica tengan una influencia especialmente significativa en la magnitud del tornado.
            

----------

''')



#-------------------- PREPROCESAMIENTO



st.markdown('''
            
## 5. PREPROCESAMIENTO DE DATOS DE TORNADOS

----------                        

### **📋 Resumen de pasos:**

- Filtramos los datos para incluir solo los tornados ocurridos a partir de febrero de 2007.

- Eliminamos los registros con magnitudes inválidas (-9, 4 y 5).

- Contamos los tornados por estado y nos quedamos solo con los estados que tienen al menos 50 registros.

- Eliminamos las filas con valores cero en longitud, ancho o latitud final, ya que indican datos incompletos.

- Eliminamos filas duplicadas para evitar repeticiones en el entrenamiento del modelo.

- Convertimos la columna de fecha al formato datetime y ordenamos los datos cronológicamente.

- Reiniciamos los índices del DataFrame para mantener un orden limpio.

- Renombramos las columnas para hacerlas más intuitivas y legibles.

- Agrupamos los estados en cuatro regiones geográficas de EE.UU. (este, sur, medio oeste y oeste) y creamos la columna region.

- Creamos la columna season según el mes del año en que ocurrió cada tornado (invierno, primavera, verano u otoño).

- Convertimos las columnas magnitude, state, region y season en variables categóricas para optimizar el uso de memoria.            

----------
### ***📊 Muestra del DataFrame Final***
            
         
''')

st.write(df.sample(20, random_state=2025))



# ---------------------- EDA

st.markdown('''





            
----------
## 5. EDA
----------            

### ***Frecuencia de Tornados por Mes***
            
Este barplot muestra visualmente cuántos tornados ocurrieron en cada mes, lo que nos permite analizar las tendencias a lo largo del año. Al observar el barplot, podemos notar que mayo tiende a tener una mayor concentración de tornados en comparación con otros meses. Este patrón sugiere que mayo podría ser un mes especialmente activo para la ocurrencia de tornados, lo que podría estar relacionado con condiciones climáticas específicas durante esa temporada.
''')


# Frecuencia de Tornados por Mes - HISTOGRAM

plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.title('📊 Frecuencia de Tornados por Mes')
df_train['month'].value_counts().sort_index().plot(kind='bar')
st.pyplot(plt)



st.markdown('''
---------
### ***Frecuencia de Tornados por Estado del 2007 al 2021***

Este barplot muestra la frecuencia de tornados por estado entre 2007 y 2021. El gráfico presenta la cantidad de tornados ocurridos en cada estado, y los colores de las barras se utilizan para representar diferentes rangos de frecuencia:

- Las barras rojas indican estados con más de 1000 tornados.
- Las barras naranjas representan estados con entre 400 y 1000 tornados.
- Las barras verdes indican estados con menos de 400 tornados.

Además, se han añadido líneas horizontales (líneas discontinuas) para marcar los umbrales de 1000 y 400 tornados, destacando visualmente las diferencias significativas entre los estados con alta, media y baja cantidad de tornados.

El gráfico nos permite observar rápidamente qué estados han sido más afectados por tornados, destacando especialmente aquellos con una frecuencia notablemente alta, lo que puede indicar regiones más propensas a este fenómeno climático. Los estados más afectados por tornados incluyen **Texas, Kansas, Oklahoma, Alabama, Mississippi, Missouri, Louisiana, Illinois, Iowa, Minnesota, Nebraska, Florida, Georgia, Colorado, Arkansas y Kentucky**.


''')



#  Frecuencia de Tornados por Estado del 2007 al 2021 GRAPHIC

state_counts = df_train['state'].value_counts().sort_values(ascending=False)
colors = []
for count in state_counts:
    if count > 1000:
        colors.append('red')
    elif 400 <= count <= 1000:
        colors.append('orange')
    else:
        colors.append('green')

fig, ax = plt.subplots(figsize=(16, 8))

bars = ax.bar(state_counts.index, state_counts.values, color=colors)

ax.set_title('Frecuencia de Tornados por Estado del 2007 al 2021', fontsize=16, pad=20)
ax.set_xlabel('Estado', fontsize=12)
ax.set_ylabel('Número de Tornados', fontsize=12)
ax.set_xticklabels(state_counts.index, rotation=45, ha='right')


ax.axhline(y=1000, color='darkred', linestyle='--', alpha=0.5)
ax.axhline(y=400, color='darkorange', linestyle='--', alpha=0.5)


legend_elements = [
    mpatches.Patch(color='red', label='> 1000 tornados'),
    mpatches.Patch(color='orange', label='400-1000 tornados'),
    mpatches.Patch(color='green', label='< 400 tornados')
]
ax.legend(handles=legend_elements, loc='upper right')
st.pyplot(fig)

st.markdown('''
---------

### ***Mapa Interactivo de Tornados por Año***

Este mapa interactivo permite seleccionar un año específico y visualizar las concentraciones de tornados ocurridos en ese periodo. Al filtrar los datos por año, podemos observar las ubicaciones de los tornados y sus trayectorias, lo que ofrece una nueva forma de analizar la distribución y frecuencia de estos fenómenos meteorológicos.

Optamos por trabajar con los datos año por año en lugar de intentar una animación con todos los años simultáneamente para evitar la sobrecarga de datos. Esto nos permite obtener una visión más clara y manejable de las concentraciones de tornados sin perder detalles importantes.

Este enfoque facilita el análisis de patrones y tendencias en los tornados, lo que puede ser útil para la toma de decisiones en estudios meteorológicos y gestión de riesgos.
''')





# INTERACTIVE MAP 

year_filter = st.selectbox('Selecciona el año:', [2017, 2018, 2019, 2020, 2021], index=2)

# Filter data based on the selected year
df_filtered = df[df['year'] == year_filter].copy()  # Cambia el año según el filtro

# Mapa
map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Agrupamiento de marcadores
marker_cluster = MarkerCluster().add_to(map)

# Añadir elementos en lotes o grupos (start points - green)
for coords in df_filtered[['start_latitude', 'start_longitude']].values:
    folium.CircleMarker(
        location=coords,
        radius=3,
        color='green',
        fill=True
    ).add_to(marker_cluster)

# Añadir elementos en lotes o grupos (end points - red)
for coords in df_filtered[['end_latitude', 'end_longitude']].values:
    folium.CircleMarker(
        location=coords,
        radius=3,
        color='red',
        fill=True
    ).add_to(marker_cluster)

# Líneas de trayectoria
for _, row in df_filtered.iterrows():
    folium.PolyLine(
        locations=[(row['start_latitude'], row['start_longitude']), 
                  (row['end_latitude'], row['end_longitude'])],
        color='blue',
        weight=1
    ).add_to(map)

# Display the map in Streamlit
st.subheader(f'Mapa de Tornados - Año {year_filter}')
st.markdown(f'Este mapa muestra la ubicación de los tornados y sus trayectorias para el año {year_filter}.')
st.components.v1.html(map._repr_html_(), height=500)


#------------------------------------------- MACHINE LEARNING

st.markdown('''
----------
## 6. Machine Learning Models & Experiments
            
-----------            
            ''')

st.markdown("""
Probamos múltiples modelos de aprendizaje automático para predecir la magnitud de los terremotos. Los tres modelos que evaluamos fueron:

- **LSTM (Long Short-Term Memory)**
- **Gradient Boosting**
- **Random Forest**

Todos los modelos fueron probados utilizando el mismo conjunto de parámetros de entrada:
- `month`
- `state`
- `region`
- `start latitude`
- `start longitude`
- `length`
- `width`

La **variable objetivo** para todos los modelos fue:
- `magnitude`

----------            
""")

st.markdown("""
### ***📊 Comparación de Modelos de Clasificación para Predicción de Magnitud de Tornados***

| Modelo                 | Accuracy | F1-Score (Macro) | Ventajas                                                       | Desventajas                                                 |
|------------------------|----------|------------------|----------------------------------------------------------------|-------------------------------------------------------------|
| **LSTM**               | 0.46     | 0.24             | Captura secuencias temporales y relaciones espaciales complejas | Bajo rendimiento general, especialmente en clases minoritarias |
| **Gradient Boosting**  | 0.64     | 0.51             | Buen manejo de datos no lineales, rápido de entrenar            | Baja precisión para clases minoritarias                    |
| **Random Forest**      | 0.64     | 0.54             | Precisión general más alta, robusto y fácil de interpretar       | Menor capacidad para captar secuencias o patrones temporales|

🔍 **Observación Final**: El modelo **Random Forest optimizado** sigue siendo el más robusto para esta tarea. El **LSTM** no logró generalizar bien en el conjunto de prueba, especialmente con clases desbalanceadas.

📌 Recomendación: Usar **Random Forest** para despliegue inicial en Streamlit, mientras se exploran mejoras o versiones refinadas del modelo LSTM.
""")
                   

#-----------------------------|
#-------------INPUT VARIABLES |
#-----------------------------|


#------------- MAP BOUNDS + LONG AND LAT INPUT


# Load the shapefile (USA boundary)
usa = gpd.read_file("/workspaces/4geeks_final_project/data/raw/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
usa = usa[usa['NAME'] == 'United States of America']

# Ensure the projection is in EPSG:4326 (WGS84)
usa = usa.to_crs(epsg=4326)

# Function to extract coordinates from a Polygon or MultiPolygon
def extract_coords(geom):
    coords = []
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:  # Iterate through each polygon in the MultiPolygon
            coords.extend(list(poly.exterior.coords))
    return coords

# Extract coordinates from the USA boundary
coords = extract_coords(usa.geometry.values[0])
usa_boundary = Polygon(coords) if isinstance(usa.geometry.values[0], Polygon) else MultiPolygon([Polygon(coords)])

# Streamlit input for user to enter longitude and latitude
longitude = st.number_input("Enter Longitude:", min_value=-180.0, max_value=180.0)
latitude = st.number_input("Enter Latitude:", min_value=-90.0, max_value=90.0)

# Function to check if point is inside USA landmass (not water)
def is_within_usa(long, lat):
    point = Point(long, lat)  # Create a Point object from the coordinates
    print(f"Checking point: {point}")  # Debugging output
    if usa_boundary.contains(point):
        return True  # Point is inside USA
    else:
        return False  # Point is outside USA

# Streamlit button to check if the input point is within the USA
if st.button("Check if point is within the USA"):
    if is_within_usa(longitude, latitude):
        st.success("The point is within the boundaries of the USA and on land!")
    else:
        st.error("The point is outside the USA or on water.")

    # Additional visualization to debug and ensure it's correctly plotted
    fig, ax = plt.subplots(figsize=(10, 10))
    usa.plot(ax=ax, color='lightblue')
    ax.scatter(longitude, latitude, color='red', marker='x')  # Plot the point
    plt.title(f"USA Boundary and Point Location\nLongitude: {longitude}, Latitude: {latitude}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    st.pyplot(fig)


#------------- WHAT STATE?

# Now, we will check the state the point is in (if it is inside the USA)

state_name = ''

if is_within_usa(longitude, latitude):
    # Load the shapefile for US states
    states = gpd.read_file("/workspaces/4geeks_final_project/data/raw/cb_2022_us_state_20m/cb_2022_us_state_20m.shp")

    # Ensure states shapefile is in EPSG:4326
    states = states.to_crs(epsg=4326)

    # Create the Point object from the user input coordinates
    point = Point(longitude, latitude)

    # Check which state the point is in
    state = states[states.geometry.contains(point)]

    if not state.empty:
        state_name = state.iloc[0]['NAME']
        st.write(f"The point is located in {state_name}.")
    else:
        st.error("The point is not inside any state.")

        
#------------- Convert State TO ABREV

abbrev_state =''

state_abbrev_map = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

abbrev_state = state_abbrev_map.get(state_name, "Unknown")


#------------- STATE TO REGION

region = ''

ne = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
s = ['DE', 'MD', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'LA', 'TX','DC']
mw = ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
w = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']


if abbrev_state in ne:
    region = 'e'
elif abbrev_state in s:
    region = 's'
elif abbrev_state in mw:
    region = 'mw'
elif abbrev_state in w:
    region = 'w'
else:
    region = 'Unknown'  # For states not in any region

#st.write(f"The region is {region}.")

###---------------- Month input


month_number = int

months = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

month_name = st.selectbox("Select a month:", list(months.keys()))
month_number = months[month_name]

#st.write(f"You selected: {month_name} (#{month_number})")



###--------------- Months into seasons:


season = ''

Winter = (12, 1, 2)
Spring = (3, 4, 5)
Summer = (6, 7, 8)
Fall = (9, 10, 11)



if month_number in Winter:
    season = 'Winter'
elif month_number in Spring:
    season = 'Spring'
elif month_number in Summer:
    season = 'Summer'
elif month_number in Fall:
    season = 'Fall'
else:
    season = 'Unknown'
    
st.write(f"The seasons is {season}.")



#-------------- # Average lenghth by month and state

length_average_input = float

if abbrev_state not in state_abbrev_map.values():
    # Skip processing if the state abbreviation is not valid (not in the map)
    pass
else:
    # Filter the length dataframe
    filtered_length_df = length_average_df[(length_average_df['state'] == abbrev_state) & (length_average_df['month'] == month_number)]

    # Check if there is data for length
    if filtered_length_df.empty or filtered_length_df['length'].values[0] == 0:
        st.error("No tornadoes have been recorded in this state during this month in over 5 years.")
    else:
        length_average_input = round(filtered_length_df['length'].values[0], 2)
        st.write(f"The average tornado length for this month and state is {length_average_input}.")



#-------------- # Average width by month and state

width_average_input = float

if abbrev_state not in state_abbrev_map.values():
    # Skip processing if the state abbreviation is not valid (not in the map)
    pass
else:
        # Filter the width dataframe
    filtered_width_df = width_average_df[(width_average_df['state'] == abbrev_state) & (width_average_df['month'] == month_number)]

    # Check if there is data for width
    if filtered_width_df.empty or filtered_width_df['width'].values[0] == 0:
        st.error("No tornadoes have been recorded in this state during this month in over 5 years.")
    else:
        width_average_input = round(filtered_width_df['width'].values[0], 2)
        st.write(f"The average tornado width for this month and state is {width_average_input}.")

#-------------- MODEL

st.title('Predicción de Magnitud de Tornado 🌪️') 

try:
    # Check required variables
    if (
        longitude is None or
        latitude is None or
        not abbrev_state or
        not region or
        month_number is None or
        not season
    ):
        raise ValueError("Missing input")

    # Load the trained model
    with open('/workspaces/4geeks_final_project/models/best_clf_rf.pkl', 'rb') as file:
        rf_model = pkl.load(file)
    
        # Load the LabelEncoder
    with open('/workspaces/4geeks_final_project/models/label_encoder_rf.pkl', 'rb') as f:
        label_encoder_rf = pkl.load(f)

    feature_names = rf_model.feature_names_in_
    #st.write("Expected feature order:", feature_names)

    # ----------------- Input Data -----------------
    input_data = pd.DataFrame({
        'start_latitude': [latitude],
        'start_longitude': [longitude],
        'length': [length_average_input],
        'width': [width_average_input],
        'month': [month_number],
        'season': [season],
        'state': [abbrev_state],
        'region': [region]
    })

    # One-hot encode
    categorical_cols = ['month', 'season', 'state', 'region']
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=False)

    # Add missing columns
    missing_cols = set(feature_names) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0

    # Reorder columns
    input_data_encoded = input_data_encoded[feature_names]

    # Predict on button click
    if st.button('Predecir Magnitud'):
        prediction = rf_model.predict(input_data_encoded)
        predicted_class = label_encoder_rf.inverse_transform(prediction)[0]
        st.success(f'Magnitud Predicha: {predicted_class}')

except Exception as e:
    st.error(f"YOU CAN'T BREAK ME!! Error en la predicción: {e}")


