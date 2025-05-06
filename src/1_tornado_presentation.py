import geopandas as gpd
import shapely
import streamlit as st
from streamlit_folium import st_folium
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

def slide_1_welcome():
    st.markdown(
        """
        <style>
        .container2 {
            margin-bottom: 30px;
        }

        .presenters-container {
            position: relative;
            margin-left: 0px;
            margin-top: 20px;
            width: fit-content;
        }

        .presenters-container h2,
        .presenters-container h3 {
            margin: 0;
        }

        .small-text-container {
            position: relative;
            margin-left: 0px;
            margin-top: 5px;
            width: fit-content;
        }

        .small-text {
            font-size: 15px;
            color: #888;
        }

        .image-container {
            position: relative;
            margin-top: -200px;
            margin-left: 350px;
            width: fit-content;
        }

        .image-container img {
            max-width: 650px;
            height: auto;
            border-radius: 10px;
        }
        </style>

        <div class="container2">
            <h1>Predicción de Magnitud de Tornados en los E.U.A.</h1>
        </div>

        <div class="presenters-container">
            <h2>Presentation by:</h2>
            <h3>• Luis Alpízar<br>• Jia</h3>
        </div>

        <div class="small-text-container">
            <span class="small-text">Optimizado para el modo ancho de streamlit</span>
        </div>

        <div class="image-container">
            <img src="https://s3.eu-west-2.amazonaws.com/sr-acf-craft/v2/images/_1920x1080_crop_center-center_line/Tornado-and-lightning.jpg" alt="Presenters">
        </div>
        """,
        unsafe_allow_html=True
    )

def slide_2_intro():
    st.markdown(
        """
        <style>
        .container-title {
            margin-bottom: 30px;
        }

        .text-container {
            position: relative;
            margin-left: 0px;
            margin-top: 20px;
            width: fit-content;
            font-size: 30px;
        }

        .image-container {
            position: relative;
            margin-left: 600px;
            margin-top: -300px;
            width: fit-content;
        }

        .image-container img {
            max-width: 500px;
            height: auto;
            border-radius: 10px;
        }
        </style>

        <div class="container-title">
            <h1>Introducción</h1>
        </div>

        <div class="text-container">
            <text>En este proyecto desarrollamos modelos <br>de Machine Learning y Deep Learning para <br>predecir la magnitud de tornados en <br>Estados Unidos, utilizando el conjunto de <br>datos histórico: <br><br>'US Tornado Dataset 1950–2021 (CSV)'.</text>
        </div>

        <div class="image-container">
            <img src="https://as2.ftcdn.net/jpg/06/71/22/41/1000_F_671224189_EzhlsPoNBHikN87UJ56cm3ePMVSzByhy.jpg" alt="Tornado visual">
        </div>
        """,
        unsafe_allow_html=True
    )



def slide_3_tornado():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            width: 100%;
        }
        .text {
            flex: 1;
        }
        .text1 {
            flex: 1;
            font-size: 20px;
        }
        .image img {
            max-width: 550px;
            height: auto;
            border-radius: 10px;
            margin-top: -375px;
            margin-left: 525px; 
        }
        </style>
        <div class="container2">
            <h1>¿Qué es un Tornado?</h1>
        </div>
        <div class="container1">
            <div class="text1">
                <ul>
                    <li><strong>Condiciones Climáticas:</strong> Un tornado se forma cuando <br>hay fuertes tormentas con aire cálido y húmedo que <br>choca con aire frío y seco, creando inestabilidad <br>atmosférica.</li>
                    <li><strong>Condiciones Regionales:</strong> Los tornados ocurren con <br>mayor frecuencia en regiones planas donde se <br>encuentran masas de aire cálido del sur con aire frío <br>del norte.</li>
                    <li><strong>Forma:</strong> A menudo de embudo característica.</li>
                </ul>
            </div>
            <div class="image">
                <img src="https://www.fundacionaquae.org/wp-content/uploads/2020/08/qu-es-y-cmo-se-forma-un-tornado1-1024x597.jpg" alt="Tornado image">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




def slide_4_f_ef():
    st.markdown(f"""
    ## EF/F Explicacción
      
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
    """)
    #------ Pic Controllssssss
    st.markdown(
        """
        <style>
        .custom-image-container {
            position: relative;
            width: 100%;
            height: 600px;
        }

        .custom-image-container img {
            margin-top: -325px;
            margin-left: 625px; 
            width: 400px;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            
        }
        </style>

        <div class="custom-image-container">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_LOzt4lZV-Mqw2DOFuB0c2IW-GUhClhno_w&s" alt="Tornado image">
        </div>
        """,
        unsafe_allow_html=True
    )

def slide_5_eda_1():
    st.markdown(f"""
    ## EDA - Frecuencia pt 1 
    #### Los tornados son más frecuentes en abril, mayo, junio y hasta julio,  
    #### ya que en esos meses se dan las condiciones climáticas ideales para su formación.
    """)
    # Frecuencia de Tornados por Mes - HISTOGRAM

    plt.style.use('ggplot')
    plt.figure(figsize=(15, 5))
    plt.title('📊 Frecuencia de Tornados por Mes')
    df_train['month'].value_counts().sort_index().plot(kind='bar')
    st.pyplot(plt)




def slide_6_eda_2():
    st.markdown(f"""
    ## EDA - Frecuencia pt 2
    - #### **Texas**, **Kansas**, **Oklahoma**, **Alabama**, **Mississippi**, **Missouri**, **Louisiana**, **Illinois** y **Iowa** tienen las tasas más altas de tornados.
    - #### Estos estados experimentan tornados debido a condiciones climáticas y topográficas específicas.
    - #### Estas regiones forman parte de **Tornado Alley**, una zona conocida por su vulnerabilidad a los tornados debido a su geografía y patrones climáticos únicos.

    """)

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


def slide_7_eda_3():
    st.markdown(f"""
    ## EDA - Tornado Concentration 2017-2021
    """)

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


def slide_8_variables():
    st.markdown(f"""
    ## Variables para Entrenar Modelos
    ### Todos los modelos fueron probados utilizando el mismo conjunto de parámetros de entrada:
    - #### `month`
    - #### `state`
    - #### `region`
    - #### `start latitude`
    - #### `start longitude`
    - #### `length`
    - #### `width`

    ### El **variable objetivo** para todos los modelos fue:
    - #### `magnitude`
    """)
    #----- Image Controlllllssssssss
    st.markdown(
    """
    <style>
    .image-container {
        position: absolute;
        top: 200px;
        left: 400px;
        width: fit-content;
    }
    .image-container img {
        margin-top: -1200px;
        margin-left: 50px; 
        width: 600px;
        height: auto;
        border-radius: 10px;
    }
    </style>

    <div class="image-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzJwllMIALciB8qE_9Vmsg7j2rEL369y0O1w&s" alt="Tornado image">
    </div>
    """,
    unsafe_allow_html=True
)


def slide_9_length_width():
    st.markdown(
        """
        <div style="margin-top: 20px; margin-left: 20px;">

        <h2 style="font-size: 28px;">Estimación de Dimensiones del Tornado para Entrada al Modelo</h2>

        <h4 style="font-size: 20px;"><strong>Problema</strong>:</h4>
        <ul>
            <li><h5 style="font-size: 16px;">La trayectoria y el ancho de un tornado solo se pueden medir <br>después del evento.</h5></li>
        </ul>

        <h4 style="font-size: 20px;"><strong>Desafío</strong>:</h4>
        <ul>
            <li><h5 style="font-size: 16px;">El usuario no puede ingresar estos valores en tiempo <br>real durante una predicción.</h5></li>
        </ul>

        <h4 style="font-size: 20px;"><strong>Solución</strong>:</h4>
        <ul>
            <li><h5 style="font-size: 16px;">Se usaron <strong>promedios históricos de los últimos 6 años</strong> <br>para la longitud y el ancho.</h5></li>
            <li><h5 style="font-size: 16px;">Los promedios están agrupados por <strong>estado</strong> y <br><strong>mes</strong> para mayor relevancia contextual.</h5></li>
        </ul>

        <h4 style="font-size: 20px;"><strong>Manejo de Casos Especiales</strong>:</h4>
        <ul>
            <li><h5 style="font-size: 16px;">Algunos estados/meses tenían <strong>cero tornados registrados</strong> <br>en los últimos 6 años.</h5></li>
        </ul>
        <h4 style="font-size: 20px;">En esos casos, el modelo lanza una <strong>salida previa</strong>:</h4>
        <em style="font-size: 16px;">“Baja probabilidad de tornado: sin registros en este estado o mes durante los últimos cinco años.”</em>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <div style="margin-top: -600px; margin-left: 500px;">
        <img src="https://t4.ftcdn.net/jpg/00/02/97/69/360_F_2976920_wqDjbyVWWYZaSyLV1Gvnf2vnoCDj6z.jpg" 
             style="width: 400px; height: auto; border-radius: 8px;" />
    </div>
    """,
    unsafe_allow_html=True
    )



def slide_10_models():
    st.markdown('## Modelos utilizados')
    st.markdown("""
        <style>
            .big-text {
                font-size: 29px;
                font-weight: bold;
            }
            .normal-text {
                font-size: 20px;
            }
            .container {
                margin-top: 20px;
                margin-left: 20px;
            }
        </style>

        <div class="container">
            <div class="big-text">LSTM:</div>
            <div class="normal-text">
                - Un modelo de aprendizaje profundo ideal para predecir secuencias, <br>capturando patrones a lo largo del tiempo.
            </div>
            <div class="big-text"><br>Gradient Boosting:</div>
            <div class="normal-text">
                - Una técnica de conjunto que construye varios modelos para reducir <br>el error y mejorar la precisión.
            </div>
            <div class="big-text"><br>Random Forest:</div>
            <div class="normal-text">
                - Un método de conjunto que utiliza múltiples árboles de decisión <br>para mejorar la robustez y precisión de las predicciones.
            </div>
            <div class="big-text"><br>¿Por qué funciona para tornados?:</div>
            <div class="normal-text">
                Estos modelos manejan relaciones complejas y no lineales, así como datos <br>que varían con el tiempo, lo cual es crucial para una predicción precisa de tornados.
            </div>
        </div>
    """, unsafe_allow_html=True
    )

    st.markdown(
    """
    <div style="margin-top: -550px; margin-left: 650px;">
        <img src="https://imageio.forbes.com/specials-images/imageserve/628b8de7a18d8436b8782e88//960x0.jpg?height=473&width=711&fit=bounds" 
             style="width: 400px; height: auto; border-radius: 8px;" />
    </div>
    """,
    unsafe_allow_html=True
    )


def slide_11_results():
    st.markdown(f"""
    ## Results Summary  
    | Modelo                 | Accuracy | F1-Score (Macro) | Ventajas                                                       | Desventajas                                                 |
    |------------------------|----------|------------------|----------------------------------------------------------------|-------------------------------------------------------------|
    | **LSTM**               | 0.46     | 0.24             | Captura secuencias temporales y relaciones espaciales complejas | Bajo rendimiento general, especialmente en clases minoritarias |
    | **Gradient Boosting**  | 0.64     | 0.51             | Buen manejo de datos no lineales, rápido de entrenar            | Baja precisión para clases minoritarias                    |
    | **Random Forest**      | 0.64     | 0.54             | Precisión general más alta, robusto y fácil de interpretar       | Menor capacidad para captar secuencias o patrones temporales|
    """)

    st.markdown(
    """
    <div style="margin-top: 50px; margin-left: 300px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5SGFq86NWk8-ZclDX20GwbjXGAQ1W2Zrq6Q&s" 
                style="width: 500px; height: auto; border-radius: 8px;" />
    </div>
    """,
    unsafe_allow_html=True
    )




def slide_12_prediction():
    st.markdown(f"""
    ## Predicción  
    Pruebe nuestra herramienta de predicción para ver el resultado del modelo en función de las entradas del usuario.
    """)

    #-----------------------------|
    #-------------INPUT VARIABLES |
    #-----------------------------|


    #------------- MAP BOUNDS + LONG AND LAT INPUT

    longitude = float
    latitude = float
    # Load and filter the USA shapefile
    usa = gpd.read_file("/workspaces/4geeks_final_project/data/raw/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    usa = usa[usa['NAME'] == 'United States of America'].to_crs(epsg=4326)

    # Extract boundary geometry
    def extract_coords(geom):
        coords = []
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                coords.extend(list(poly.exterior.coords))
        return coords

    coords = extract_coords(usa.geometry.values[0])
    usa_boundary = Polygon(coords) if isinstance(usa.geometry.values[0], Polygon) else MultiPolygon([Polygon(coords)])

    # Initialize Folium map centered on the US
    m = folium.Map(location=[39.5, -98.35], zoom_start=4)

    # Add click functionality to the map
    m.add_child(folium.LatLngPopup())

    # Define this function at the top level
    def is_within_usa(long, lat):
        if long is None or lat is None:
            return False  # Don't process if longitude or latitude is None
        point = Point(long, lat)
        return usa_boundary.contains(point)

    # Render the map and capture click input
    st.markdown("### Haga clic en el mapa para seleccionar una ubicación")
    map_data = st_folium(m, width=1200, height=500)

    longitude = latitude = None

    # Extract coordinates if a location was clicked
    if map_data and map_data.get("last_clicked"):
        latitude = map_data["last_clicked"]["lat"]
        longitude = map_data["last_clicked"]["lng"]
        st.write(f"Longitude: {longitude}, Latitude: {latitude}")

        # Check if the point is within the USA boundary
        if is_within_usa(longitude, latitude):
            st.success("✅ El punto está dentro de los EE.UU. (en tierra)")
        else:
            st.error("❌ El punto está fuera de los EE.UU. o en el oceano.")


    else:
        st.warning("Por favor haga clic en el mapa para seleccionar una ubicación.")


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
            st.write(f"La ubicación se encuentra en {state_name}.")
        else:
            st.error("La ubicación no está dentro de ningún estado.")

            
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
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
        "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
        "Setiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }

    month_name = st.selectbox("Seleccione un mes:", list(months.keys()))
    month_number = months[month_name]

    #st.write(f"You selected: {month_name} (#{month_number})")



    ###--------------- Months into seasons:


    season = ''

    Winter = (12, 1, 2)
    Spring = (3, 4, 5)
    Summer = (6, 7, 8)
    Fall = (9, 10, 11)



    if month_number in Winter:
        season = 'Invierno'
    elif month_number in Spring:
        season = 'Primavera'
    elif month_number in Summer:
        season = 'Verano'
    elif month_number in Fall:
        season = 'Otoño'
    else:
        season = 'Unknown'
        
    st.write(f"La estación es {season}.")



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
            st.error("No se han registrado tornados en este estado durante este mes en más de 5 años. Baja probabilidad de tornado.")
        else:
            length_average_input = round(filtered_length_df['length'].values[0], 2)
            st.write(f"El largo de la trayectoria promedio de un tornado para este mes y estado es {length_average_input} millas.")



    #-------------- # Average width by month and state

    width_average_input = float

    if abbrev_state not in state_abbrev_map.values():
        # Skip processing if the state abbreviation is not valid (not in the map)
        pass
    else:

        filtered_width_df = width_average_df[(width_average_df['state'] == abbrev_state) & (width_average_df['month'] == month_number)]

    # Check if there is data for length
        if filtered_width_df.empty or filtered_width_df['width'].values[0] == 0:
            pass
        else:
            width_average_input = round(filtered_width_df['width'].values[0], 2)
            st.write(f"El ancho promedio de un tornado para este mes y estado es {width_average_input} yardas.")

        
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
        st.error(f"Error en la predicción: {e}")


# -------------------- Slide Navigator

# Initialize session state for slide index
if 'slide' not in st.session_state:
    st.session_state.slide = 0

# Define your slides as function references (don't call them!)
slides = [
    slide_1_welcome,
    slide_2_intro,
    slide_3_tornado,
    slide_4_f_ef,
    slide_5_eda_1,
    slide_6_eda_2,
    slide_7_eda_3,
    slide_8_variables,
    slide_9_length_width,
    slide_10_models,
    slide_11_results,
    slide_12_prediction
]
# ------- Buttons

col1, col2, col3 = st.columns([1, 6, 1])

# Scaling options
scale_factor = 0.8  # Adjust scale factor as needed (example: 80% of available space)

with col1:
    if st.button("⬅️") and st.session_state.slide > 0:
        st.session_state.slide -= 1

with col3:
    if st.button("➡️") and st.session_state.slide < len(slides) - 1:
        st.session_state.slide += 1

# Container for scaling the content dynamically
with col2:
    # Apply scaling directly to the content (set the width based on scale_factor)
    st.markdown(
        f"""
        <style>
        .scaled-container {{
            width: {scale_factor * 100}%;  /* Scale content by the factor */
            margin: 0 auto;
        }}
        </style>
        <div class="scaled-container">
        """,
        unsafe_allow_html=True
    )

    # ✅ Display the current slide
    slides[st.session_state.slide]()

    # Close the scaled container
    st.markdown("</div>", unsafe_allow_html=True)



