
![Steam](https://github.com/karinakozlowski/MLOPS_API/raw/main/assets/Steam_Proyect_Mlops.png)
<br />
# Proyecto MLOps: Sistema de Recomendación de Videojuegos para Usuarios de Steam

<div>
    <div align='center'>
    <a href="https://kozlowskikarina.wixsite.com/mlops"_blank">
          <img  src="https://github.com/karinakozlowski/MLOPS_API/blob/main/assets/BotonAPI.png"/>
       </a>
   <a href="https://youtu.be/-OIHCNS6qLc" target="_blank">
          <img  src="https://github.com/karinakozlowski/MLOPS_API/blob/main/assets/BotonYoutube.png"/>
      </a>
      </div>
</div>
https://youtu.be/-OIHCNS6qLc










### Descripción del Proyecto


Se genero un Producto Mínimo Viable que muestre una API deployada en un servicio en la nube y la aplicación de dos modelos de Machine Learning. Este proyecto simula el rol de un MLOps Engineer, es decir, la combinación de un Data Engineer y Data Scientist, para la plataforma multinacional de videojuegos Steam.

 Se realizo un análisis de sentimientos sobre los comentarios de los usuarios de los juegos y, por otro lado, la recomendación de juegos a partir de dar el nombre de un juego y/o a partir de los gustos de un usuario en particular.

Se desarrolló un caso de negocio real utilizando conjuntos de datos públicos de la industria de videojuegos.

### Objetivo
El propósito central es la creación del primer modelo de Machine Learning (end to end) a través de un enfoque que involucra tareas de Data Engineering (ETL, EDA, API) hasta la implementación del ML. Se busca lograr un rápido desarrollo y tener un Producto Mínimo Viable (MVP).

El ciclo de vida de un proyecto de Machine Learning contempla desde el tratamiento y recolección de los datos hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.

<br />
<br />

## Etapas del Proyecto <br />
![Etapas](https://github.com/karinakozlowski/MLOPS_API/raw/main/assets/Diagrama_2.png)  
<br />

**1. Análisis Exploratorio de Datos (EDA)** <br />
Inicialmente, recibí tres (3) archivos en formato JSON, los cuales están almacenados en la carpeta **Input** de un repositorio público en **[Google Drive](https://bit.ly/3UudUxb).**

Se comienza el proyecto realizando un analisis explotatorio en los Dataset para ver que decisiones tenemos que tomar para luego en el ETL hacer las transformacion necesarias para realizar las consultas y optimizar tanto el rendimiento de la API como el entrenamiento del modelo.

Para ello se utilizó la librería Pandas para la manipulación de los datos y las librerías Matplotlib y Seaborn para la visualización.

Luego de la transformacion de los dataset se volvera hacer un EDA para investigar las relaciones entre variables, identifiqué outliers y busqué patrones interesantes en los datos. El notebook [EDA_Análisis Exploratorio de Datos](Notebooks/EDA/EDA_AnálisisExploratorioDatos.ipynb)<br />


**2. Ingeniería de Datos (ETL y API)** <br />

- **2.1 *Transformaciones de Datos:***  

Realicé transformaciones esenciales para cargar los conjuntos de datos con el formato adecuado. Estas transformaciones se llevaron a cabo con el propósito de optimizar tanto el rendimiento de la API como el entrenamiento del modelo. <br />
  + [australian_user_reviews.json](https://bit.ly/3SLt3sB): Contiene las reseñas de juegos específicamente realizadas por usuarios australianos. Se puede hacer referencia al notebook [ETL_user_reviews](Notebooks/ETL/ETL_user_review.ipynb) para obtener más detalles sobre cómo se procesaron las reseñas dando como resultado un nuevo archivo con datos limpios, [df_reviews.parquet](dataset/df_reviews.parquet).<br />
  + [output_steam_games.json](https://bit.ly/486GGHB): Este archivo proporciona información detallada sobre los juegos disponibles en la plataforma Steam. Incluye datos como géneros, etiquetas, especificaciones, desarrolladores, año de lanzamiento, precio y otros atributos relevantes de cada juego. En el notebook [ETL_steam_game](Notebooks/ETL/ETL_steam_game.ipynb) <br /> 
  + [australian_users_items.json](https://bit.ly/490VRD7): El archivo australian_users_items.json contiene información sobre los ítems relacionados con usuarios australianos. Este conjunto de datos ha pasado por un proceso de Extracción, Transformación y Carga (ETL), que se detalla en el notebook [ETL_user_items](Notebooks/ETL/ETL_user_items.ipynb). Como resultado de este proceso, se generó un nuevo archivo [df_items_developer.parquet](dataset/df_items_developer.parquet) para facilitar su manipulación y análisis, brindando así una estructura más amigable y lista para su integración en el modelo.<br />
  
- **2.2 *Feature Engineering:*** Creé la columna **``` sentiment_analysis ```** aplicando análisis de sentimiento a las reseñas de los usuarios. Se optó por utilizar la biblioteca NLTK (Natural Language Toolkit) con el analizador de sentimientos de Vader, que proporciona una puntuación compuesta que puede ser utilizada para clasificar la polaridad de las reseñas en negativas (valor '0'), neutrales (valor '1') o positivas (valor '2'). A las reseñas escritas ausentes, se les asignó el valor de '1'.
puede ver el detalle del desarrollo en el notebook [ETL_user_reviews](Notebooks/ETL/ETL_user_review.ipynb) y profundizar un poco más en el análisis en el [EDA_Análisis Exploratorio de Datos](Notebooks/EDA/EDA_AnálisisExploratorioDatos.ipynb). <br />

- **2.3 *Desarrollo de API:*** 
Implementé una API con FastAPI y se deployó en Render, ésta proporciona cinco (5) consultas sobre información de videojuegos. Puede ver el detalle del código en los notebooks [Funciones](Notebooks/FUNCIONES/Consultas.ipynb).<br />
  + Endpoint 1 (developer): Devuelve cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora<br />
  + Endpoint 2 (userdata): Devuelve cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items <br />
  + Endpoint 3 (UserForGenre): Devuelve el usuario que acumulo mas horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.<br />
  + Endpoint 4 (best_developer_year): Devuelve el top 3 de desarrolladoras con juegos MAS recomendados por usuarios para el año dado.<br />
  + Endpoint 5 (developer_reviews_analysis): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.  
  .<br />
Para acceder a la funcionalidad completa de la API y explorar las recomendaciones de juegos, puedes visitar este enlace [URL de la API](https://kozlowskikarina.wixsite.com/mlops). En este sitio, encontrarás las diversas funciones desarrolladas. ¡Disfruta explorando!.
  



**3. Modelo de Aprendizaje Automático** <br />
Creé el sistema de recomendación con uno de los enfoques propuestos:
- **3.1 *[Sistema de Recomendación ítem-ítem](Notebooks/ML/recomienda_item_item.ipynb)***: Desarrollé un modelo que recomienda juegos similares en base a un juego dado, utilizando similitud del coseno. Con CountVectorizer se convirtieron los textos de la columna 'specs' en vectores numéricos para posterior calcular la similitud del coseno.<br />
Se utilizó la métrica de **similitud del coseno**, ya que mide el coseno del ángulo entre dos vectores. Cuanto más cercano a 1, más similares son los vectores. Este método fue clave para determinar qué tan parecidos son los juegos entre sí. Esto se utiliza para generar recomendaciones, ya que los juegos con vectores similares son considerados como recomendaciones potenciales.<br />

**4. Implementación de MLOps** <br />
**Deploy del Modelo:** Desplegué el modelo de recomendación como parte de la API, la cual puedes consultar acá: **[URL de la API](https://kozlowskikarina.wixsite.com/mlops)**. 

Para el deploy de la API se seleccionó la plataforma Render que es una nube unificada para crear y ejecutar aplicaciones y sitios web, permitiendo el despliegue automático desde GitHub. 

Como se indicó anteriormente, para el despliegue automático, Render utiliza GitHub y dado que el servicio gratuito cuenta con una limitada capacidad de almacenamiento, se realizó un repositorio exclusivo para el deploy, el cual se encuenta aqui.

<br />

**5. Video Explicativo** <br />
Grabé un video explicativo que muestra el funcionamiento de la API, consultas realizadas y una breve explicación del de ML utilizado [Youtube](https://youtu.be/-OIHCNS6qLc).<br />
<br />

## Estructura del Repositorio <br />
**1. [/Notebooks](Notebooks/):** Contiene los Notebooks de python en Visual studio code, Google Colab y con el Código completo y bien comentado donde se realizaron las extracciones, transformaciones y carga de datos (ETL), análisis exploratorio de los datos (EDA).<br />

**2. [/assets](assets/):** Carpeta con imágenes y recursos utilizados en el desarrollo del proyecto.<br />

**3. [/dataset](dataset/):** Almacena los datasets utilizados en una versión limpia y procesada de los mismos. Las fuentes de datos iniciales se encuentra almacenadas en la carpeta input en el siguiente repositorio [Google Drive](https://bit.ly/3UudUxb)<br />
- ** *Archivos_API:*** Contiene los datasets en formato parquet consumidos por la API.<br />
- ** *Archivos_Limpios:*** Contiene los archivos depurados después de haber realizado el ETL.<br />
- ** *Archivos_ML:*** Contiene los archivos consumidos por la API para hacer el sistema de recomendación.<br />

<br />

## Ejecutar la API (en su máquina local) <br />
1. Clonar el repositorio <br />
```
git clone https://github.com/karinakozlowski/MLOPS_API
```
2. Crear entorno virtual<br />
```
python3 -m venv <nombre_del_entonto>
```
3. Vaya al directorio del entorno virtual y actívelo<br />
- 3.1. Para Windows:
```
Scripts/activate
```
- 3.2. Para Linux/Mac:
```
bin/activate
```
4. Instalar los requerimientos<br />
```
pip install -r requirements.txt
```
5. Ejecute la API con uvicorn<br />
```
uvicorn main:app --reload
```

## Propuesta de Mejoras:

Dado que el objetivo de este proyecto fue presentar un Producto Mínimo Viable, se realizaron algunos análisis básicos que se podrían mejorar en próximas etapas, con la idea de lograr un producto completo. Por ejemplo:

  + Análsis de sentimiento: se contemplo solo los mensajes en ingles, pero habian de diferentes idiomas.Por otra parte, se puede evaluar el rendimiento del modelo probando con distintos umbrales de clasificación.

  + Modelos de recomendación: se puede crear un rating que considere la influencia de las horas de juego de los usuarios, la utilizadad hacia otros usuarios de los comentarios, el precio de los juegos, entre otras variables. También se podrían evaluar otras librerías que realizar este tipo de modelos.

  + EDA más exhaustivo: se puede hacer un análisis exploratorio de datos mas exhaustivo, buscando mas relaciones entre los juego y usarios que permitan crear un puntaje mas representativo para hacer las recomendaciones.

  + ETL más exhaustivo: se pueden haces más transformaciones en algunas variables usadas en la API, como por ejemplo precios, donde muchos campos tenían palabras y solo se cambió por precio cero, porque muchos textos se referian a juegos gratuitos, pero no se observó en detalle. También había datos faltantes que se completaron con 0, pero no se investigó si eran juegos gratuitos. Esto puede afectar a los resultados de la API donde pregunta por porcentaje de juegos gratuitos.

  + Otros servicios de nube: se pueden investigar otras formas de deployar la API de modo de no tener las limitaciones de capacidad de almacenamiento y poder utilizar la última función del modelo de recomendación o buscar alternativas para almacenar los datos por fuera de Render y conectar con esa fuente para las consultas.


## Autor <br />

<div align="center">
  <a href="https://www.linkedin.com/in/karina-kozlowski-625535217/" target="_blank">
    <img src="https://avatars.githubusercontent.com/u/838109" width="200" alt="Karina Kozlowski">
  </a>
  <br>
  <span>Karina Kozlowski</span>
  <br>
  <span>Role: Machine Engineer</span>
  <br>
  <a href="https://www.linkedin.com/in/karina-kozlowski-625535217/" target="_blank">
    <img src="https://img.shields.io/badge/linkedin%20-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href='mailto:kozlowskikarina@gmail.com'>
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/>
  </a>
</div>



## Karina Kozlowski. <br />
Para cualquier duda/sugerencia/recomendación/mejora respecto al proyecto con toda libertad puedes contactarme por [LinkedIn](https://bit.ly/3waAAs6)<br />



