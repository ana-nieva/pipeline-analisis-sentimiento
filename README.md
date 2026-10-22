# Proyecto Integrador: Análisis de Sentimiento (Strata)

**Institución:** UNSTA - Ingeniería en Inteligencia Artificial  
**Asignatura:** Taller II  
**Estudiante:** Ana Nieva Pastoriza  

---

##  Objetivo del Proyecto

Este repositorio documenta el desarrollo de un pipeline completo de Procesamiento de Lenguaje Natural (NLP) orientado a la clasificación de reseñas de usuarios. El flujo de trabajo abarca la ingesta de datos a través de una API, el preprocesamiento exhaustivo del texto (manejando casos límite como las negaciones), la generación de representaciones matemáticas (Bag of Words, TF-IDF, y Embeddings pre-entrenados con Hugging Face) y la futura evaluación de modelos sobre un set de datos adversarial.

---

##  Organización del Código

El proyecto sigue una arquitectura modular orientada a los pasos del pipeline de datos:

```text
pipeline-analisis-sentimiento/
├── datasets/                 # Almacenamiento local de CSVs y matrices (ignorado en Git)
├── entregables/              # Documentación generada automáticamente (ej. tamaño de vocabularios)
├── experimentacion/          # Notebooks (.ipynb) para pruebas de concepto y EDA
├── pipeline/                 
│   ├── 01_ingesta.py         # Script de conexión a la API y descarga masiva
│   ├── 02_preprocesamiento.py # Limpieza, expansión de contracciones y lematización
│   ├── 03_vectorizacion.py   # Implementación de BoW y TF-IDF (Scikit-Learn)
│   └── 04_embeddings.py      # Vectorización densa utilizando modelos de SentenceTransformers
├── .gitignore                # Reglas de exclusión para datos pesados
├── requirements.txt          # Control de dependencias de Python
└── README.md                 # Documentación del repositorio
```

---

##  Configuración Local

Para ejecutar este proyecto en tu entorno local, se recomienda el uso de **venv** para aislar las librerías instaladas.

1. **Clonación:**
   ```bash
   git clone https://github.com/TU_USUARIO/pipeline-analisis-sentimiento.git
   cd pipeline-analisis-sentimiento
   ```

2. **Creación y activación del entorno (Windows):**
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Instalación de paquetes requeridos:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  1. Extracción de Datos

El corpus original se obtiene consumiendo la API proporcionada por la cátedra. El proceso se automatiza ejecutando:

```bash
python pipeline/01_ingesta.py
```

**Detalles de la extracción:**
* Se descargan **210.000 reseñas** perfectamente balanceadas en 5 clases.
* El código maneja la paginación a través del parámetro `offset` y cuenta con un mecanismo de reintentos (`try-except`) para superar posibles *timeouts* o errores del servidor (como *Cold Starts* en Render).
* **Nota de Seguridad:** El archivo resultante (`datasets/amazon_reviews_raw.csv`) pesa más de 100MB, por lo cual está explícitamente excluido en el `.gitignore`.

---

##  2. Preprocesamiento de Texto

```bash
python pipeline/02_preprocesamiento.py
```

Este script transforma el texto crudo preparándolo para el modelado:
* **Filtro de Clases:** Se eliminan las reseñas neutrales y se agrupan las restantes en un problema binario (`0` = negativo, `1` = positivo).
* **Gestión de Negaciones:** A diferencia de una limpieza estándar, se expanden las contracciones (`can't` a `can not`) y se preservan palabras clave como `not`, `no` y `nor` eliminándolas de la lista de *stopwords* de NLTK. Esto es vital para no alterar la semántica original y superar las pruebas del banco adversarial.
* **Normalización:** Se aplica tokenización, eliminación de caracteres no alfabéticos y lematización (WordNet).

---

##  3. Representación Numérica

```bash
python pipeline/03_vectorizacion.py
python pipeline/04_embeddings.py
```

La estrategia de vectorización previene la fuga de datos (*data leakage*) realizando la partición de validación (Train 80% / Test 20%) **antes** de ajustar los modelos.

| Método Aplicado | Librería / Herramienta | Entrada Requerida | Formato de Salida |
| :--- | :--- | :--- | :--- |
| **BoW (Bag of Words)** | `CountVectorizer` | Texto limpio | Matriz dispersa (`.npz`) |
| **TF-IDF (1-grama)** | `TfidfVectorizer` | Texto limpio | Matriz dispersa (`.npz`) |
| **TF-IDF (2-gramas)** | `TfidfVectorizer` | Texto limpio | Matriz dispersa (`.npz`) |
| **Embeddings Densos** | `SentenceTransformer (MiniLM)` | Texto limpio | Matriz densa (`.npy`) |

>  Los reportes detallando el tamaño de vocabulario de cada representación se generan dinámicamente y se alojan en la carpeta `/entregables`.

---
