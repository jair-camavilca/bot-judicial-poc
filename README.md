# Bot Judicial: PoC de Clasificación Legal con NLP

[![SENATI](https://img.shields.io/badge/SENATI-00529B?style=for-the-badge)](https://www.senati.edu.pe/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
![Estado](https://img.shields.io/badge/Estado-PoC%20completada-2EA44F?style=for-the-badge)

Prueba de Concepto (**PoC**) desarrollada en Python para la clasificación y derivación automática de consultas ciudadanas en el ámbito judicial. Mediante técnicas de **Procesamiento de Lenguaje Natural (NLP)** y algoritmos de **Machine Learning supervisado**, el sistema categoriza el nivel de urgencia (`Alta`, `Media`, `Baja`) para su posterior atención procesal o administrativa.

---

## Alcance de la PoC

* **Dataset de Validación:** Lote balanceado de consultas procesales y administrativas con simulación controlada de ruido léxico y valores nulos.
* **Pipeline de Preprocesamiento:** Normalización léxica, limpieza con expresiones regulares y filtrado de *stopwords* en español con mecanismo de contingencia (*fallback*).
* **Vectorización y Modelado:** Conversión de texto a representaciones matriciales mediante `TfidfVectorizer` y análisis de similitud espacial con producto punto (similitud coseno).
* **Benchmarking Multimodelo:** Evaluación empírica de desempeño entre clasificadores supervisados:
  * Naive Bayes Multinomial (`MultinomialNB`).
  * Regresión Logística (`LogisticRegression`).
  * Árbol de Decisión (`DecisionTreeClassifier`).
* **Inferencia y Triaje Interactivo (`app.py`):** Asistente conversacional en consola que carga los artefactos serializados (`.pkl`), gestiona entradas fuera de vocabulario (OOV) con opción de derivación a asesores humanos y ejecuta flujos de respuesta según el nivel de urgencia predicho.

---

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **NLP & Machine Learning:** scikit-learn, NLTK
* **Manipulación de Datos:** pandas, numpy, scipy
* **Visualización:** matplotlib, seaborn
* **Serialización de Artefactos:** joblib

---

## Estructura del Repositorio

```text
bot-judicial-poc/
├── modelos_exportados/
│   ├── clasificador_lr.pkl       # Clasificador logístico entrenado
│   └── vectorizador_tfidf.pkl    # Vectorizador TF-IDF ajustado
├── app.py                         # Asistente interactivo en terminal
├── pipeline_nlp.ipynb             # Pipeline EDA, entrenamiento y métricas
├── requirements.txt               # Dependencias del entorno
└── .gitignore                     # Exclusión de venv y temporales
```

---

## Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/jair-camavilca/bot-judicial-poc.git
```
Entra a la carpeta del proyecto:
```bash
cd bot-judicial-poc
```
### 2. Crear y activar entorno virtual
Crear el entorno (Windows):
```bash
python -m venv venv
```
Activar el entorno:
```bash
venv\Scripts\activate
```
### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
### 4. Ejecutar el asistente interactivo
```bash
python app.py
```
### 5. Explorar el pipeline de entrenamiento
Para inspeccionar el análisis exploratorio, las matrices y las gráficas comparativas:
```bash
jupyter notebook pipeline_nlp.ipynb
```

---

### Contexto Académico
* Proyecto personal (4to ciclo)





