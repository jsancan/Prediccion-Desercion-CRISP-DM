# 🎓 Sistema de Predicción de Deserción Estudiantil

Este proyecto aplica técnicas de **Minería de Datos** siguiendo la metodología **CRISP-DM** para identificar estudiantes en riesgo de deserción utilizando su historial académico. El objetivo es permitir a las instituciones de educación superior implementar estrategias de intervención oportunas.

## 📋 Objetivos del Proyecto
* **General**: Desarrollar un sistema de predicción de deserción estudiantil aplicando técnicas de minería de datos, que permita identificar estudiantes en riesgo y visualizar los resultados mediante una interfaz gráfica interactiva.
* **Específicos**: 
  * Realizar un análisis exploratorio del conjunto de datos (EDA).
  * Identificar y seleccionar las variables más relevantes para la predicción.
  * Definir la variable objetivo (deserción) a partir de los datos históricos.
  * Aplicar técnicas de preprocesamiento y transformación de datos.
  * Construir y evaluar el modelo de clasificación.
  * Desarrollar una interfaz gráfica utilizando **Streamlit**.

## 🛠️ Requerimientos Técnicos
El proyecto se desarrolló bajo los siguientes requerimientos técnicos:
- **Lenguaje**: Python.
- **Interfaz**: Streamlit (obligatorio).
- **Metodología**: CRISP-DM (Cross-Industry Standard Process for Data Mining).
- **Librerías principales**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn` y `Pillow`.

## 🚀 Funcionalidades de la Interfaz
La aplicación incluye:
1. **Visualización del EDA**: Gráficos y estadísticas descriptivas.
2. **Métricas de Evaluación**: Presentación de Accuracy, Precisión, Recall, F1-score y Matriz de Confusión.
3. **Predicción Individual**: Formulario para ingresar datos de un estudiante y obtener su riesgo de deserción.
4. **Importancia de Variables**: Visualización de los factores que más influyen en la predicción.

## 📦 Entregables
* Informe técnico en formato PDF siguiendo CRISP-DM.
* Código fuente completo y comentado.
* Aplicación Streamlit funcional.
