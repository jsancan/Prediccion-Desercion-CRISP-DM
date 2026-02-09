!pip install streamlit pyngrok scikit-learn pandas matplotlib seaborn Pillow -q

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- CARGA Y PREPROCESAMIENTO ---
@st.cache_data
def datos_crudos():
    # Intentamos leer con codificación 'latin-1' para evitar el error de Unicode en Dataframe
    try:
        df = pd.read_csv('REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv', sep=';', encoding='latin-1')
    except:
        df = pd.read_csv('/content/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv', sep=';', encoding='latin-1') #**

    # Limpieza de PROMEDIO: de string con coma a float
    df['PROMEDIO'] = df['PROMEDIO'].str.replace(',', '.').astype(float)

    # Identificamos el último periodo disponible
    ultimo_periodo = df['PERIODO'].unique()[-1]
    estudiante_activo = df[df['PERIODO'] == ultimo_periodo]['ESTUDIANTE'].unique()

    # Agrupación por estudiante
    perfil_estudiante = df.groupby('ESTUDIANTE').agg({
        'PROMEDIO': 'mean',
        'ASISTENCIA': 'mean',
        'NO. VEZ': 'max',
        'NIVEL': 'max'
    }).reset_index()

    # Contar materias reprobadas
    reprobadas = df[df['ESTADO'] == 'REPROBADA'].groupby('ESTUDIANTE').size().reset_index(name='REPROBADAS')
    perfil_estudiante = perfil_estudiante.merge(reprobadas, on='ESTUDIANTE', how='left').fillna(0)

    # Definición de Deserción: 1 si no está en el último periodo, 0 si está
    perfil_estudiante['DESERCION'] = perfil_estudiante['ESTUDIANTE'].apply(lambda x: 0 if x in estudiante_activo else 1)

    return perfil_estudiante

df_procesados = datos_crudos()
imagen1 = Image.open("/content/imagen1.jpg") #**
imagen2 = Image.open("/content/imagen2.jpeg") #**

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Predicción Deserción", layout="wide")
st.markdown("<h3 style='text-align: center;'>Proyecto de Almacenes de Datos y Minería de Datos</h3>", unsafe_allow_html=True)
st.title("Sistema de Predicción de Deserción Estudiantil Aplicando Técnicas de Minería de Datos")
st.title("🎓 Predicción de Deserción Estudiantil (CRISP-DM)")

menu = st.sidebar.radio("Navegación", ["Resumen", "Análisis de Datos", "Evaluación del Modelo", "Predicción de Riesgo"])

if menu == "Resumen":
    with st.container():
      st.write("La deserción estudiantil representa uno de los principales desafíos que enfrentan las instituciones de educación superior. Identificar"
          "de manera temprana a los estudiantes con mayor riesgo de abandono permite implementar estrategias de intervención oportunas que pueden "
          " marcar la diferencia entre la permanencia y el retiro del estudiante. En este proyecto, aplicarán técnicas de minería de datos para "
          " desarrollar un modelo predictivo que permita identificar estudiantes en riesgo de deserción, utilizando datos históricos "
          " del record académico estudiantil.")
      st.write("[Mas informacion >](https://www.youtube.com/watch?v=jahs9lTcp-w)")

    with st.container():
      st.write("---")
      left_column, right_column = st.columns(2)
      with left_column:
        st.header("Objetivo")
        st.write(
          """
            Desarrollar un sistema de predicción de deserción estudiantil aplicando técnicas de minería de datos, que permita identificar estudiantes en
            riesgo y visualizar los resultados mediante una interfaz gráfica interactiva.
          """
        )
      with right_column:
        st.image(imagen1)

    with st.container():
      st.write("---")
      st.header("CRISP-DM")
      image_column, text_column = st.columns((1, 2))
      with image_column:
        st.image(imagen2)
      with text_column:
        st.write(
          """
          El uso del estándar CRISP-DM proporcionará un marco estructurado que facilitará desde la comprensión profunda de los datos históricos hasta el despliegue 
          funcional del sistema, asegurando que cada decisión técnica esté alineada con el problema de negocio.
          """
        )

    with st.container():
      st.write("---")
      st.header("Recomendaciones Futuras del proyecto")
      st.markdown(
          """
          •	Integración de Variables Cualitativas: Se recomienda ampliar el dataset en futuras fases para incluir factores socioeconómicos, niveles de satisfacción 
          estudiantil y datos sobre el bienestar emocional, permitiendo una visión más holística del estudiante.\n
          •	Implementación de un Sistema de Alertas Tempranas (SAT): Utilizar la interfaz desarrollada en Streamlit para enviar notificaciones 
          automáticas a los tutores académicos cuando un estudiante sea detectado con una probabilidad de riesgo superior al 70%.\n
          •	Reentrenamiento Periódico del Modelo: Siguiendo el ciclo continuo de CRISP-DM, se sugiere actualizar el modelo al finalizar cada periodo 
          académico para capturar cambios en las tendencias de deserción y ajustar los pesos de las variables.\n
          •	Fomento de la Cultura de Datos: Capacitar al personal administrativo y docente en el uso de la interfaz gráfica para que el modelo 
          predictivo se convierta en una herramienta de apoyo cotidiano en la toma de decisiones.\n
          •	Escalabilidad Institucional: Pilotar este sistema en otras facultades de la Universidad de Guayaquil, adaptando los hiperparámetros 
          del modelo según las particularidades de cada carrera.
          """
        )
     
elif menu == "Análisis de Datos":
    st.header("📊 Análisis Exploratorio (EDA)")
    st.write(f"Datos basados en {len(df_procesados)} perfiles estudiantiles únicos.")
    st.dataframe(df_procesados.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de la Deserción")
        fig, ax = plt.subplots()
        sns.countplot(x='DESERCION', data=df_procesados, palette='viridis', ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Relación Asistencia vs Deserción")
        fig, ax = plt.subplots()
        sns.boxplot(x='DESERCION', y='ASISTENCIA', data=df_procesados, ax=ax)
        st.pyplot(fig)

elif menu == "Evaluación del Modelo":
    st.header("🤖 Rendimiento del Modelo Predictivo")
    X = df_procesados[['PROMEDIO', 'ASISTENCIA', 'NO. VEZ', 'NIVEL', 'REPROBADAS']]
    y = df_procesados['DESERCION']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión Global (Accuracy)", f"{accuracy_score(y_test, preds):.2%}")
        st.subheader("Matriz de Confusión")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Deserta', 'Deserta'])
        disp.plot(cmap='Blues', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Importancia de Variables")
        feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
        st.bar_chart(feat_importances)

    st.text("Reporte de Clasificación Detallado:")
    st.text(classification_report(y_test, preds))

elif menu == "Predicción de Riesgo":
    st.header("🔍 Consultar Riesgo de Estudiante")
    with st.form("input_estudiante"):
        prom = st.number_input("Promedio Histórico", 0.0, 10.0, 7.5)
        asis = st.slider("% Asistencia", 0, 100, 80)
        repro = st.number_input("Total Materias Reprobadas", 0, 20, 0)
        nivel = st.number_input("Nivel Académico Actual", 1, 10, 1)
        vez = st.number_input("Máximo de veces que repitió materia", 1, 5, 1)

        btn = st.form_submit_button("Analizar Riesgo")
        if btn:
            X = df_procesados[['PROMEDIO', 'ASISTENCIA', 'NO. VEZ', 'NIVEL', 'REPROBADAS']]
            y = df_procesados['DESERCION']
            model = RandomForestClassifier().fit(X, y)
            prob = model.predict_proba([[prom, asis, vez, nivel, repro]])[0][1]

            if prob > 0.5:
                st.error(f"⚠️ RIESGO ALTO DE DESERCIÓN. Probabilidad: {prob:.2%}")
            else:
                st.success(f"✅ BAJO RIESGO. Probabilidad de deserción: {prob:.2%}")


from pyngrok import ngrok
import os

# PEGA TU TOKEN AQUÍ (entre las comillas)
ngrok.set_auth_token("39M1esx6eABdi7VnmmVd5NlbIWJ_3G5UQ8EfBfaUeGWGg2GUi") #**

# Matar túneles anteriores
ngrok.kill()

# Crear túnel
public_url = ngrok.connect(8501)
print(f"\n🚀 TU APLICACIÓN ESTÁ AQUÍ: {public_url}\n")
print("Haz clic en el enlace de arriba ☝️")

--

!streamlit run app.py --server.port 8501 --server.headless true
