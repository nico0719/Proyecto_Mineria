# -*- coding: utf-8 -*-
"""
Proyecto final - Minería de Datos
Iris Species Classification
Dashboard desarrollado en Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# Configuración general
# =========================================================
st.set_page_config(
    page_title="Clasificación de Especies Iris",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Carga de datos y entrenamiento
# =========================================================
@st.cache_data
def cargar_datos():
    iris_data = load_iris()
    datos = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    datos["species_id"] = iris_data.target
    datos["species"] = pd.Categorical.from_codes(
        iris_data.target,
        iris_data.target_names,
    )
    return iris_data, datos


@st.cache_resource
def preparar_modelo():
    iris_data, datos = cargar_datos()

    datos_modelo = datos.drop_duplicates().reset_index(drop=True)
    x = datos_modelo[iris_data.feature_names]
    y = datos_modelo["species_id"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    escalador = StandardScaler()
    x_train_scaled = escalador.fit_transform(x_train)
    x_test_scaled = escalador.transform(x_test)

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    modelo.fit(x_train_scaled, y_train)

    y_pred = modelo.predict(x_test_scaled)

    metricas = {
        "exactitud": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    reporte = classification_report(
        y_test,
        y_pred,
        target_names=iris_data.target_names,
        output_dict=True,
    )

    matriz = confusion_matrix(y_test, y_pred)

    importancia = pd.DataFrame(
        {
            "Variable": iris_data.feature_names,
            "Importancia": modelo.feature_importances_,
        }
    ).sort_values("Importancia", ascending=False)

    return iris_data, datos, modelo, escalador, metricas, reporte, matriz, importancia


iris, df, modelo, escalador, metricas, reporte, cm, importancias = preparar_modelo()


# =========================================================
# Encabezado
# =========================================================
st.title("🌸 Clasificación de Especies Iris")
st.markdown(
    "Aplicación interactiva para explorar el dataset Iris, evaluar el desempeño de un modelo "
    "de clasificación y realizar predicciones a partir de las medidas del sépalo y del pétalo."
)
st.caption(
    "**Integrantes:** Nicolle Trujillo Albor | David Calderón | "
    "Lopez Dahl Mariela Catalina | Juan Esteban Jiménez López | Jorge Eliecer de la Hoz Epiayu"
)
st.caption("Minería de Datos · Universidad de la Costa · Prof. José Escorcia-Gutierrez, Ph.D.")

st.divider()


# =========================================================
# Sidebar - parámetros de entrada
# =========================================================
st.sidebar.header("Datos de entrada")
st.sidebar.write("Ingrese las medidas de la flor en centímetros.")

longitud_sepalo = st.sidebar.number_input(
    "Longitud del sépalo (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1,
)

ancho_sepalo = st.sidebar.number_input(
    "Ancho del sépalo (cm)",
    min_value=2.0,
    max_value=5.0,
    value=3.5,
    step=0.1,
)

longitud_petalo = st.sidebar.number_input(
    "Longitud del pétalo (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1,
)

ancho_petalo = st.sidebar.number_input(
    "Ancho del pétalo (cm)",
    min_value=0.1,
    max_value=3.0,
    value=0.2,
    step=0.1,
)

nueva_muestra = pd.DataFrame(
    [[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]],
    columns=iris.feature_names,
)

nueva_muestra_scaled = escalador.transform(nueva_muestra)
prediccion_id = modelo.predict(nueva_muestra_scaled)[0]
prediccion_proba = modelo.predict_proba(nueva_muestra_scaled)[0]
prediccion_nombre = iris.target_names[prediccion_id]

st.sidebar.divider()
st.sidebar.subheader("Resumen de entrada")
st.sidebar.dataframe(nueva_muestra, use_container_width=True, hide_index=True)


# =========================================================
# Métricas del modelo
# =========================================================
st.subheader("Desempeño general del modelo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Exactitud", f"{metricas['exactitud']:.2%}")
col2.metric("Precisión", f"{metricas['precision']:.2%}")
col3.metric("Recall", f"{metricas['recall']:.2%}")
col4.metric("F1-Score", f"{metricas['f1']:.2%}")

st.caption(
    "Métricas calculadas sobre el conjunto de prueba. Modelo utilizado: Random Forest Classifier."
)

st.divider()


# =========================================================
# Predicción interactiva
# =========================================================
st.subheader("Predicción interactiva")

col_resultado, col_grafico = st.columns([1, 2])

with col_resultado:
    st.markdown("#### Predicción obtenida")
    st.success(f"La especie predicha es: **{prediccion_nombre}**")

    proba_df = pd.DataFrame(
        {
            "Especie": iris.target_names,
            "Probabilidad": prediccion_proba,
        }
    )

    fig_proba = px.bar(
        proba_df,
        x="Especie",
        y="Probabilidad",
        text_auto=".1%",
        labels={"Probabilidad": "Probabilidad estimada"},
        title="Probabilidad por clase",
    )
    fig_proba.update_layout(
        height=330,
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_proba, use_container_width=True)

with col_grafico:
    df_grafico = df.copy()
    df_grafico["Tipo"] = "Dataset"

    nueva_muestra_grafico = nueva_muestra.copy()
    nueva_muestra_grafico["species_id"] = prediccion_id
    nueva_muestra_grafico["species"] = prediccion_nombre
    nueva_muestra_grafico["Tipo"] = "Nueva muestra"

    df_3d = pd.concat([df_grafico, nueva_muestra_grafico], ignore_index=True)

    fig_3d = px.scatter_3d(
        df_3d,
        x="petal length (cm)",
        y="petal width (cm)",
        z="sepal length (cm)",
        color="species",
        symbol="Tipo",
        title="Posición de la nueva muestra frente al dataset",
        opacity=0.78,
        labels={
            "petal length (cm)": "Longitud del pétalo",
            "petal width (cm)": "Ancho del pétalo",
            "sepal length (cm)": "Longitud del sépalo",
            "species": "Especie",
            "Tipo": "Tipo",
        },
    )
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=45, b=0),
        legend_title_text="Especie",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()


# =========================================================
# Visualizaciones adicionales
# =========================================================
st.subheader("Visualización y análisis del dataset")

pestana1, pestana2, pestana3, pestana4 = st.tabs(
    [
        "Distribución",
        "Relación entre variables",
        "Matriz de confusión",
        "Importancia de variables",
    ]
)

with pestana1:
    variable = st.selectbox("Seleccione una variable", iris.feature_names)

    fig_hist = px.histogram(
        df,
        x=variable,
        color="species",
        nbins=20,
        marginal="box",
        title=f"Distribución de {variable} por especie",
        labels={"species": "Especie"},
    )
    fig_hist.update_layout(
        height=520,
        bargap=0.08,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with pestana2:
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species",
        title="Matriz de dispersión de las variables",
        labels={"species": "Especie"},
    )
    fig_matrix.update_traces(diagonal_visible=False)
    fig_matrix.update_layout(height=720)
    st.plotly_chart(fig_matrix, use_container_width=True)

with pestana3:
    cm_df = pd.DataFrame(
        cm,
        index=iris.target_names,
        columns=iris.target_names,
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Matriz de confusión",
        labels=dict(x="Predicción", y="Valor real", color="Cantidad"),
    )
    fig_cm.update_layout(height=500)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("Reporte de clasificación por especie")
    reporte_df = pd.DataFrame(reporte).transpose().round(4)
    st.dataframe(reporte_df, use_container_width=True)

with pestana4:
    fig_imp = px.bar(
        importancias,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de las variables en el modelo Random Forest",
    )
    fig_imp.update_layout(
        height=500,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# Datos originales
# =========================================================
with st.expander("Ver registros del dataset"):
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("Proyecto final de Minería de Datos · Clasificación de Especies Iris")
