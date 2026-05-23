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
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .stApp {
            background-color: #F5F7FA;
        }

        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }

        h1, h2, h3 {
            color: #1F2937;
        }

        .section-note {
            color: #4B5563;
            font-size: 15px;
            line-height: 1.6;
        }

        .info-box {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.06);
        }

        .footer-text {
            color: #6B7280;
            font-size: 13px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
        "accuracy": accuracy_score(y_test, y_pred),
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


iris, df, model, scaler, metrics, report, cm, importances = preparar_modelo()


# =========================================================
# Encabezado
# =========================================================
st.title("Iris Species Classification")
st.markdown(
    """
    <p class="section-note">
    Aplicación interactiva para explorar el dataset Iris, evaluar el desempeño de un modelo
    de clasificación y realizar predicciones a partir de las medidas del sépalo y del pétalo.
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# Sidebar - parámetros de entrada
# =========================================================
st.sidebar.header("Datos de entrada")
st.sidebar.write("Ingrese las medidas de la flor en centímetros.")

sepal_length = st.sidebar.number_input(
    "Sepal length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1,
)

sepal_width = st.sidebar.number_input(
    "Sepal width (cm)",
    min_value=2.0,
    max_value=5.0,
    value=3.5,
    step=0.1,
)

petal_length = st.sidebar.number_input(
    "Petal length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1,
)

petal_width = st.sidebar.number_input(
    "Petal width (cm)",
    min_value=0.1,
    max_value=3.0,
    value=0.2,
    step=0.1,
)

nueva_muestra = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names,
)

nueva_muestra_scaled = scaler.transform(nueva_muestra)
prediction_id = model.predict(nueva_muestra_scaled)[0]
prediction_proba = model.predict_proba(nueva_muestra_scaled)[0]
prediction_name = iris.target_names[prediction_id]

st.sidebar.divider()
st.sidebar.subheader("Resumen de entrada")
st.sidebar.dataframe(nueva_muestra, use_container_width=True, hide_index=True)


# =========================================================
# Métricas del modelo
# =========================================================
st.subheader("Desempeño general del modelo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Precision", f"{metrics['precision']:.2%}")
col3.metric("Recall", f"{metrics['recall']:.2%}")
col4.metric("F1-Score", f"{metrics['f1']:.2%}")

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
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("#### Predicción obtenida")
    st.success(f"La especie predicha es: **{prediction_name}**")

    proba_df = pd.DataFrame(
        {
            "Especie": iris.target_names,
            "Probabilidad": prediction_proba,
        }
    )

    fig_proba = px.bar(
        proba_df,
        x="Especie",
        y="Probabilidad",
        text_auto=".1%",
        labels={"Probabilidad": "Probabilidad estimada"},
    )
    fig_proba.update_layout(
        template="plotly_white",
        height=330,
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_proba, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_grafico:
    df_grafico = df.copy()
    df_grafico["Tipo"] = "Dataset"

    nueva_muestra_grafico = nueva_muestra.copy()
    nueva_muestra_grafico["species_id"] = prediction_id
    nueva_muestra_grafico["species"] = prediction_name
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
            "petal length (cm)": "Petal length",
            "petal width (cm)": "Petal width",
            "sepal length (cm)": "Sepal length",
            "species": "Especie",
        },
    )
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(
        template="plotly_white",
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
        template="plotly_white",
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
    fig_matrix.update_layout(
        template="plotly_white",
        height=720,
    )
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
    fig_cm.update_layout(
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("Reporte de clasificación por especie")
    reporte_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(reporte_df, use_container_width=True)

with pestana4:
    fig_imp = px.bar(
        importances,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de las variables en el modelo Random Forest",
    )
    fig_imp.update_layout(
        template="plotly_white",
        height=500,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# Datos originales
# =========================================================
with st.expander("Ver registros del dataset"):
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <p class="footer-text">
    Proyecto final de Minería de Datos · Iris Species Classification
    </p>
    """,
    unsafe_allow_html=True,
)
