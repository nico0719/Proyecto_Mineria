# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - Iris Species Classification
Proyecto final de Minería de Datos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)


# =========================
# Estilos
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #274b8d;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-top: 0px;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #DDDDDD;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .prediction-box {
        background-color: #f4f7fb;
        padding: 18px;
        border-radius: 16px;
        border-left: 6px solid #274b8d;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Carga y preparación de datos
# =========================
@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species_id"] = iris.target
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, df


@st.cache_resource
def entrenar_modelo():
    iris, df = cargar_datos()

    df_model = df.drop_duplicates().reset_index(drop=True)
    X = df_model[iris.feature_names]
    y = df_model["species_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    importances = pd.DataFrame({
        "feature": iris.feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return iris, df, model, scaler, metrics, report, cm, importances


iris, df, model, scaler, metrics, report, cm, importances = entrenar_modelo()


# =========================
# Encabezado
# =========================
st.markdown('<p class="main-title">🌸 Iris Species Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Dashboard interactivo para visualizar el dataset Iris, evaluar el modelo y predecir la especie de una nueva flor.</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================
# Sidebar - Entrada de usuario
# =========================
st.sidebar.header("🔎 Predicción de una nueva flor")
st.sidebar.write("Ingresa las medidas morfológicas de la flor en centímetros.")

sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

new_sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)
new_sample_scaled = scaler.transform(new_sample)
prediction_id = model.predict(new_sample_scaled)[0]
prediction_proba = model.predict_proba(new_sample_scaled)[0]
prediction_name = iris.target_names[prediction_id]


# =========================
# Métricas principales
# =========================
st.subheader("📊 Métricas del modelo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Precision", f"{metrics['precision']:.2%}")
col3.metric("Recall", f"{metrics['recall']:.2%}")
col4.metric("F1-Score", f"{metrics['f1']:.2%}")

st.caption("Las métricas se calculan sobre el conjunto de prueba usando Random Forest.")


# =========================
# Predicción
# =========================
st.subheader("🌼 Resultado de la predicción")

left, right = st.columns([1, 2])

with left:
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Especie predicha:</h3>
        <h2 style="color:#274b8d;">{prediction_name}</h2>
        <p>El modelo clasificó la flor ingresada como <b>{prediction_name}</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    proba_df = pd.DataFrame({
        "species": iris.target_names,
        "probability": prediction_proba
    })
    fig_proba = px.bar(
        proba_df,
        x="species",
        y="probability",
        title="Probabilidad por especie",
        text_auto=".2%",
        labels={"species": "Especie", "probability": "Probabilidad"}
    )
    fig_proba.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_proba, use_container_width=True)

with right:
    df_plot = df.copy()
    df_plot["tipo"] = "Dataset"

    new_plot = new_sample.copy()
    new_plot["species"] = prediction_name
    new_plot["species_id"] = prediction_id
    new_plot["tipo"] = "Nueva muestra"

    df_3d = pd.concat([df_plot, new_plot], ignore_index=True)

    fig_3d = px.scatter_3d(
        df_3d,
        x="petal length (cm)",
        y="petal width (cm)",
        z="sepal length (cm)",
        color="species",
        symbol="tipo",
        title="Ubicación de la nueva muestra frente al dataset",
        opacity=0.75,
        labels={
            "petal length (cm)": "Petal length",
            "petal width (cm)": "Petal width",
            "sepal length (cm)": "Sepal length"
        }
    )
    fig_3d.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_3d, use_container_width=True)


# =========================
# Visualizaciones adicionales
# =========================
st.subheader("📈 Visualizaciones adicionales")

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribución de variables",
    "Matriz de dispersión",
    "Matriz de confusión",
    "Importancia de variables"
])

with tab1:
    feature_hist = st.selectbox("Selecciona una característica", iris.feature_names)
    fig_hist = px.histogram(
        df,
        x=feature_hist,
        color="species",
        nbins=20,
        marginal="box",
        title=f"Distribución de {feature_hist} por especie",
        labels={feature_hist: feature_hist, "species": "Especie"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species",
        title="Scatter matrix de las variables del dataset Iris"
    )
    fig_matrix.update_traces(diagonal_visible=False)
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab3:
    cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Matriz de confusión",
        labels=dict(x="Predicción", y="Valor real", color="Cantidad")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("Reporte de clasificación por especie:")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with tab4:
    fig_imp = px.bar(
        importances,
        x="importance",
        y="feature",
        orientation="h",
        title="Importancia de las variables en Random Forest",
        labels={"importance": "Importancia relativa", "feature": "Variable"}
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)


# =========================
# Dataset
# =========================
with st.expander("📄 Ver dataset Iris"):
    st.dataframe(df, use_container_width=True)

st.divider()
st.markdown("**Proyecto final de Minería de Datos — Iris Species Classification**")
