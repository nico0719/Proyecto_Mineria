# -*- coding: utf-8 -*-
"""
Iris Species Classification - Streamlit Dashboard
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Classification | Data Mining",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Estilos visuales
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary: #274b8d;
            --primary-dark: #1d3768;
            --bg-soft: #f5f7fb;
            --text-muted: #667085;
            --border: #e5e7eb;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
            border-right: 1px solid var(--border);
        }

        .hero {
            padding: 30px 34px;
            border-radius: 24px;
            background: linear-gradient(135deg, #ffffff 0%, #eef4ff 52%, #f8fbff 100%);
            border: 1px solid #d9e4f5;
            box-shadow: 0 14px 35px rgba(39,75,141,.09);
            margin-bottom: 1.2rem;
        }

        .eyebrow {
            color: var(--primary);
            font-size: .83rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .35rem;
        }

        .hero h1 {
            color: #101828;
            font-size: 2.45rem;
            line-height: 1.1;
            font-weight: 850;
            margin: 0;
        }

        .hero p {
            color: var(--text-muted);
            font-size: 1rem;
            max-width: 890px;
            margin-top: .75rem;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #182230;
            margin-top: .25rem;
            margin-bottom: .5rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(16,24,40,.06);
            min-height: 112px;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: .85rem;
            font-weight: 700;
            margin-bottom: .35rem;
        }

        .metric-value {
            color: var(--primary);
            font-size: 1.75rem;
            font-weight: 850;
            margin-bottom: .1rem;
        }

        .metric-note {
            color: #98a2b3;
            font-size: .77rem;
        }

        .prediction-card {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid #d9e4f5;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(39,75,141,.10);
        }

        .prediction-card h2 {
            color: var(--primary);
            font-size: 2rem;
            margin: .1rem 0 .3rem 0;
        }

        .prediction-tag {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: #eef4ff;
            color: var(--primary);
            font-weight: 800;
            font-size: .78rem;
            margin-bottom: .65rem;
        }

        .mini-note {
            color: var(--text-muted);
            font-size: .9rem;
        }

        .footer {
            color: #667085;
            font-size: .86rem;
            text-align: center;
            padding-top: 1rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            padding: 16px;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(16,24,40,.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Datos y entrenamiento
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    iris_data = load_iris()
    data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    data["species_id"] = iris_data.target
    data["species"] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    return iris_data, data


@st.cache_resource
def train_classifier():
    iris_data, data = load_dataset()

    clean_data = data.drop_duplicates().reset_index(drop=True)
    X = clean_data[iris_data.feature_names]
    y = clean_data["species_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    scaler_model = StandardScaler()
    X_train_scaled = scaler_model.fit_transform(X_train)
    X_test_scaled = scaler_model.transform(X_test)

    classifier = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train_scaled, y_train)

    predictions = classifier.predict(X_test_scaled)

    model_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted"),
        "Recall": recall_score(y_test, predictions, average="weighted"),
        "F1-Score": f1_score(y_test, predictions, average="weighted"),
    }

    model_report = classification_report(
        y_test,
        predictions,
        target_names=iris_data.target_names,
        output_dict=True,
    )

    matrix = confusion_matrix(y_test, predictions)

    feature_importance = pd.DataFrame(
        {
            "Variable": iris_data.feature_names,
            "Importancia": classifier.feature_importances_,
        }
    ).sort_values("Importancia", ascending=False)

    return (
        iris_data,
        data,
        classifier,
        scaler_model,
        model_metrics,
        model_report,
        matrix,
        feature_importance,
        X_train,
        X_test,
    )


(
    iris,
    df,
    model,
    scaler,
    metrics,
    report,
    cm,
    importances,
    X_train,
    X_test,
) = train_classifier()

# -----------------------------------------------------------------------------
# Sidebar: parámetros de entrada
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Panel de predicción")
st.sidebar.caption("Ingresa las medidas de una flor Iris para estimar su especie.")
st.sidebar.divider()

sepal_length = st.sidebar.slider("Longitud del sépalo (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.sidebar.slider("Ancho del sépalo (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.sidebar.slider("Longitud del pétalo (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.sidebar.slider("Ancho del pétalo (cm)", 0.1, 2.5, 0.2, 0.1)

sample = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names,
)

sample_scaled = scaler.transform(sample)
prediction_id = model.predict(sample_scaled)[0]
prediction_name = iris.target_names[prediction_id]
prediction_proba = model.predict_proba(sample_scaled)[0]
confidence = float(np.max(prediction_proba))

st.sidebar.divider()
st.sidebar.markdown("### Resumen de entrada")
st.sidebar.dataframe(sample.rename(columns={
    "sepal length (cm)": "Sepal length",
    "sepal width (cm)": "Sepal width",
    "petal length (cm)": "Petal length",
    "petal width (cm)": "Petal width",
}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Proyecto final · Minería de Datos</div>
        <h1>Iris Species Classification</h1>
        <p>
            Dashboard interactivo para explorar el dataset Iris, evaluar el desempeño del modelo
            y clasificar nuevas muestras a partir de las medidas del sépalo y pétalo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Desempeño general del modelo</div>', unsafe_allow_html=True)

metric_cols = st.columns(4)
for col, (name, value) in zip(metric_cols, metrics.items()):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{value:.2%}</div>
                <div class="metric-note">Calculado sobre el conjunto de prueba</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    f"Modelo utilizado: Random Forest Classifier · Registros del dataset: {df.shape[0]} · Variables predictoras: {len(iris.feature_names)}"
)

# -----------------------------------------------------------------------------
# Predicción y gráfico principal
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Predicción interactiva</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([0.95, 1.65], gap="large")

with left_col:
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-tag">Resultado del modelo</div>
            <div class="mini-note">Especie estimada</div>
            <h2>{prediction_name.title()}</h2>
            <p class="mini-note">
                La muestra ingresada fue clasificada como <b>{prediction_name.title()}</b>,
                con una confianza aproximada de <b>{confidence:.2%}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {
            "Especie": [name.title() for name in iris.target_names],
            "Probabilidad": prediction_proba,
        }
    ).sort_values("Probabilidad", ascending=True)

    fig_probability = px.bar(
        probability_df,
        x="Probabilidad",
        y="Especie",
        orientation="h",
        text="Probabilidad",
        title="Probabilidad estimada por especie",
        range_x=[0, 1],
    )
    fig_probability.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_probability.update_layout(
        height=305,
        margin=dict(l=10, r=30, t=55, b=10),
        xaxis_tickformat=".0%",
        showlegend=False,
    )
    st.plotly_chart(fig_probability, use_container_width=True)

with right_col:
    base_plot = df.copy()
    base_plot["Tipo de punto"] = "Dataset"

    sample_plot = sample.copy()
    sample_plot["species_id"] = prediction_id
    sample_plot["species"] = prediction_name
    sample_plot["Tipo de punto"] = "Nueva muestra"

    chart_data = pd.concat([base_plot, sample_plot], ignore_index=True)

    fig_3d = px.scatter_3d(
        chart_data,
        x="petal length (cm)",
        y="petal width (cm)",
        z="sepal length (cm)",
        color="species",
        symbol="Tipo de punto",
        title="Posición de la nueva muestra frente al dataset",
        opacity=0.82,
        labels={
            "petal length (cm)": "Longitud del pétalo",
            "petal width (cm)": "Ancho del pétalo",
            "sepal length (cm)": "Longitud del sépalo",
            "species": "Especie",
        },
    )
    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(
        height=520,
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# Análisis visual
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Exploración y explicación del modelo</div>', unsafe_allow_html=True)

eda_tab, matrix_tab, model_tab, data_tab = st.tabs(
    [
        "Distribuciones",
        "Relaciones entre variables",
        "Evaluación del modelo",
        "Datos",
    ]
)

with eda_tab:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        selected_feature = st.selectbox(
            "Variable a visualizar",
            iris.feature_names,
            index=2,
        )
        fig_hist = px.histogram(
            df,
            x=selected_feature,
            color="species",
            nbins=22,
            marginal="box",
            title="Distribución por especie",
            labels={selected_feature: selected_feature, "species": "Especie"},
        )
        fig_hist.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(
            df,
            x="species",
            y=selected_feature,
            points="all",
            title="Comparación de valores por especie",
            labels={"species": "Especie", selected_feature: selected_feature},
        )
        fig_box.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

with matrix_tab:
    st.caption(
        "Esta vista permite observar qué variables separan mejor las especies. En el dataset Iris, las medidas del pétalo suelen ser las más discriminantes."
    )
    fig_scatter_matrix = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species",
        title="Matriz de dispersión de variables",
        labels={col: col for col in iris.feature_names},
    )
    fig_scatter_matrix.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.72))
    fig_scatter_matrix.update_layout(height=760, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)

with model_tab:
    model_left, model_right = st.columns([1, 1])

    with model_left:
        confusion_df = pd.DataFrame(
            cm,
            index=[name.title() for name in iris.target_names],
            columns=[name.title() for name in iris.target_names],
        )
        fig_cm = px.imshow(
            confusion_df,
            text_auto=True,
            title="Matriz de confusión",
            labels=dict(x="Predicción", y="Valor real", color="Cantidad"),
        )
        fig_cm.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with model_right:
        fig_importance = px.bar(
            importances.sort_values("Importancia", ascending=True),
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia de variables",
            labels={"Importancia": "Importancia relativa", "Variable": "Variable"},
        )
        fig_importance.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("#### Reporte de clasificación")
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

with data_tab:
    st.markdown("#### Vista previa del dataset")
    st.dataframe(df, use_container_width=True, hide_index=True)

    summary = df.drop(columns=["species_id"]).groupby("species").agg(["mean", "std", "min", "max"]).round(2)
    st.markdown("#### Resumen estadístico por especie")
    st.dataframe(summary, use_container_width=True)

st.divider()
st.markdown(
    '<div class="footer">Proyecto final de Minería de Datos · Iris Species Classification · Dashboard desarrollado con Streamlit</div>',
    unsafe_allow_html=True,
)
