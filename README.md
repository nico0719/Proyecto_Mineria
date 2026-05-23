# Iris Species Classification Dashboard

Proyecto final de Minería de Datos basado en el dataset Iris.

## Objetivo

Entrenar un modelo de clasificación capaz de predecir la especie de una flor Iris a partir de cuatro mediciones:

- Sepal length
- Sepal width
- Petal length
- Petal width

El dashboard fue desarrollado con Streamlit e incluye:

- Métricas del modelo: Accuracy, Precision, Recall y F1-Score.
- Panel interactivo para ingresar las medidas de una nueva flor.
- Predicción automática de la especie.
- Gráfico 3D con la nueva muestra frente al dataset.
- Histogramas, matriz de dispersión, matriz de confusión e importancia de variables.

## Modelo utilizado

Se utiliza Random Forest Classifier, debido a que es un algoritmo robusto para clasificación, funciona bien con datasets pequeños y permite interpretar la importancia de cada variable.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run Proyect.py
```

## Integrantes

- Jorge Eliecer De La Hoz Epiayu
- Agregar aquí los demás integrantes del grupo
