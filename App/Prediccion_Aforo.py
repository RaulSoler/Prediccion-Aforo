import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
try:
    import joblib
except ImportError:
    import os
    os.system("pip install joblib")
    import joblib


modelos = joblib.load("stacking_manual_aforo.pkl")

# Función para preparar el dataframe de entrada igual que antes (misma que definimos)
def preparar_input(fecha, clima_str, temp_min, temp_max, vuelos, cruceros, festivos_sel):
    clima_opciones = {'Soleado':0, 'Parcialmente nublado':1, 'Nublado':2, 'Lluvia':3}
    clima = clima_opciones[clima_str]
    dia_semana = fecha.weekday() + 1
    mes = fecha.month
    semana = fecha.isocalendar()[1]

    festivo_nacional = 1 if 'Nacional' in festivos_sel else 0
    festivo_valencia = 1 if 'Valencia' in festivos_sel else 0
    festivo_madrid = 1 if 'Madrid' in festivos_sel else 0
    festivo_andalucia = 1 if 'Andalucia' in festivos_sel else 0
    festivo_cataluna = 1 if 'Cataluña' in festivos_sel else 0

    X_pred = pd.DataFrame({
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'clima': [clima], 
        'vuelos': [vuelos],
        'cruceros': [cruceros],
        'dia_semana': [dia_semana],
        'mes': [mes],
        'semana': [semana],
        'festivo_valencia': [festivo_valencia],
        'festivo_nacional': [festivo_nacional],
        'festivo_andalucia': [festivo_andalucia],
        'festivo_madrid': [festivo_madrid],
        'festivo_cataluna': [festivo_cataluna]
    })

    # Convertimos las columnas categóricas a tipo 'category' 
    cat_cols = ['mes', 'dia_semana', 'semana']
    for col in cat_cols:
        X_pred[col] = X_pred[col].astype('category')

    return X_pred

st.title("Predicción de aforo")

# Inputs usuario
fecha = st.date_input("Selecciona una fecha", value=date.today())
clima_str = st.selectbox('Que clima hace', ['Soleado', 'Parcialmente nublado', 'Nublado', 'Lluvia'])
temp_min = st.slider('Temperatura mínima', -10.0, 45.0, 0.0, 0.5)
temp_max = st.slider('Temperatura máxima', -10.0, 45.0, 0.0, 0.5)
vuelos = st.number_input("Vuelos", 0, 300, 0)
cruceros = st.number_input("Cruceros", 0, 120, 0)
festivos_sel = st.multiselect('Donde es festivo', ['Nacional', 'Valencia', 'Madrid', 'Andalucia', 'Cataluña'])

if st.button("Predecir aforo"):
    X_nuevo = preparar_input(fecha, clima_str, temp_min, temp_max, vuelos, cruceros, festivos_sel)

    # Predicciones individuales
    pred_xgb = modelos["modelo_xgb"].predict(X_nuevo)
    pred_svr = modelos["modelo_svr"].predict(X_nuevo)
    pred_lgbm = modelos["modelo_lgbm"].predict(X_nuevo)

    # Meta-predicción
    X_meta = np.column_stack((pred_xgb, pred_svr, pred_lgbm))
    y_pred = modelos["meta_model"].predict(X_meta)

    st.success(f"Predicción de aforo: {np.round(y_pred[0], 2)} personas")
