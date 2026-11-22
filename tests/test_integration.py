import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from mibici.features import generate_lag_features
from mibici.modeling.train import build_preprocessor, get_feature_groups

def test_generate_lag_features_logic():
    """
    Prueba Unitaria: Valida que la función calcule correctamente los lags de 1h y 24h.
    Utiliza 26 horas consecutivas y dos estaciones para verificar que no se mezclan.
    """
    # Estación 1: 26 horas consecutivas
    hours = pd.date_range(start="2026-01-01 00:00:00", periods=26, freq="h")
    df1 = pd.DataFrame({
        "origen_id": [1] * 26,
        "fecha": hours.normalize(),
        "hora": hours.hour,
        "demanda": range(1, 27)
    })
    
    # Estación 2: 26 horas consecutivas con valores distintos
    df2 = pd.DataFrame({
        "origen_id": [2] * 26,
        "fecha": hours.normalize(),
        "hora": hours.hour,
        "demanda": range(101, 127)
    })
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Ejecutar preprocesamiento
    result_df = generate_lag_features(df)
    
    # Validar que no está vacío
    assert not result_df.empty
    
    # Validar Estación 1 (fila 0 de result_df será la hora 24 tras el dropna)
    estacion1_res = result_df[result_df["origen_id"] == 1]
    assert estacion1_res.iloc[0]["demanda_lag_1h"] == 24
    assert estacion1_res.iloc[0]["demanda_lag_24h"] == 1
    
    # Validar Estación 2 (Comprueba que los lags no se cruzaron con la estación 1)
    estacion2_res = result_df[result_df["origen_id"] == 2]
    assert estacion2_res.iloc[0]["demanda_lag_1h"] == 124
    assert estacion2_res.iloc[0]["demanda_lag_24h"] == 101

def test_pipeline_integration_end_to_end():
    """
    Prueba de Integración: Valida el flujo datos -> preprocesamiento -> predicción -> métricas
    usando los componentes reales del proyecto (sin Mocks).
    """
    # 1. Datos sintéticos con las columnas requeridas por el modelo
    data = {
        "origen_id": [1, 1, 2, 2],
        "hora": [8, 9, 8, 9],
        "dia_semana": [1, 1, 1, 1],
        "mes": [11, 11, 11, 11],
        "demanda_lag_1h": [5.0, 10.0, 3.0, 8.0],
        "demanda_lag_24h": [20.0, 25.0, 15.0, 18.0],
        "es_fin_de_semana": [0, 0, 0, 0],
        "demanda": [12.0, 15.0, 8.0, 10.0]  # Variable objetivo (Target)
    }
    df = pd.DataFrame(data)
    
    # 2. Cargar configuraciones reales del proyecto
    cat_cols, num_cols, bin_cols, feature_cols = get_feature_groups()
    preprocessor = build_preprocessor(cat_cols, num_cols, bin_cols)
    
    X = df[feature_cols]
    y = df["demanda"]
    
    # 3. Construir y entrenar el modelo (Flujo completo real)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])
    
    model.fit(X, y)
    
    # 4. Predicción y cálculo de Métricas
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    # 5. Validaciones finales
    assert len(predictions) == len(y)
    assert rmse >= 0
