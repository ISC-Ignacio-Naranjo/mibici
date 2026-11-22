import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from mibici.api import app 

# 1. Crear un modelo simulado (Mock)
mock_model = MagicMock()
# Configurar el mock para que devuelva una predicción ficticia (ej. 15.5 viajes)
mock_model.predict.return_value = [15.5] 

# 2. Inyectar el modelo simulado en el estado de la aplicación ANTES de probar
app.state.model = mock_model

# 3. Inicializar el cliente de pruebas
client = TestClient(app)

def test_health_endpoint():
    """Prueba que el endpoint /health responda correctamente."""
    response = client.get("/health")
    assert response.status_code == 200

def test_ready_endpoint():
    """Prueba que el endpoint /ready verifique que el modelo está cargado."""
    # Ahora que inyectamos el mock_model, el endpoint /ready debería encontrarlo y regresar 200
    response = client.get("/ready")
    assert response.status_code == 200

def test_predict_valid_request():
    """Prueba una predicción exitosa enviando un payload correcto."""
    valid_payload = {
        "origen_id": "63",
        "hora": 8,
        "dia_semana": 1,
        "mes": 11,
        "demanda_lag_1h": 2.5,
        "demanda_lag_24h": 5.0,
        "es_fin_de_semana": 0
    }
    
    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()

    assert "predicted_demand" in data
    assert "model_uri" in data
    
    # Verificamos que el mock fue llamado
    mock_model.predict.assert_called_once()

def test_predict_invalid_request():
    """Prueba que la API valide correctamente los errores con Pydantic."""
    invalid_payload = {
        "origen_id": "63",
        "hora": 8
        # Faltan variables intencionalmente
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422