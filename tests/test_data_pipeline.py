import sys
import os
import json
import numpy as np
import pytest
import tempfile
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_pipeline import (
    compute_and_save_scaler, 
    apply_scaler, 
    create_sliding_windows,
    FEATURE_COLS
)

@pytest.fixture
def dummy_imu_data():
    """Genera 1000 filas de datos de IMU simulados."""
    np.random.seed(42)
    # 1000 muestras, 6 características (acelerómetro y giroscopio)
    return np.random.randn(1000, 6) * 5.0 + 2.0  # Media aprox 2.0, Std aprox 5.0

def test_scaler_logic(dummy_imu_data):
    """Valida que los datos normalizados tengan media 0 y std 1, y que el JSON se guarde bien."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        scaler_path = os.path.join(tmpdirname, "test_scaler.json")
        
        # Computar
        scaler_params = compute_and_save_scaler(dummy_imu_data, save_path=scaler_path)
        
        # Verificar que el archivo se creó
        assert os.path.exists(scaler_path), "El archivo JSON del scaler no se generó."
        
        # Verificar matemáticas de la normalización
        scaled_data = apply_scaler(dummy_imu_data, scaler_params)
        
        mean_scaled = np.mean(scaled_data, axis=0)
        std_scaled = np.std(scaled_data, axis=0)
        
        np.testing.assert_allclose(mean_scaled, 0.0, atol=1e-7, err_msg="La media escalada no es 0")
        np.testing.assert_allclose(std_scaled, 1.0, atol=1e-7, err_msg="La desviación estándar escalada no es 1")

def test_sliding_windows(dummy_imu_data):
    """Valida que la creación de secuencias 3D mantenga la integridad dimensional."""
    seq_length = 50
    total_samples = len(dummy_imu_data)
    features = dummy_imu_data.shape[1]
    
    X, y = create_sliding_windows(dummy_imu_data, seq_length)
    
    expected_samples = total_samples - seq_length
    
    # Forma de X: (950, 50, 6)
    assert X.shape == (expected_samples, seq_length, features), "Las dimensiones de la matriz X (histórico) son incorrectas."
    # Forma de Y: (950, 6)
    assert y.shape == (expected_samples, features), "Las dimensiones de la matriz Y (target) son incorrectas."
    
    # Validar integridad secuencial: 
    # El primer target Y debe ser igual al paso 50 de los datos originales
    np.testing.assert_array_equal(y[0], dummy_imu_data[seq_length], err_msg="Desfase temporal en las ventanas deslizantes.")
    # El final de la primera ventana de X debe ser el paso 49
    np.testing.assert_array_equal(X[0][-1], dummy_imu_data[seq_length - 1], err_msg="Desfase temporal dentro de la ventana X.")