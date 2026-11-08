import sys
import os
import torch
import pytest

# Inyectar la carpeta src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models import IMURNN, IMULSTM

@pytest.fixture
def tensor_shapes():
    return {
        "batch_size": 32,
        "seq_length": 50,
        "input_dim": 6,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 6
    }

def test_imurnn_forward_pass(tensor_shapes):
    """Prueba que la RNN acepte las dimensiones correctas y retorne la forma esperada."""
    model = IMURNN(
        input_dim=tensor_shapes["input_dim"],
        hidden_dim=tensor_shapes["hidden_dim"],
        num_layers=tensor_shapes["num_layers"],
        output_dim=tensor_shapes["output_dim"]
    )
    
    # Crear un tensor de entrada simulado (Batch, Seq_Len, Features)
    dummy_input = torch.randn(
        tensor_shapes["batch_size"], 
        tensor_shapes["seq_length"], 
        tensor_shapes["input_dim"]
    )
    
    output = model(dummy_input)
    
    # La salida debe predecir 1 paso futuro para todo el batch (Batch, Features)
    expected_shape = (tensor_shapes["batch_size"], tensor_shapes["output_dim"])
    assert output.shape == expected_shape, f"Forma incorrecta en RNN. Esperado {expected_shape}, obtenido {output.shape}"
    assert not torch.isnan(output).any(), "La salida de la RNN contiene valores NaN."

def test_imulstm_forward_pass(tensor_shapes):
    """Prueba que la LSTM acepte las dimensiones correctas y retorne la forma esperada."""
    model = IMULSTM(
        input_dim=tensor_shapes["input_dim"],
        hidden_dim=tensor_shapes["hidden_dim"],
        num_layers=tensor_shapes["num_layers"],
        output_dim=tensor_shapes["output_dim"]
    )
    
    dummy_input = torch.randn(
        tensor_shapes["batch_size"], 
        tensor_shapes["seq_length"], 
        tensor_shapes["input_dim"]
    )
    
    output = model(dummy_input)
    
    expected_shape = (tensor_shapes["batch_size"], tensor_shapes["output_dim"])
    assert output.shape == expected_shape, f"Forma incorrecta en LSTM. Esperado {expected_shape}, obtenido {output.shape}"
    assert not torch.isnan(output).any(), "La salida de la LSTM contiene valores NaN."