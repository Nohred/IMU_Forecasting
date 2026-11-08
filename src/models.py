import torch
import torch.nn as nn

class IMURNN(nn.Module):
    """
    Modelo basado en Redes Neuronales Recurrentes (RNN) para el forecasting de señales IMU.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=6):
        super(IMURNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layer: batch_first=True indica que los tensores de entrada 
        # tendrán la forma (batch_size, sequence_length, input_dim)
        self.rnn = nn.RNN(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            nonlinearity='tanh' # 'tanh' es más estable que 'relu' para series de tiempo sin normalización perfecta
        )
        
        # Capa completamente conectada para mapear el estado oculto a la predicción final
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # No inicializamos (h0) explícitamente para que PyTorch asuma ceros por defecto,
        # esto facilita la exportación a ONNX y el despliegue en Triton.
        out, _ = self.rnn(x)
        
        # Extraemos la salida correspondiente al último paso de tiempo (última fila de la secuencia)
        # out[:, -1, :] shape: (batch, hidden_dim)
        last_time_step_out = out[:, -1, :]
        
        # Pasamos el último estado oculto por la capa lineal
        # prediction shape: (batch, output_dim)
        prediction = self.fc(last_time_step_out)
        
        return prediction


class IMULSTM(nn.Module):
    """
    Modelo basado en Long Short-Term Memory (LSTM) para el forecasting de señales IMU.
    Mejora la captura de dependencias a largo plazo mitigando el desvanecimiento del gradiente.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=6):
        super(IMULSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        
        # Extraemos la salida del último paso de tiempo
        last_time_step_out = out[:, -1, :]
        
        # Pasamos por la capa lineal
        prediction = self.fc(last_time_step_out)
        
        return prediction

# ==========================================
# Bloque de prueba local rápida (Opcional)
# ==========================================
if __name__ == "__main__":
    # Parámetros simulados
    batch_size = 32
    seq_length = 50   # Ventana histórica (ej. 50 muestras previas)
    features = 6      # 3 de Acelerómetro + 3 de Giroscopio
    
    # Tensor de entrada aleatorio imitando un batch de DataLoader
    dummy_input = torch.randn(batch_size, seq_length, features)
    
    # Instanciar modelos
    rnn_model = IMURNN(input_dim=features, hidden_dim=128, num_layers=2, output_dim=features)
    lstm_model = IMULSTM(input_dim=features, hidden_dim=128, num_layers=2, output_dim=features)
    
    # Forward pass
    rnn_out = rnn_model(dummy_input)
    lstm_out = lstm_model(dummy_input)
    
    print(f"Forma de entrada: {dummy_input.shape} -> (batch, seq_len, features)")
    print(f"Salida RNN: {rnn_out.shape} -> (batch, features_predichas)")
    print(f"Salida LSTM: {lstm_out.shape} -> (batch, features_predichas)")