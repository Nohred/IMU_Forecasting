import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Importes locales
from data_pipeline import get_dataloaders
from models import IMURNN, IMULSTM

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de Modelos de Forecasting IMU")
    parser.add_argument("--model_type", type=str, choices=['rnn', 'lstm'], default='lstm', help="Tipo de arquitectura (rnn o lstm)")
    parser.add_argument("--sequence", type=str, default="data/raw/V1_01_easy/imu0/data.csv", help="Ruta al CSV de la secuencia")
    parser.add_argument("--seq_length", type=int, default=50, help="Tamaño de la ventana histórica")
    parser.add_argument("--batch_size", type=int, default=256, help="Tamaño del batch")
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje (Learning Rate)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensión oculta de la red")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas recurrentes")
    parser.add_argument("--patience", type=int, default=10, help="Épocas de paciencia para Early Stopping")
    return parser.parse_args()

def export_to_onnx(model, model_type, seq_length, device):
    """
    Exporta el mejor modelo entrenado al formato ONNX requerido por Triton Inference Server.
    """
    model.eval()
    # Tensor de ejemplo (Batch=1, Seq_Len, Features=6)
    dummy_input = torch.randn(1, seq_length, 6).to(device)
    
    # Ruta estricta de Triton
    export_dir = f"model_repository/{model_type}_forecaster/1"
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "model.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"\n[MLOps] Modelo exportado a ONNX para Triton en: {onnx_path}")

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    # 1. Cargar DataLoaders
    print(f"Cargando datos desde: {args.sequence}...")
    train_loader, val_loader, _ = get_dataloaders(
        file_path=args.sequence, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size
    )
    
    # 2. Instanciar Modelo
    if args.model_type == 'rnn':
        model = IMURNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    else:
        model = IMULSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        
    print(f"Modelo seleccionado: {args.model_type.upper()}")
    
    # 3. Función de pérdida y optimizador
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Configurar Checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_{args.model_type}.pth")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 5. Loop de Entrenamiento
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 6. Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 7. Early Stopping y Guardado del Mejor Modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping disparado tras {epoch+1} épocas.")
                break
                
    # 8. Exportar para Producción (Triton)
    print("\nCargando los mejores pesos para exportación...")
    model.load_state_dict(torch.load(best_model_path))
    export_to_onnx(model, args.model_type, args.seq_length, device)

if __name__ == "__main__":
    train()