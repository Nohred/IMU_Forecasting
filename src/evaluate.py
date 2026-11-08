import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importes locales
from data_pipeline import get_dataloaders, load_scaler, FEATURE_COLS
from models import IMURNN, IMULSTM

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de Modelos de Forecasting IMU")
    parser.add_argument("--model_type", type=str, choices=['rnn', 'lstm'], default='lstm', help="Tipo de modelo a evaluar")
    parser.add_argument("--sequence", type=str, default="data/raw/V1_01_easy/imu0/data.csv", help="Ruta al CSV de prueba")
    parser.add_argument("--seq_length", type=int, default=50, help="Tamaño de la ventana histórica usada en entrenamiento")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensión oculta usada en entrenamiento")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas usadas en entrenamiento")
    parser.add_argument("--plot_samples", type=int, default=500, help="Número de muestras a graficar para claridad visual")
    return parser.parse_args()

def inverse_transform(data_scaled, scaler_params):
    """
    Revierte la normalización Z-score para calcular métricas y graficar
    en las unidades físicas reales (rad/s y m/s^2).
    """
    mean = np.array(scaler_params["mean"])
    std = np.array(scaler_params["std"])
    return (data_scaled * std) + mean

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en dispositivo: {device}")
    
    # 1. Cargar el DataLoader de Test (El split interno asigna el último 15% a test)
    print(f"Cargando datos de prueba desde: {args.sequence}...")
    _, _, test_loader = get_dataloaders(
        file_path=args.sequence, 
        seq_length=args.seq_length, 
        batch_size=256,
        split_ratios=(0.7, 0.15, 0.15)
    )
    
    # Cargar el scaler para revertir la normalización
    scaler_params = load_scaler("data/processed/scaler.json")
    
    # 2. Instanciar el modelo y cargar los pesos del checkpoint
    checkpoint_path = f"checkpoints/best_{args.model_type}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}. Entrena el modelo primero.")
        
    if args.model_type == 'rnn':
        model = IMURNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    else:
        model = IMULSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Modelo {args.model_type.upper()} cargado exitosamente.")
    
    # 3. Realizar predicciones sobre todo el conjunto de prueba
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            targets = y_batch.numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
            
    # Concatenar todos los batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 4. Deshacer la normalización para obtener valores físicos reales
    preds_real = inverse_transform(all_preds, scaler_params)
    targets_real = inverse_transform(all_targets, scaler_params)
    
    # 5. Calcular métricas
    print("\n" + "="*40)
    print(f"RESULTADOS DE EVALUACIÓN ({args.model_type.upper()})")
    print("="*40)
    
    mse_total = mean_squared_error(targets_real, preds_real)
    mae_total = mean_absolute_error(targets_real, preds_real)
    print(f"MSE Global: {mse_total:.4f}")
    print(f"MAE Global: {mae_total:.4f}\n")
    
    for i, feature in enumerate(FEATURE_COLS):
        mse_feat = mean_squared_error(targets_real[:, i], preds_real[:, i])
        mae_feat = mean_absolute_error(targets_real[:, i], preds_real[:, i])
        print(f"{feature:25s} -> MSE: {mse_feat:.4f} | MAE: {mae_feat:.4f}")
        
    # 6. Graficar resultados (solo un subconjunto para no saturar la imagen)
    plot_len = min(args.plot_samples, len(targets_real))
    time_axis = np.arange(plot_len) * (1/200.0) # Aproximación de 200Hz para el eje X en segundos
    
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f'Forecasting IMU - Ground Truth vs {args.model_type.upper()} Prediction', fontsize=16)
    
    for i, ax in enumerate(axes):
        ax.plot(time_axis, targets_real[:plot_len, i], label='Ground Truth', color='black', alpha=0.8)
        ax.plot(time_axis, preds_real[:plot_len, i], label='Prediction', color='red', linestyle='--', alpha=0.8)
        ax.set_ylabel(FEATURE_COLS[i])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Tiempo de prueba (s)')
    plt.tight_layout()
    
    # Guardar gráfica
    os.makedirs("data/processed", exist_ok=True)
    plot_path = f"data/processed/eval_plot_{args.model_type}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nGráfica comparativa guardada en: {plot_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()