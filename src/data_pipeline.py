import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Las columnas de características en el dataset EuRoC (índices 1 al 6)
FEATURE_COLS = [
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
]


def load_imu_data(file_path):
    df = pd.read_csv(file_path)
    return df[FEATURE_COLS].values


def compute_and_save_scaler(train_data, save_path="data/processed/scaler.json"):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1e-6

    scaler_params = {"mean": mean.tolist(), "std": std.tolist()}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(scaler_params, f, indent=4)

    return scaler_params


def apply_scaler(data, scaler_params):
    mean = np.array(scaler_params["mean"])
    std = np.array(scaler_params["std"])
    return (data - mean) / std

def load_scaler(scaler_path="data/processed/scaler.json"):
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)
    return scaler_params


def create_sliding_windows(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    file_path,
    seq_length=50,
    batch_size=256,
    scaler_path="data/processed/scaler.json",
    split_ratios=(0.7, 0.15, 0.15),
):
    """
    Toma una única secuencia IMU, la divide cronológicamente (Train/Val/Test)
    y retorna los DataLoaders.
    """
    assert sum(split_ratios) == 1.0, "Las proporciones de split deben sumar 1.0"

    # 1. Cargar datos de la secuencia completa
    data_raw = load_imu_data(file_path)
    num_samples = len(data_raw)

    # 2. Calcular los índices de corte
    train_end = int(num_samples * split_ratios[0])
    val_end = train_end + int(num_samples * split_ratios[1])

    # 3. Dividir cronológicamente (Temporal Split)
    train_raw = data_raw[:train_end]
    val_raw = data_raw[train_end:val_end]
    test_raw = data_raw[val_end:]

    # 4. Computar scaler SOLO con Train Data y guardarlo
    scaler_params = compute_and_save_scaler(train_raw, scaler_path)

    # 5. Normalizar todos los conjuntos
    train_scaled = apply_scaler(train_raw, scaler_params)
    val_scaled = apply_scaler(val_raw, scaler_params)
    test_scaled = apply_scaler(test_raw, scaler_params)

    # 6. Crear ventanas deslizantes
    X_train, y_train = create_sliding_windows(train_scaled, seq_length)
    X_val, y_val = create_sliding_windows(val_scaled, seq_length)
    X_test, y_test = create_sliding_windows(test_scaled, seq_length)

    # 7. Inicializar Datasets y DataLoaders
    train_loader = DataLoader(
        IMUDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        IMUDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        IMUDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


# ==========================================
# Ejecución de validación del pipeline (Corregida)
# ==========================================
if __name__ == "__main__":
    # Ahora probamos el pipeline dividiendo una sola secuencia (ej. Easy)
    sequence_file = "data/raw/V1_01_easy/imu0/data.csv"

    if os.path.exists(sequence_file):
        print(f"Procesando secuencia: {sequence_file}")
        train_dl, val_dl, test_dl = get_dataloaders(
            sequence_file, seq_length=50, batch_size=256
        )

        print(f"Batches en Train: {len(train_dl)}")
        print(f"Batches en Val: {len(val_dl)}")
        print(f"Batches en Test: {len(test_dl)}")

        X_batch, y_batch = next(iter(train_dl))
        print(f"Forma de X (Batch, Seq_Len, Features): {X_batch.shape}")
        print(f"Forma de Y (Batch, Features): {y_batch.shape}")
