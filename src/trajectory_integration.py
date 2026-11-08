import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from data_pipeline import (
    FEATURE_COLS,
    apply_scaler,
    compute_and_save_scaler,
    load_imu_data,
    load_scaler,
)
from models import IMULSTM, IMURNN

GT_POSITION_COLS = [
    " p_RS_R_x [m]",
    " p_RS_R_y [m]",
    " p_RS_R_z [m]",
]
GT_QUATERNION_COLS = [" q_RS_w []", " q_RS_x []", " q_RS_y []", " q_RS_z []"]
GT_VELOCITY_COLS = [
    " v_RS_R_x [m s^-1]",
    " v_RS_R_y [m s^-1]",
    " v_RS_R_z [m s^-1]",
]
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integra señales IMU reales y predichas para visualizar una trayectoria 2D."
    )
    parser.add_argument(
        "--model_type", choices=["rnn", "lstm"], default="lstm"
    )
    parser.add_argument(
        "--sequence",
        default="data/raw/V1_01_easy/imu0/data.csv",
        help="CSV de la secuencia IMU.",
    )
    parser.add_argument(
        "--ground_truth",
        default=None,
        help="CSV de ground truth. Por defecto se infiere desde --sequence.",
    )
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de la figura. Por defecto data/processed/trajectory_<model>.png.",
    )
    return parser.parse_args()


def infer_ground_truth_path(sequence_path):
    sequence = os.path.abspath(sequence_path)
    marker = os.path.join("imu0", "data.csv")
    if marker not in sequence:
        raise ValueError(
            "--ground_truth es obligatorio cuando --sequence no está en una carpeta imu0."
        )
    return sequence.replace(
        marker, os.path.join("state_groundtruth_estimate0", "data.csv")
    )


def load_ground_truth(ground_truth_path, imu_timestamps):
    ground_truth = pd.read_csv(ground_truth_path)
    timestamps = ground_truth["#timestamp"].to_numpy(dtype=np.float64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]

    def interpolate_columns(columns):
        values = ground_truth[columns].to_numpy(dtype=np.float64)[order]
        return np.column_stack(
            [np.interp(imu_timestamps, timestamps, values[:, i]) for i in range(values.shape[1])]
        )

    positions = interpolate_columns(GT_POSITION_COLS)
    velocities = interpolate_columns(GT_VELOCITY_COLS)
    quaternion_wxyz = interpolate_columns(GT_QUATERNION_COLS)
    quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
    quaternion_xyzw /= np.linalg.norm(
        quaternion_xyzw, axis=1, keepdims=True
    ).clip(min=1e-12)
    return positions, velocities, quaternion_xyzw


def integrate_imu(
    imu_signals,
    timestamps,
    initial_position,
    initial_velocity,
    initial_quaternion,
    gravity=9.81,
):
    """Integrate gyro and accelerometer samples with a first-order Euler method."""
    signals = np.asarray(imu_signals, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != 6:
        raise ValueError("imu_signals debe tener forma (muestras, 6).")
    if len(signals) != len(timestamps) or len(signals) == 0:
        raise ValueError("Las señales IMU y timestamps deben tener la misma longitud no nula.")

    positions = np.empty((len(signals), 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    orientations = np.empty((len(signals), 4), dtype=np.float64)
    position = np.asarray(initial_position, dtype=np.float64).copy()
    velocity = np.asarray(initial_velocity, dtype=np.float64).copy()
    orientation = Rotation.from_quat(np.asarray(initial_quaternion, dtype=np.float64))
    default_dt = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 0.005
    if default_dt <= 0:
        raise ValueError("Los timestamps deben ser estrictamente crecientes.")

    for index, signal in enumerate(signals):
        if index:
            dt = (timestamps[index] - timestamps[index - 1]) / 1e9
            if dt <= 0:
                raise ValueError("Los timestamps deben ser estrictamente crecientes.")
        else:
            dt = default_dt / 1e9 if timestamps[1] - timestamps[0] > 1 else default_dt

        if index:
            orientation = orientation * Rotation.from_rotvec(signal[:3] * dt)
        acceleration_global = orientation.apply(signal[3:]) + np.array(
            [0.0, 0.0, -gravity]
        )
        position += velocity * dt + 0.5 * acceleration_global * dt**2
        velocity += acceleration_global * dt
        positions[index] = position
        velocities[index] = velocity
        orientations[index] = orientation.as_quat()

    return positions, velocities, orientations


def load_model(model_type, hidden_dim, num_layers, checkpoint_path, device):
    model_class = IMURNN if model_type == "rnn" else IMULSTM
    model = model_class(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def forecast_test_signal(
    imu_real,
    test_start,
    seq_length,
    model,
    scaler_params,
    device,
):
    """Forecast every Test sample recursively, using prior predictions as context."""
    scaled_signal = apply_scaler(imu_real, scaler_params).astype(np.float32)
    hybrid_scaled = scaled_signal.copy()
    predictions = np.empty_like(imu_real, dtype=np.float64)

    with torch.no_grad():
        for index in range(test_start, len(imu_real)):
            window_start = index - seq_length
            if window_start < 0:
                raise ValueError("seq_length excede el historial disponible antes de Test.")
            window = torch.from_numpy(hybrid_scaled[window_start:index]).unsqueeze(0)
            prediction_scaled = model(window.to(device)).cpu().numpy()[0]
            prediction = (
                prediction_scaled * np.asarray(scaler_params["std"])
                + np.asarray(scaler_params["mean"])
            )
            predictions[index] = prediction
            hybrid_scaled[index] = prediction_scaled

    hybrid_signal = imu_real.copy()
    hybrid_signal[test_start:] = predictions[test_start:]
    return hybrid_signal, predictions


def plot_trajectories(ground_truth, real_estimate, predicted_estimate, output_path):
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        color="black",
        alpha=0.75,
        label="Ground Truth",
    )
    axis.plot(
        real_estimate[:, 0],
        real_estimate[:, 1],
        color="tab:blue",
        label="IMU real (Train + Val)",
    )
    axis.plot(
        predicted_estimate[:, 0],
        predicted_estimate[:, 1],
        color="tab:red",
        linestyle=":",
        linewidth=2,
        label="IMU predicha (Test)",
    )
    axis.set_title("Trayectoria 2D por integración Strapdown")
    axis.set_xlabel("X global [m]")
    axis.set_ylabel("Y global [m]")
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    return figure


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imu_frame = pd.read_csv(args.sequence)
    imu_timestamps = imu_frame["#timestamp [ns]"].to_numpy(dtype=np.float64)
    imu_real = imu_frame[FEATURE_COLS].to_numpy(dtype=np.float64)
    test_start = int(len(imu_real) * 0.85)

    ground_truth_path = args.ground_truth or infer_ground_truth_path(args.sequence)
    ground_truth, initial_velocities, initial_orientations = load_ground_truth(
        ground_truth_path, imu_timestamps
    )
    scaler_path = "data/processed/scaler.json"
    if not os.path.exists(scaler_path):
        compute_and_save_scaler(imu_real[: int(len(imu_real) * 0.7)], scaler_path)
    scaler_params = load_scaler(scaler_path)

    model = load_model(
        args.model_type,
        args.hidden_dim,
        args.num_layers,
        f"checkpoints/best_{args.model_type}.pth",
        device,
    )
    hybrid_signal, predictions = forecast_test_signal(
        imu_real, test_start, args.seq_length, model, scaler_params, device
    )
    initial_position = ground_truth[0]
    initial_velocity = initial_velocities[0]
    initial_quaternion = initial_orientations[0]
    estimated, _, _ = integrate_imu(
        hybrid_signal,
        imu_timestamps,
        initial_position,
        initial_velocity,
        initial_quaternion,
        gravity=args.gravity,
    )
    real_estimated, _, _ = integrate_imu(
        imu_real,
        imu_timestamps,
        initial_position,
        initial_velocity,
        initial_quaternion,
        gravity=args.gravity,
    )
    output_path = args.output or f"data/processed/trajectory_{args.model_type}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_trajectories(
        ground_truth,
        real_estimated[:test_start],
        estimated[test_start:],
        output_path,
    )
    print(f"Modelo {args.model_type.upper()} cargado en {device}.")
    print(f"Predicciones generadas para {len(predictions) - test_start} muestras de Test.")
    print(f"Trayectoria guardada en: {output_path}")


if __name__ == "__main__":
    main()
