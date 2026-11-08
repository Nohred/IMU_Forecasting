"""Integra señales IMU reales y predichas para visualizar trayectorias 2D.

Correcciones respecto a la versión anterior:
- La orientación usada para rotar la aceleración al marco global se toma del
  Ground Truth (interpolada con SLERP a los timestamps del IMU). Antes se
  integraba el giroscopio sin corrección de bias: el error de orientación
  acumulado proyectaba la gravedad en sentido horizontal y la doble
  integración divergía a kilómetros.
- Se corrigen los biases de giroscopio y acelerómetro con los reportados en
  el GT de EuRoC (columnas b_w y b_a), interpolados por muestra.
- Se advierte si el scaler se genera en el momento: debe coincidir con el
  usado en el entrenamiento o las predicciones no tienen sentido.
- La predicción sigue siendo recursiva (las salidas se realimentan), pero su
  error ya no se multiplica por una orientación divergente.

Uso:
    python trajectory_integration.py --model_type lstm
    python trajectory_integration.py --attitude strapdown   # método original
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation, Slerp

from data_pipeline import (
    FEATURE_COLS,
    apply_scaler,
    compute_and_save_scaler,
    load_scaler,
)
from models import IMULSTM, IMURNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integra señales IMU reales y predichas para visualizar una trayectoria 2D."
    )
    parser.add_argument("--model_type", choices=["rnn", "lstm"], default="lstm")
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
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Anula el hidden_dim detectado en el checkpoint.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Anula num_layers detectado en el checkpoint.",
    )
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument(
        "--attitude",
        choices=["gt", "strapdown"],
        default="gt",
        help=(
            "gt: orienta con el cuaternión del GT interpolado (recomendado, "
            "elimina la divergencia por bias del giroscopio). "
            "strapdown: integra el giroscopio como la versión anterior."
        ),
    )
    parser.add_argument(
        "--no_bias_correction",
        action="store_true",
        help="Desactiva la corrección de biases con los valores del GT.",
    )
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


def validate_input_paths(sequence_path, ground_truth_path):
    if not os.path.isfile(sequence_path):
        available = sorted(
            os.path.normpath(path)
            for path in [
                "data/raw/V1_01_easy/imu0/data.csv",
                "data/raw/V1_02_medium/imu0/data.csv",
                "data/raw/V1_03_difficult/imu0/data.csv",
            ]
            if os.path.isfile(path)
        )
        options = f" Secuencias disponibles: {', '.join(available)}" if available else ""
        raise FileNotFoundError(f"No existe la secuencia IMU: {sequence_path}.{options}")
    if not os.path.isfile(ground_truth_path):
        raise FileNotFoundError(
            f"No existe el archivo Ground Truth: {ground_truth_path}. "
            "Use --ground_truth para indicar su ubicación."
        )


def load_ground_truth(ground_truth_path):
    """Carga el GT completo de EuRoC: pose, velocidad y biases por muestra."""
    data = pd.read_csv(ground_truth_path, comment="#", header=None).to_numpy(
        dtype=np.float64
    )
    if data.shape[1] < 17:
        raise ValueError(
            "El CSV de Ground Truth no contiene las 17 columnas esperadas "
            "(pose, velocidad y biases)."
        )

    quaternions_xyzw = data[:, 4:8][:, [1, 2, 3, 0]]
    quaternions_xyzw /= np.maximum(
        np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True), 1e-12
    )
    gt = {
        "timestamps": data[:, 0],
        "positions": data[:, 1:4],
        "quaternions": quaternions_xyzw,
        "velocities": data[:, 8:11],
        "gyro_biases": data[:, 11:14],
        "accel_biases": data[:, 14:17],
    }

    # Descarta timestamps duplicados o no crecientes para que Slerp/interp funcionen.
    keep = np.concatenate([[True], np.diff(gt["timestamps"]) > 0])
    if not keep.all():
        gt = {key: value[keep] for key, value in gt.items()}
    return gt


def interpolate_ground_truth(gt, timestamps):
    """Interpola el GT a los timestamps del IMU (SLERP para orientación)."""
    gt_timestamps = gt["timestamps"]
    clipped = np.clip(timestamps, gt_timestamps[0], gt_timestamps[-1])

    def interp_columns(columns):
        return np.column_stack(
            [
                np.interp(clipped, gt_timestamps, columns[:, axis])
                for axis in range(columns.shape[1])
            ]
        )

    slerp = Slerp(gt_timestamps, Rotation.from_quat(gt["quaternions"]))
    return {
        "positions": interp_columns(gt["positions"]),
        "quaternions": slerp(clipped).as_quat(),
        "velocities": interp_columns(gt["velocities"]),
        "gyro_biases": interp_columns(gt["gyro_biases"]),
        "accel_biases": interp_columns(gt["accel_biases"]),
    }


def _expand_bias(bias, num_samples):
    """Normaliza un bias a forma (num_samples, 3); None significa cero."""
    if bias is None:
        return np.zeros((num_samples, 3), dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape == (3,):
        return np.tile(bias, (num_samples, 1))
    if bias.shape == (num_samples, 3):
        return bias
    raise ValueError("El bias debe tener forma (3,) o (muestras, 3).")


def integrate_imu(
    imu_signals,
    timestamps,
    initial_position,
    initial_velocity,
    initial_quaternion,
    gravity=9.81,
    gyro_biases=None,
    accel_biases=None,
    orientations_gt=None,
):
    """Integración Euler semi-implícita de la aceleración específica.

    orientations_gt: cuaterniones xyzw (muestras, 4) que dan la rotación
    cuerpo->global por muestra (interpolados del GT). Si se pasa, sustituye
    la integración del giroscopio y evita la divergencia por su bias.
    """
    signals = np.asarray(imu_signals, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != 6:
        raise ValueError("imu_signals debe tener forma (muestras, 6).")
    if len(signals) != len(timestamps) or len(signals) == 0:
        raise ValueError("Las señales IMU y timestamps deben tener la misma longitud no nula.")
    if orientations_gt is not None and len(orientations_gt) != len(signals):
        raise ValueError("orientations_gt debe tener la misma longitud que imu_signals.")

    num_samples = len(signals)
    gyro_biases = _expand_bias(gyro_biases, num_samples)
    accel_biases = _expand_bias(accel_biases, num_samples)

    positions = np.empty((num_samples, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    orientations = np.empty((num_samples, 4), dtype=np.float64)
    position = np.asarray(initial_position, dtype=np.float64).copy()
    velocity = np.asarray(initial_velocity, dtype=np.float64).copy()
    orientation = Rotation.from_quat(np.asarray(initial_quaternion, dtype=np.float64))

    timestamp_diffs = np.diff(timestamps)
    if len(timestamp_diffs) and np.any(timestamp_diffs <= 0):
        raise ValueError("Los timestamps deben ser estrictamente crecientes.")

    gravity_vector = np.array([0.0, 0.0, gravity], dtype=np.float64)

    positions[0] = position
    velocities[0] = velocity
    orientations[0] = orientation.as_quat()
    for index in range(1, num_samples):
        dt = (timestamps[index] - timestamps[index - 1]) / 1e9

        if orientations_gt is not None:
            orientation = Rotation.from_quat(orientations_gt[index])
        else:
            angular_rate = signals[index, :3] - gyro_biases[index]
            orientation = orientation * Rotation.from_rotvec(angular_rate * dt)

        acceleration_global = orientation.apply(
            signals[index, 3:] - accel_biases[index]
        )
        acceleration_linear = acceleration_global - gravity_vector

        velocity = velocity + acceleration_linear * dt
        position = position + velocity * dt
        positions[index] = position
        velocities[index] = velocity
        orientations[index] = orientation.as_quat()

    return positions, velocities, orientations


def load_model(model_type, hidden_dim, num_layers, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    recurrent_prefix = "rnn" if model_type == "rnn" else "lstm"
    weight_key = f"{recurrent_prefix}.weight_ih_l0"
    if weight_key not in state_dict:
        raise ValueError(
            f"El checkpoint {checkpoint_path} no corresponde a un modelo {model_type.upper()}."
        )
    checkpoint_hidden_dim = state_dict[f"{recurrent_prefix}.weight_hh_l0"].shape[1]
    checkpoint_num_layers = sum(
        key.startswith(f"{recurrent_prefix}.weight_ih_l") for key in state_dict
    )
    if hidden_dim is not None and hidden_dim != checkpoint_hidden_dim:
        raise ValueError(
            f"hidden_dim={hidden_dim} no coincide con el checkpoint "
            f"({checkpoint_hidden_dim}). Omita --hidden_dim para autodetectarlo."
        )
    if num_layers is not None and num_layers != checkpoint_num_layers:
        raise ValueError(
            f"num_layers={num_layers} no coincide con el checkpoint "
            f"({checkpoint_num_layers}). Omita --num_layers para autodetectarlo."
        )
    hidden_dim = checkpoint_hidden_dim
    num_layers = checkpoint_num_layers
    model_class = IMURNN if model_type == "rnn" else IMULSTM
    model = model_class(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, hidden_dim, num_layers


def forecast_test_signal(
    imu_real, test_start, seq_length, model, scaler_params, device
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
    axis.plot(
        real_estimate[-1, 0],
        real_estimate[-1, 1],
        marker="o",
        color="tab:blue",
        markersize=9,
        linestyle="none",
        label="Inicio de Test",
    )
    axis.set_title("Trayectoria 2D por integración de IMU")
    axis.set_xlabel("X global [m]")
    axis.set_ylabel("Y global [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    return figure


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ground_truth_path = args.ground_truth or infer_ground_truth_path(args.sequence)
    validate_input_paths(args.sequence, ground_truth_path)
    imu_frame = pd.read_csv(args.sequence)
    imu_timestamps = imu_frame["#timestamp [ns]"].to_numpy(dtype=np.float64)
    imu_real = imu_frame[FEATURE_COLS].to_numpy(dtype=np.float64)
    test_start = int(len(imu_real) * 0.85)

    gt = load_ground_truth(ground_truth_path)
    gt_at_imu = interpolate_ground_truth(gt, imu_timestamps)

    if args.no_bias_correction:
        gyro_biases = None
        accel_biases = None
    else:
        gyro_biases = gt_at_imu["gyro_biases"]
        accel_biases = gt_at_imu["accel_biases"]
    orientations_gt = (
        gt_at_imu["quaternions"] if args.attitude == "gt" else None
    )

    scaler_path = "data/processed/scaler.json"
    if not os.path.exists(scaler_path):
        print(
            "ADVERTENCIA: no existe data/processed/scaler.json y se generará con "
            "esta secuencia. Debe coincidir con el scaler usado en el "
            "entrenamiento del checkpoint; si difiere, las predicciones no "
            "tendrán sentido. Borre el archivo generado y coloque el original "
            "si es necesario."
        )
        compute_and_save_scaler(imu_real[: int(len(imu_real) * 0.7)], scaler_path)
    scaler_params = load_scaler(scaler_path)

    model, hidden_dim, num_layers = load_model(
        args.model_type,
        args.hidden_dim,
        args.num_layers,
        f"checkpoints/best_{args.model_type}.pth",
        device,
    )
    hybrid_signal, predictions = forecast_test_signal(
        imu_real, test_start, args.seq_length, model, scaler_params, device
    )

    real_estimated, real_velocities, real_orientations = integrate_imu(
        imu_real[:test_start],
        imu_timestamps[:test_start],
        gt_at_imu["positions"][0],
        gt_at_imu["velocities"][0],
        gt_at_imu["quaternions"][0],
        gravity=args.gravity,
        gyro_biases=gyro_biases[:test_start] if gyro_biases is not None else None,
        accel_biases=accel_biases[:test_start] if accel_biases is not None else None,
        orientations_gt=(
            orientations_gt[:test_start] if orientations_gt is not None else None
        ),
    )
    predicted_estimated, _, _ = integrate_imu(
        hybrid_signal[test_start - 1 :],
        imu_timestamps[test_start - 1 :],
        real_estimated[-1],
        real_velocities[-1],
        real_orientations[-1],
        gravity=args.gravity,
        gyro_biases=gyro_biases[test_start - 1 :] if gyro_biases is not None else None,
        accel_biases=(
            accel_biases[test_start - 1 :] if accel_biases is not None else None
        ),
        orientations_gt=(
            orientations_gt[test_start - 1 :]
            if orientations_gt is not None
            else None
        ),
    )

    output_path = args.output or f"data/processed/trajectory_{args.model_type}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_trajectories(
        gt_at_imu["positions"],
        real_estimated,
        predicted_estimated[1:],
        output_path,
    )

    error_real = np.linalg.norm(
        real_estimated[-1] - gt_at_imu["positions"][test_start - 1]
    )
    error_predicted = np.linalg.norm(
        predicted_estimated[-1] - gt_at_imu["positions"][-1]
    )
    print(
        f"Modelo {args.model_type.upper()} cargado en {device} "
        f"(hidden_dim={hidden_dim}, num_layers={num_layers})."
    )
    print(f"Actitud: {args.attitude}, corrección de bias: {not args.no_bias_correction}.")
    print(f"Predicciones generadas para {len(predictions) - test_start} muestras de Test.")
    print(f"Error final IMU real vs GT (fin de Train+Val): {error_real:.3f} m")
    print(f"Error final IMU predicha vs GT (fin de Test): {error_predicted:.3f} m")
    print(f"Trayectoria guardada en: {output_path}")


if __name__ == "__main__":
    main()