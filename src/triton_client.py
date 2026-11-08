import argparse

import numpy as np
import tritonclient.http as httpclient

from data_pipeline import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envía una ventana IMU normalizada a Triton por HTTP."
    )
    parser.add_argument(
        "--model",
        choices=["lstm_forecaster", "rnn_forecaster"],
        default="lstm_forecaster",
        help="Nombre del modelo desplegado en Triton.",
    )
    parser.add_argument(
        "--sequence",
        default="data/raw/V1_01_easy/imu0/data.csv",
        help="CSV EuRoC usado para obtener la ventana de entrada.",
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="URL HTTP de Triton, sin el esquema http://.",
    )
    parser.add_argument("--seq-length", type=int, default=50)
    return parser.parse_args()


def get_input_batch(sequence_path, seq_length):
    train_loader, _, _ = get_dataloaders(
        file_path=sequence_path,
        seq_length=seq_length,
        batch_size=1,
        split_ratios=(0.7, 0.15, 0.15),
    )
    try:
        features, _ = next(iter(train_loader))
    except StopIteration as error:
        raise RuntimeError("La secuencia no contiene ventanas suficientes.") from error

    input_data = features[:1].numpy().astype(np.float32, copy=False)
    expected_shape = (1, seq_length, 6)
    if input_data.shape != expected_shape:
        raise RuntimeError(
            f"La entrada generada tiene forma {input_data.shape}; "
            f"se esperaba {expected_shape}."
        )
    return input_data


def infer(model_name, url, input_data):
    client = httpclient.InferenceServerClient(url=url)
    if not client.is_server_ready():
        raise RuntimeError(f"Triton no está listo en http://{url}.")

    infer_input = httpclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)
    requested_output = httpclient.InferRequestedOutput("output")
    response = client.infer(
        model_name=model_name,
        inputs=[infer_input],
        outputs=[requested_output],
    )
    prediction = response.as_numpy("output")
    if prediction is None or prediction.shape != (1, 6):
        raise RuntimeError(
            f"Triton devolvió una salida inválida: "
            f"{None if prediction is None else prediction.shape}; se esperaba (1, 6)."
        )
    return prediction


def main():
    args = parse_args()
    input_data = get_input_batch(args.sequence, args.seq_length)
    prediction = infer(args.model, args.url, input_data)
    print(f"Modelo: {args.model}")
    print(f"Entrada enviada: {input_data.shape} (normalizada)")
    print(f"Predicción recibida: {prediction.shape}")
    print(prediction[0])


if __name__ == "__main__":
    main()
